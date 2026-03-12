"""Activation extraction using PyTorch forward hooks.

Loads the target model, runs all dataset examples in batches, and saves
per-layer hidden states to disk.

Saved artefacts
---------------
  {output_dir}/
    layer_{i:02d}.pt   – activations for layer i,
                          shape (n, hidden) for "last"/"mean"/"answer_token"/"evidence_spans",
                          shape (n, seq_len, hidden) for "all" strategy
    labels.pt          – int64 labels, shape (n,)
    metadata.json      – model id, token_strategy, n_examples, etc.

token_strategy options
----------------------
"last"            – last non-padding token (default; good for SAD stages_oversight)
"mean"            – mean over non-padding tokens
"all"             – all positions padded to max_length (needed for ema/attention probes)
"answer_token"    – position of a specific answer token per example, identified by
                    character offset stored in the dataset field `answer_token_char`.
                    Required for Simple Contrastive Dataset (arXiv:2507.01786).
"evidence_spans"  – mean-pool over token positions covered by evidence character
                    spans (list of [start_char, end_char] per example). Falls back
                    to last token if no spans are available for an example.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def _evidence_char_spans_to_token_masks(
    tokenizer,
    texts: list[str],
    char_spans_batch: list[list[list[int]]],
    enc,
) -> torch.Tensor:
    """Convert per-example evidence character spans to a boolean token mask.

    Returns a (bs, seq_len) boolean mask where True marks tokens covered by
    any evidence span.  If an example has no valid spans, falls back to
    marking only the last non-padding token.
    """
    enc_with_offsets = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=enc["input_ids"].shape[1],
        return_tensors="pt",
        return_offsets_mapping=True,
    )
    offset_mapping = enc_with_offsets["offset_mapping"]  # (bs, seq_len, 2)
    bs, seq_len = offset_mapping.shape[:2]
    mask = torch.zeros(bs, seq_len, dtype=torch.bool)

    for b in range(bs):
        spans = char_spans_batch[b]
        for span_start, span_end in spans:
            for tok_idx, (tc_start, tc_end) in enumerate(offset_mapping[b].tolist()):
                if tc_end == 0:
                    continue  # skip special/padding tokens
                # Token overlaps span if token_start < span_end and token_end > span_start
                if tc_start < span_end and tc_end > span_start:
                    mask[b, tok_idx] = True
        # Fallback: if no tokens were marked, use last non-padding token
        if not mask[b].any():
            last_pos = enc["attention_mask"][b].sum().item() - 1
            mask[b, last_pos] = True

    return mask


def _char_offsets_to_token_positions(
    tokenizer,
    texts: list[str],
    char_offsets: list[int],
    enc,
) -> torch.Tensor:
    """Convert per-example character offsets to token positions.

    Uses the tokenizer's offset_mapping to find which token covers the
    character at `char_offset` in each text.  Falls back to last token
    if the offset falls outside the truncated sequence.
    """
    enc_with_offsets = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=enc["input_ids"].shape[1],
        return_tensors="pt",
        return_offsets_mapping=True,
    )
    offset_mapping = enc_with_offsets["offset_mapping"]  # (bs, seq_len, 2)
    bs = len(texts)
    positions = torch.zeros(bs, dtype=torch.long)

    for b in range(bs):
        target_char = char_offsets[b]
        found = False
        for tok_idx, (start_c, end_c) in enumerate(offset_mapping[b].tolist()):
            if start_c <= target_char < end_c:
                positions[b] = tok_idx
                found = True
                break
        if not found:
            positions[b] = enc["attention_mask"][b].sum().item() - 1

    return positions


def _get_layer_list(model, layer_path: str):
    """Follow dot-separated attribute path to the layer ModuleList."""
    obj = model
    for attr in layer_path.split("."):
        obj = getattr(obj, attr)
    return obj


def extract_activations(
    model_id: str,
    texts: list[str],
    labels: list[int],
    output_dir: str | Path,
    num_layers: int,
    layer_path: str = "model.layers",
    layer_output_index: int | None = 0,
    token_strategy: str = "last",
    batch_size: int = 8,
    max_length: int = 2048,
    skip_if_exists: bool = True,
    answer_token_chars: list[int] | None = None,
    evidence_char_spans: list[list[list[int]]] | None = None,
) -> None:
    """Extract and save hidden states for all layers."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    meta_path = output_dir / "metadata.json"
    if skip_if_exists and meta_path.exists():
        print(f"[extraction] Already done — {output_dir}. Skipping.")
        return

    assert len(texts) == len(labels), "texts and labels must have equal length."
    if token_strategy == "answer_token":
        assert answer_token_chars is not None and len(answer_token_chars) == len(texts), (
            "answer_token_chars must be provided (one char offset per example) "
            "when token_strategy='answer_token'."
        )
    if token_strategy == "evidence_spans":
        assert evidence_char_spans is not None and len(evidence_char_spans) == len(texts), (
            "evidence_char_spans must be provided (list of [start, end] per example) "
            "when token_strategy='evidence_spans'."
        )

    print(f"[extraction] Loading model {model_id} …")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Get the layer ModuleList and register forward hooks.
    layers = _get_layer_list(model, layer_path)
    captured: dict[int, torch.Tensor] = {}

    def _make_hook(layer_idx: int):
        def hook(_module, _input, output):
            # output is either a tensor or a tuple (hidden_states, ...).
            if layer_output_index is not None:
                hidden = output[layer_output_index]
            else:
                hidden = output if isinstance(output, torch.Tensor) else output[0]
            captured[layer_idx] = hidden.detach()
        return hook

    hooks = []
    for i in range(num_layers):
        hooks.append(layers[i].register_forward_hook(_make_hook(i)))

    n = len(texts)

    # Incremental saving: write each layer to a temporary file and append
    # batch results via a list of tensors.  After the loop, concatenate once
    # per layer and save the final file.  For models with many layers (64+)
    # this avoids keeping all layers × all batches in RAM simultaneously.
    # We process one layer-group at a time: accumulate batch tensors per layer,
    # but flush completed layers to disk periodically.
    all_acts: dict[int, list[torch.Tensor]] = {i: [] for i in range(num_layers)}
    hidden_size: int | None = None

    print(
        f"[extraction] Extracting {num_layers} layers, "
        f"token_strategy={token_strategy!r}, n={n}"
    )

    # Determine the input device for multi-GPU models (device_map="auto").
    # For sharded models, the first embedding layer is on the input device.
    if hasattr(model, "hf_device_map"):
        # Transformers places inputs on the device of the first module.
        first_device = min(set(model.hf_device_map.values()), key=lambda d: 0 if d == "cpu" else int(str(d).replace("cuda:", "")))
        device = torch.device(first_device)
    else:
        device = next(model.parameters()).device

    try:
        for start in tqdm(range(0, n, batch_size), desc="batches"):
            batch_texts = texts[start : start + batch_size]
            bs = len(batch_texts)

            # Use max_length padding for "all" strategy so all batches have
            # the same seq_len dimension; otherwise pad to longest in batch.
            pad_strategy = "max_length" if token_strategy == "all" else True
            enc = tokenizer(
                batch_texts,
                padding=pad_strategy,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )

            seq_lengths = enc["attention_mask"].sum(dim=1).long()
            last_positions = seq_lengths - 1

            if token_strategy == "answer_token":
                answer_positions = _char_offsets_to_token_positions(
                    tokenizer, batch_texts,
                    answer_token_chars[start : start + bs], enc,
                )

            if token_strategy == "evidence_spans":
                evidence_mask = _evidence_char_spans_to_token_masks(
                    tokenizer, batch_texts,
                    evidence_char_spans[start : start + bs], enc,
                )

            inputs = {k: v.to(device) for k, v in enc.items()}
            with torch.no_grad():
                model(**inputs)

            # Collect captured activations (move to CPU immediately).
            for i in range(num_layers):
                act = captured[i].float().cpu()  # (bs, seq_len, hidden_size)

                if token_strategy == "last":
                    act = act[torch.arange(bs), last_positions, :]
                elif token_strategy == "mean":
                    mask = enc["attention_mask"].unsqueeze(-1).float()
                    act = (act * mask).sum(1) / mask.sum(1)
                elif token_strategy == "answer_token":
                    act = act[torch.arange(bs), answer_positions, :]
                elif token_strategy == "evidence_spans":
                    # Mean-pool over evidence tokens per example
                    emask = evidence_mask.unsqueeze(-1).float()  # (bs, seq_len, 1)
                    act = (act * emask).sum(1) / emask.sum(1).clamp(min=1)
                # else "all": keep (bs, seq_len, hidden)

                if hidden_size is None and act.dim() >= 2:
                    hidden_size = act.shape[-1]

                all_acts[i].append(act)

            captured.clear()
    finally:
        for h in hooks:
            h.remove()

    print("[extraction] Saving …")
    for i in range(num_layers):
        layer_acts = torch.cat(all_acts[i], dim=0)
        torch.save(layer_acts, output_dir / f"layer_{i:02d}.pt")
        del all_acts[i]  # free memory as we go

    torch.save(torch.tensor(labels, dtype=torch.long), output_dir / "labels.pt")

    with open(meta_path, "w") as f:
        json.dump(
            {
                "model_id": model_id,
                "num_layers": num_layers,
                "n_examples": n,
                "token_strategy": token_strategy,
                "max_length": max_length,
                "layer_path": layer_path,
                "layer_output_index": layer_output_index,
                "hidden_size": hidden_size,
            },
            f,
            indent=2,
        )

    print(f"[extraction] Done → {output_dir}")


def load_activations(
    activations_dir: str | Path, layer: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load activations and labels for one layer."""
    d = Path(activations_dir)
    acts = torch.load(d / f"layer_{layer:02d}.pt", weights_only=True)
    labels = torch.load(d / "labels.pt", weights_only=True)
    return acts, labels


def load_metadata(activations_dir: str | Path) -> dict:
    with open(Path(activations_dir) / "metadata.json") as f:
        return json.load(f)
