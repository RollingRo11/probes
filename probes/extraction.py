"""Activation extraction using NNsight 0.6.

Loads the target model via NNsight's LanguageModel, runs all dataset examples
in batches, and saves per-layer hidden states to disk.

NNsight 0.6 API notes
---------------------
- `LanguageModel(model_id, device_map="auto", dispatch=True)` loads eagerly.
- Inside `model.trace(inputs)`, model attributes are proxies.
- `.output[0]` is the hidden-state tensor for Llama/OLMo decoder layers
  (first element of the `(hidden_states, *rest)` output tuple).
- Call `.save()` on any proxy you want to read after the context exits;
  access the concrete tensor via `.value`.
- NNsight 0.6 supports a vLLM backend for remote/distributed execution
  (same tracing API, add `backend="vllm"` or `remote=True` for NDIF).

Saved artefacts
---------------
  {output_dir}/
    layer_{i:02d}.pt   – activations for layer i,
                          shape (n, hidden) for "last"/"mean"/"answer_token" strategy,
                          shape (n, seq_len, hidden) for "all" strategy
    labels.pt          – int64 labels, shape (n,)
    metadata.json      – model id, token_strategy, n_examples, etc.

token_strategy options
----------------------
"last"          – last non-padding token (default; good for SAD stages_oversight)
"mean"          – mean over non-padding tokens
"all"           – all positions padded to max_length (needed for ema/attention probes)
"answer_token"  – position of a specific answer token per example, identified by
                  character offset stored in the dataset field `answer_token_char`.
                  Required for Simple Contrastive Dataset (arXiv:2507.01786).
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoTokenizer


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
    # Re-tokenize with offset_mapping (can't get it from the already-padded enc).
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
            # Fall back to last real token.
            positions[b] = enc["attention_mask"][b].sum().item() - 1

    return positions


def _navigate(proxy, path: str):
    """Follow dot-separated attribute path on an NNsight proxy or plain module."""
    obj = proxy
    for attr in path.split("."):
        obj = getattr(obj, attr)
    return obj


def extract_activations(
    model_id: str,
    texts: list[str],
    labels: list[int],
    output_dir: str | Path,
    num_layers: int,
    layer_path: str = "model.model.layers",
    layer_output_index: int = 0,
    token_strategy: str = "last",
    batch_size: int = 8,
    max_length: int = 2048,
    skip_if_exists: bool = True,
    answer_token_chars: list[int] | None = None,
) -> None:
    """Extract and save hidden states for all layers.

    Parameters
    ----------
    model_id          : HuggingFace model id.
    texts             : list of input strings (already chat-formatted if needed).
    labels            : int label per example (same length as texts).
    output_dir        : directory to write tensors and metadata.
    num_layers        : number of transformer layers to extract.
    layer_path        : dot-separated attribute path to the layer ModuleList,
                        e.g. "model.model.layers" for OLMo/Llama models.
    layer_output_index: which element of the layer's output tuple is hidden states.
    token_strategy    : "last" | "mean" | "all" | "answer_token"
                          last         → last non-padding token
                          mean         → mean over non-padding tokens
                          all          → all positions (padded to max_length)
                          answer_token → position of a specific answer token per example,
                                         located via character offset in answer_token_chars.
                                         Required for the Simple Contrastive Dataset.
    batch_size        : examples per forward pass.
    max_length        : tokenizer truncation length.
    skip_if_exists    : if True, skip extraction when metadata.json already exists.
    answer_token_chars: character offsets (one per example) pointing to the answer token
                        within each text. Required when token_strategy="answer_token".
    """
    from nnsight import LanguageModel  # imported here to keep startup fast

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

    print(f"[extraction] Loading model {model_id} via NNsight …")
    model = LanguageModel(model_id, device_map="auto", dispatch=True)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    n = len(texts)
    all_acts: dict[int, list[torch.Tensor]] = {i: [] for i in range(num_layers)}

    print(
        f"[extraction] Extracting {num_layers} layers, "
        f"token_strategy={token_strategy!r}, n={n}"
    )

    for start in tqdm(range(0, n, batch_size), desc="batches"):
        batch_texts = texts[start : start + batch_size]
        bs = len(batch_texts)

        enc = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        seq_lengths = enc["attention_mask"].sum(dim=1).long()   # (bs,)
        last_positions = seq_lengths - 1                        # (bs,)

        # For answer_token strategy: find the token position corresponding to the
        # character offset of the answer token (e.g. "(A)" or "(B)") in each text.
        if token_strategy == "answer_token":
            answer_positions = _char_offsets_to_token_positions(
                tokenizer,
                batch_texts,
                answer_token_chars[start : start + bs],
                enc,
            )  # (bs,)

        # ----- NNsight trace -----
        with model.trace(enc):
            layer_list = _navigate(model, layer_path)
            saved = [
                layer_list[i].output[layer_output_index].save()
                for i in range(num_layers)
            ]
        # ----- collect -----
        for i, s in enumerate(saved):
            act = s.value.float().cpu()   # (bs, seq_len, hidden_size)

            if token_strategy == "last":
                act = act[torch.arange(bs), last_positions, :]  # (bs, hidden)
            elif token_strategy == "mean":
                mask = enc["attention_mask"].unsqueeze(-1).float()  # (bs, seq, 1)
                act = (act * mask).sum(1) / mask.sum(1)             # (bs, hidden)
            elif token_strategy == "answer_token":
                act = act[torch.arange(bs), answer_positions, :]    # (bs, hidden)
            # else "all": keep (bs, seq_len, hidden)

            all_acts[i].append(act)

    print("[extraction] Saving …")
    for i in range(num_layers):
        layer_acts = torch.cat(all_acts[i], dim=0)
        torch.save(layer_acts, output_dir / f"layer_{i:02d}.pt")

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
                "hidden_size": all_acts[0][0].shape[-1],
            },
            f,
            indent=2,
        )

    print(f"[extraction] Done → {output_dir}")


def load_activations(
    activations_dir: str | Path, layer: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load activations and labels for one layer.

    Returns
    -------
    acts   : (n, hidden) or (n, seq_len, hidden)
    labels : (n,) int64
    """
    d = Path(activations_dir)
    acts = torch.load(d / f"layer_{layer:02d}.pt", weights_only=True)
    labels = torch.load(d / "labels.pt", weights_only=True)
    return acts, labels


def load_metadata(activations_dir: str | Path) -> dict:
    with open(Path(activations_dir) / "metadata.json") as f:
        return json.load(f)
