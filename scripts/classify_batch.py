"""Classify multiple prompts for eval-awareness using all trained probes.

Loads the model once, generates CoT responses for each prompt, extracts
activations at the last token of the generated response, then runs every
trained probe on each.

Usage (SLURM):
    uv run python scripts/classify_batch.py \
        --prompts-file prompts.json \
        --run-dir logs/20260310_173530_olmo3-7b-think \
        --slurm
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from probes.architectures import EMAProbe, build_probe, SEQUENCE_PROBES


def extract_prompts(
    model_id: str,
    prompts: list[str],
    num_layers: int,
    layer_path: str = "model.layers",
    layer_output_index: int | None = None,
    max_new_tokens: int = 32768,
    generate_cot: bool = True,
) -> list[dict[int, dict[str, torch.Tensor]]]:
    """Extract last-token activations, optionally after generating CoT."""
    print(f"[classify] Loading model {model_id} ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    obj = model
    for attr in layer_path.split("."):
        obj = getattr(obj, attr)
    layers = obj

    captured: dict[int, torch.Tensor] = {}

    def _make_hook(idx: int):
        def hook(_module, _input, output):
            if layer_output_index is not None:
                hidden = output[layer_output_index]
            else:
                hidden = output if isinstance(output, torch.Tensor) else output[0]
            captured[idx] = hidden.detach()
        return hook

    device = next(model.parameters()).device
    all_activations = []
    generated_texts = []

    for pi, prompt in enumerate(prompts):
        # Format with chat template.
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

        if generate_cot:
            # Generate full CoT response, then extract at last generated token.
            enc = tokenizer(
                [formatted], padding=True, truncation=True,
                max_length=max_new_tokens, return_tensors="pt",
            )
            prompt_len = enc["input_ids"].shape[1]
            inputs = {k: v.to(device) for k, v in enc.items()}

            print(f"[classify] [{pi+1}/{len(prompts)}] Generating CoT: {prompt[:60]}...")
            with torch.no_grad():
                gen_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )

            response_text = tokenizer.decode(gen_ids[0][prompt_len:], skip_special_tokens=False)
            gen_len = gen_ids.shape[1]
            generated_texts.append(response_text)
            print(f"           Generated {gen_len - prompt_len} tokens (total seq: {gen_len})")
            print(f"           Response: {response_text[:200]}...")

            forward_ids = gen_ids
        else:
            # Extract at last token of the formatted prompt (no generation).
            enc = tokenizer(
                [formatted], padding=True, truncation=True,
                max_length=max_new_tokens, return_tensors="pt",
            )
            gen_len = enc["input_ids"].shape[1]
            generated_texts.append("")
            print(f"[classify] [{pi+1}/{len(prompts)}] Prompt only ({gen_len} tokens): {prompt[:60]}...")

            forward_ids = enc["input_ids"]

        # Forward pass with hooks.
        hooks = []
        for i in range(num_layers):
            hooks.append(layers[i].register_forward_hook(_make_hook(i)))

        with torch.no_grad():
            model(forward_ids.to(device))

        for h in hooks:
            h.remove()

        # Extract last-token activation.
        last_pos = gen_len - 1
        activations = {}
        for i in range(num_layers):
            act = captured[i].float().cpu()
            last_act = act[:, last_pos, :]  # (1, hidden)
            activations[i] = {
                "last": last_act,
                "all": last_act.unsqueeze(1),  # (1, 1, hidden)
            }
        captured.clear()
        all_activations.append(activations)

    del model
    torch.cuda.empty_cache()

    return all_activations, generated_texts


def load_and_run_probe(
    probe_type: str,
    checkpoint_path: Path,
    metrics_path: Path,
    activation: dict[str, torch.Tensor],
    hidden_size: int,
) -> dict:
    ckpt = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
    metrics = json.loads(metrics_path.read_text())
    threshold = metrics["threshold"]

    if ckpt["type"] == "mean_diff":
        direction = ckpt["direction"]
        thresh = ckpt["threshold"]
        act_np = activation["last"].numpy()
        score = float(act_np @ direction)
        pred = int(score >= thresh)
        return {"score": score, "threshold": thresh, "prediction": pred}

    probe = build_probe(probe_type, hidden_size)
    probe.load_state_dict(ckpt["state_dict"])
    probe.eval()

    x = activation["all"] if probe_type in SEQUENCE_PROBES else activation["last"]

    with torch.no_grad():
        if isinstance(probe, EMAProbe) and x.dim() == 3:
            logit = probe.forward_ema(x)
        else:
            logit = probe(x)
        prob = torch.sigmoid(logit).item()

    pred = int(prob >= threshold)
    return {"score": prob, "threshold": threshold, "prediction": pred}


def classify_batch(prompts: list[str], run_dir: str, output_path: str | None = None, generate_cot: bool = True) -> dict:
    run_dir = Path(run_dir)
    import yaml
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())

    model_id = cfg["model"]["id"]
    num_layers = cfg["model"]["num_layers"]
    hidden_size = cfg["model"]["hidden_size"]
    layer_path = cfg["model"].get("layer_path", "model.layers")
    layer_output_index = cfg["model"].get("layer_output_index")
    results_dir = run_dir / "results"

    all_activations, generated_texts = extract_prompts(
        model_id, prompts, num_layers, layer_path, layer_output_index,
        generate_cot=generate_cot,
    )

    # Discover probes once.
    probe_specs = []
    for layer_dir in sorted(results_dir.iterdir()):
        if not layer_dir.is_dir() or not layer_dir.name.startswith("layer_"):
            continue
        layer_idx = int(layer_dir.name.split("_")[1])
        for probe_dir in sorted(layer_dir.iterdir()):
            if not probe_dir.is_dir():
                continue
            ckpt = probe_dir / "probe.pt"
            met = probe_dir / "metrics.json"
            if ckpt.exists() and met.exists():
                probe_specs.append((layer_idx, probe_dir.name, ckpt, met))

    print(f"\n[classify] Running {len(probe_specs)} probes on {len(prompts)} prompts ...\n")

    all_results = []
    for pi, prompt in enumerate(prompts):
        vote_eval = 0
        vote_deploy = 0
        probe_results = []

        for layer_idx, probe_type, ckpt_path, metrics_path in probe_specs:
            try:
                r = load_and_run_probe(
                    probe_type, ckpt_path, metrics_path,
                    all_activations[pi][layer_idx], hidden_size,
                )
                r["layer"] = layer_idx
                r["probe_type"] = probe_type
                probe_results.append(r)
                if r["prediction"] == 1:
                    vote_eval += 1
                else:
                    vote_deploy += 1
            except Exception as e:
                print(f"  WARN: {probe_type} layer {layer_idx}: {e}")

        total = vote_eval + vote_deploy
        frac = vote_eval / total if total > 0 else 0
        all_results.append({
            "prompt": prompt,
            "generated_response": generated_texts[pi],
            "vote_eval": vote_eval,
            "vote_deploy": vote_deploy,
            "total_probes": total,
            "eval_fraction": frac,
            "verdict": "EVAL-AWARE" if vote_eval > vote_deploy else "NOT EVAL-AWARE",
            "probes": probe_results,
        })

    # Print summary table.
    print("\n" + "=" * 90)
    print(f"{'#':>3}  {'EVAL':>5}  {'DEPLOY':>7}  {'FRAC':>7}  {'VERDICT':>15}  PROMPT")
    print("-" * 90)
    for i, r in enumerate(all_results):
        short = r["prompt"][:45] + ("..." if len(r["prompt"]) > 45 else "")
        print(f"{i+1:>3}  {r['vote_eval']:>5}  {r['vote_deploy']:>7}  {r['eval_fraction']:>6.1%}  {r['verdict']:>15}  {short}")
    print("=" * 90)

    output = {
        "model": model_id,
        "run_dir": str(run_dir),
        "results": all_results,
    }

    out_path = Path(output_path) if output_path else run_dir / "classify_batch_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nFull results saved to: {out_path}")

    return output


def submit_slurm(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    import yaml
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    slurm_cfg = cfg.get("slurm", {})

    partition = slurm_cfg.get("partition", "compute")
    timeout = slurm_cfg.get("extract_timeout", "4:00:00")

    output_path = args.output or str(run_dir / "classify_batch_results.json")
    log_path = run_dir / "classify_batch_slurm.log"

    no_cot_flag = " \\\n    --no-cot" if args.no_cot else ""
    script = f"""#!/bin/bash
#SBATCH --job-name=probe-classify-batch
#SBATCH --partition={partition}
#SBATCH --gres=gpu:1
#SBATCH --time={timeout}
#SBATCH --output={log_path}
#SBATCH --error={log_path}

cd {PROJECT_ROOT}
uv run python scripts/classify_batch.py \\
    --prompts-file {repr(str(args.prompts_file))} \\
    --run-dir {repr(str(args.run_dir))} \\
    --output {repr(output_path)}{no_cot_flag}
"""

    script_path = run_dir / "classify_batch_job.sh"
    script_path.write_text(script)
    print(f"[classify] Submitting SLURM job ...")
    result = subprocess.run(["sbatch", str(script_path)], capture_output=True, text=True)
    print(result.stdout.strip())
    if result.returncode != 0:
        print(f"SLURM error: {result.stderr}", file=sys.stderr)
        sys.exit(1)
    print(f"Log: {log_path}")


def main():
    parser = argparse.ArgumentParser(description="Classify prompts for eval-awareness (batch)")
    parser.add_argument("--prompts-file", required=True, help="JSON file with list of prompt strings")
    parser.add_argument("--run-dir", default="logs/20260310_173530_olmo3-7b-think")
    parser.add_argument("--output", help="Output JSON path")
    parser.add_argument("--slurm", action="store_true", help="Submit as SLURM job")
    parser.add_argument("--no-cot", action="store_true", help="Extract at end of prompt instead of after CoT generation")
    args = parser.parse_args()

    if args.slurm:
        submit_slurm(args)
    else:
        prompts = json.loads(Path(args.prompts_file).read_text())
        classify_batch(prompts, args.run_dir, args.output, generate_cot=not args.no_cot)


if __name__ == "__main__":
    main()
