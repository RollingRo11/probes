#!/usr/bin/env python
"""Train linear and nonlinear probes on LLM hidden states.

Three execution modes, controlled by --mode:

  Orchestrator (default, no --mode):
    Reads the YAML config, submits one SLURM extraction job, waits for it to
    complete, then submits one SLURM training job per (layer, probe_type) pair,
    waits for all to finish, and aggregates results.

  extract (--mode extract):
    Runs on a SLURM-allocated GPU node. Loads the model via NNsight, extracts
    hidden states for every configured layer, and writes activations to disk.

  train (--mode train --layer N --probe-type P):
    Runs on a SLURM-allocated node. Loads pre-extracted activations for layer N,
    trains probe type P, and writes results JSON + checkpoint.

Usage examples
--------------
  # Local run (no SLURM) — set --no-slurm to run everything in-process:
  uv run python train.py --config configs/default.yaml --no-slurm

  # Orchestrator (submits SLURM jobs):
  uv run python train.py --config configs/default.yaml

  # Manual worker invocations (called by sbatch scripts):
  uv run python train.py --config configs/default.yaml --mode extract --log-dir logs/run1
  uv run python train.py --config configs/default.yaml --mode train --layer 16 --probe-type linear --log-dir logs/run1
"""

import argparse
import asyncio
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv

load_dotenv()

REPO_DIR = Path(__file__).parent.resolve()


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_layers(config: dict) -> list[int]:
    """Expand config['probes']['layers'] to a concrete list of layer indices."""
    spec = config["probes"].get("layers", "all")
    n = config["model"]["num_layers"]
    if spec == "all":
        return list(range(n))
    if spec == "mid":
        return [n // 2]
    if isinstance(spec, list):
        return [int(i) for i in spec]
    raise ValueError(f"probes.layers must be 'all', 'mid', or a list, got {spec!r}")


def activations_dir(config: dict, log_dir: Path) -> Path:
    short = config["model"]["short_name"]
    return log_dir / "activations" / short


def results_dir(log_dir: Path) -> Path:
    return log_dir / "results"


def slurm_log_dir(log_dir: Path) -> Path:
    d = log_dir / "slurm_logs"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def load_dataset(config: dict) -> tuple[list[str], list[int], list[int] | None]:
    """Load (texts, labels, answer_token_chars) from the configured JSONL file.

    answer_token_chars is only populated when entries have an "answer_token_char"
    field (Simple Contrastive Dataset). Otherwise returns None, meaning the
    extraction will use the configured token_strategy without per-example offsets.
    """
    import json as _json
    path = Path(config["dataset"]["path"])
    text_col = config["dataset"].get("text_col", "text")
    label_col = config["dataset"].get("label_col", "label")

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. "
            f"Run `uv run python scripts/generate_data.py` to create it."
        )

    texts, labels, atc = [], [], []
    has_atc = False
    with open(path) as f:
        for line in f:
            row = _json.loads(line)
            texts.append(row[text_col])
            labels.append(int(row[label_col]))
            if "answer_token_char" in row:
                atc.append(int(row["answer_token_char"]))
                has_atc = True
            else:
                atc.append(-1)

    return texts, labels, (atc if has_atc else None)


def apply_chat_template(config: dict, texts: list[str]) -> list[str]:
    """Wrap raw text in the model's chat template."""
    from transformers import AutoTokenizer
    model_id = config["model"]["id"]
    tok = AutoTokenizer.from_pretrained(model_id)
    out = []
    for t in texts:
        formatted = tok.apply_chat_template(
            [{"role": "user", "content": t}],
            tokenize=False,
            add_generation_prompt=True,
        )
        out.append(formatted)
    return out


# ---------------------------------------------------------------------------
# Worker: extract
# ---------------------------------------------------------------------------


def run_extract(config: dict, log_dir: Path) -> None:
    """Extract activations for all layers and save to disk."""
    from probes.extraction import extract_activations

    texts, labels, answer_token_chars = load_dataset(config)
    print(f"[extract] {len(texts)} examples loaded.")

    ext_cfg = config.get("extraction", {})
    token_strategy = ext_cfg.get("token_strategy", "last")

    # Auto-switch to answer_token strategy when the dataset contains per-example
    # character offsets (Simple Contrastive Dataset).
    if answer_token_chars is not None and token_strategy not in ("answer_token", "all"):
        print(
            "[extract] Dataset contains answer_token_char fields; "
            "switching token_strategy to 'answer_token'."
        )
        token_strategy = "answer_token"

    # Chat-template formatting is NOT applied for simple contrastive entries:
    # the MCQ prompt is already fully specified.  Apply only when no offsets present.
    if answer_token_chars is None:
        formatted = apply_chat_template(config, texts)
    else:
        formatted = texts  # keep raw MCQ text so char offsets remain valid

    model_cfg = config["model"]
    out_dir = activations_dir(config, log_dir)

    extract_activations(
        model_id=model_cfg["id"],
        texts=formatted,
        labels=labels,
        output_dir=out_dir,
        num_layers=model_cfg["num_layers"],
        layer_path=model_cfg.get("layer_path", "model.model.layers"),
        layer_output_index=model_cfg.get("layer_output_index", 0),
        token_strategy=token_strategy,
        batch_size=ext_cfg.get("batch_size", 8),
        max_length=ext_cfg.get("max_length", 2048),
        answer_token_chars=answer_token_chars,
    )


# ---------------------------------------------------------------------------
# Worker: train one (layer, probe_type)
# ---------------------------------------------------------------------------


def run_train(config: dict, log_dir: Path, layer: int, probe_type: str) -> None:
    """Train and evaluate one (layer, probe_type) combination."""
    import torch
    from probes.extraction import load_activations
    from probes.trainer import save_probe, save_results, train_and_evaluate

    acts_dir = activations_dir(config, log_dir)
    out_dir = results_dir(log_dir) / f"layer_{layer:02d}" / probe_type
    out_dir.mkdir(parents=True, exist_ok=True)

    result_path = out_dir / "metrics.json"
    if result_path.exists():
        print(f"[train] Already done: layer={layer} probe={probe_type}. Skipping.")
        return

    print(f"[train] layer={layer}  probe={probe_type}")
    acts, labels = load_activations(acts_dir, layer)

    probe_cfg = config.get("probes", {})
    training_cfg = config.get("training", {})
    eval_cfg = config.get("evaluation", {})
    dataset_cfg = config.get("dataset", {})

    probe_kwargs = {
        "mlp_hidden_dim": probe_cfg.get("mlp_hidden_dim", 100),
        "mlp_n_layers": probe_cfg.get("mlp_n_layers", 2),
        "n_heads": probe_cfg.get("n_heads", 10),
        "rolling_window": probe_cfg.get("rolling_window", 10),
        "alpha": probe_cfg.get("ema_alpha", 0.5),
    }

    t0 = __import__("time").time()
    metrics, probe = train_and_evaluate(
        probe_type=probe_type,
        activations=acts,
        labels=labels,
        training_cfg=training_cfg,
        eval_cfg=eval_cfg,
        dataset_cfg=dataset_cfg,
        probe_kwargs=probe_kwargs,
    )
    elapsed = __import__("time").time() - t0

    metrics["layer"] = layer
    metrics["probe_type"] = probe_type
    metrics["elapsed_s"] = round(elapsed, 2)

    save_results(metrics, result_path)
    save_probe(probe, out_dir / "probe.pt")

    print(
        f"[train] layer={layer} probe={probe_type} "
        f"auroc={metrics.get('auroc', float('nan')):.4f} "
        f"weighted_error={metrics.get('weighted_error', float('nan')):.4f}"
    )


# ---------------------------------------------------------------------------
# Analysis: aggregate results
# ---------------------------------------------------------------------------


def analyze(config: dict, log_dir: Path) -> None:
    """Collect all results JSON files and write summary CSV."""
    import csv

    res_dir = results_dir(log_dir)
    rows = []

    for metrics_file in sorted(res_dir.glob("layer_*/**/metrics.json")):
        with open(metrics_file) as f:
            rows.append(json.load(f))

    if not rows:
        print("[analyze] No results found.")
        return

    csv_path = log_dir / "summary.csv"
    fieldnames = sorted({k for r in rows for k in r if not isinstance(r[k], list)})
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{'='*60}")
    print("Probe Results (AUROC by layer and type)")
    print("=" * 60)
    header = f"{'Layer':>6}  {'Probe':>16}  {'AUROC':>8}  {'Weighted Err':>14}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r.get('layer', '?'):>6}  {r.get('probe_type', '?'):>16}  "
            f"{r.get('auroc', float('nan')):>8.4f}  {r.get('weighted_error', float('nan')):>14.4f}"
        )
    print(f"\nSaved → {csv_path}")


# ---------------------------------------------------------------------------
# SLURM script writers
# ---------------------------------------------------------------------------


def _write_extract_script(config: dict, config_path: Path, log_dir: Path) -> Path:
    slurm = config.get("slurm", {})
    short = config["model"]["short_name"]
    job_name = slurm.get("job_name", "probes")
    partition = slurm.get("partition", "compute")
    timeout = slurm.get("extract_timeout", "4:00:00")
    gpus = slurm.get("extract_gpus", 1)
    cpus_per_node = slurm.get("cpus_per_node", 64)
    gpus_per_node = slurm.get("gpus_per_node", 8)
    cpus = max(1, cpus_per_node // gpus_per_node * gpus)

    sld = slurm_log_dir(log_dir)

    script = f"""#!/bin/bash
#SBATCH --job-name={job_name}_extract_{short}
#SBATCH --nodes=1
#SBATCH --gpus-per-node={gpus}
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --partition={partition}
#SBATCH --time={timeout}
#SBATCH --output={sld}/extract.out
#SBATCH --error={sld}/extract.err

set -euo pipefail
cd {REPO_DIR}

uv run python train.py \\
    --config '{config_path}' \\
    --mode extract \\
    --log-dir '{log_dir}'
"""
    fd, path = tempfile.mkstemp(suffix=".sh", prefix="extract_", dir=sld)
    with os.fdopen(fd, "w") as f:
        f.write(script)
    return Path(path)


def _write_train_script(
    layer: int,
    probe_type: str,
    config: dict,
    config_path: Path,
    log_dir: Path,
) -> Path:
    slurm = config.get("slurm", {})
    short = config["model"]["short_name"]
    job_name = slurm.get("job_name", "probes")
    partition = slurm.get("partition", "compute")
    timeout = slurm.get("train_timeout", "1:00:00")
    gpus = slurm.get("train_gpus", 1)
    cpus_per_node = slurm.get("cpus_per_node", 64)
    gpus_per_node = slurm.get("gpus_per_node", 8)
    cpus = max(1, cpus_per_node // gpus_per_node * gpus)

    sld = slurm_log_dir(log_dir)

    script = f"""#!/bin/bash
#SBATCH --job-name={job_name}_{short}_l{layer:02d}_{probe_type}
#SBATCH --nodes=1
#SBATCH --gpus-per-node={gpus}
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --partition={partition}
#SBATCH --time={timeout}
#SBATCH --output={sld}/train_l{layer:02d}_{probe_type}.out
#SBATCH --error={sld}/train_l{layer:02d}_{probe_type}.err

set -euo pipefail
cd {REPO_DIR}

uv run python train.py \\
    --config '{config_path}' \\
    --mode train \\
    --layer {layer} \\
    --probe-type '{probe_type}' \\
    --log-dir '{log_dir}'
"""
    fd, path = tempfile.mkstemp(
        suffix=".sh", prefix=f"train_l{layer:02d}_{probe_type}_", dir=sld
    )
    with os.fdopen(fd, "w") as f:
        f.write(script)
    return Path(path)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


async def _sbatch_wait(script_path: Path, label: str) -> int:
    """Submit an sbatch script with --wait and return exit code."""
    proc = await asyncio.create_subprocess_exec(
        "sbatch", "--parsable", "--wait", str(script_path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    job_id_line = await proc.stdout.readline()
    job_id = job_id_line.decode().strip()
    print(f"[slurm] Submitted {label} (job {job_id})")
    ret = await proc.wait()
    status = "Done" if ret == 0 else f"FAILED (exit {ret})"
    print(f"[slurm] {status}: {label} (job {job_id})")
    return ret


async def _train_worker(
    slot: int,
    queue: asyncio.Queue,
    config: dict,
    config_path: Path,
    log_dir: Path,
) -> None:
    """Pull (layer, probe_type) jobs from shared queue and submit via sbatch."""
    while True:
        try:
            layer, probe_type = queue.get_nowait()
        except asyncio.QueueEmpty:
            return

        # Skip if already completed.
        result_path = results_dir(log_dir) / f"layer_{layer:02d}" / probe_type / "metrics.json"
        if result_path.exists():
            print(f"[slot {slot}] Skipping layer={layer} probe={probe_type} (already done)")
            queue.task_done()
            continue

        script = _write_train_script(layer, probe_type, config, config_path, log_dir)
        ret = await _sbatch_wait(script, f"layer={layer} probe={probe_type}")
        queue.task_done()

        if ret != 0:
            print(f"[slot {slot}] Train job FAILED: layer={layer} probe={probe_type}")


async def run_orchestrator(config: dict, config_path: Path, log_dir: Path) -> None:
    """Orchestrate the full pipeline: extract → train all → analyze."""
    slurm = config.get("slurm", {})
    max_concurrent = int(slurm.get("max_concurrent", 8))

    # Copy config for reproducibility.
    shutil.copy2(config_path, log_dir / "config.yaml")

    layers = resolve_layers(config)
    probe_types = config["probes"]["types"]

    print(f"[orchestrator] log_dir:     {log_dir}")
    print(f"[orchestrator] model:       {config['model']['id']}")
    print(f"[orchestrator] layers:      {len(layers)} ({layers[0]}..{layers[-1]})")
    print(f"[orchestrator] probe types: {probe_types}")
    print(f"[orchestrator] max concurrent train jobs: {max_concurrent}")
    print()

    # Phase 1: extraction
    acts_dir = activations_dir(config, log_dir)
    meta_path = acts_dir / "metadata.json"
    if meta_path.exists():
        print("[orchestrator] Activations already extracted, skipping extract job.")
    else:
        extract_script = _write_extract_script(config, config_path, log_dir)
        ret = await _sbatch_wait(extract_script, "extract")
        if ret != 0:
            print("[orchestrator] Extraction FAILED. Aborting.")
            sys.exit(1)

    # Phase 2: train all (layer, probe_type) combinations
    queue: asyncio.Queue = asyncio.Queue()
    for layer in layers:
        for probe_type in probe_types:
            queue.put_nowait((layer, probe_type))

    workers = [
        asyncio.create_task(
            _train_worker(i, queue, config, config_path, log_dir)
        )
        for i in range(max_concurrent)
    ]
    await asyncio.gather(*workers)

    # Phase 3: analyze
    analyze(config, log_dir)


# ---------------------------------------------------------------------------
# Local (no-SLURM) runner
# ---------------------------------------------------------------------------


def run_local(config: dict, log_dir: Path) -> None:
    """Run extract + all train jobs sequentially in-process (no SLURM)."""
    print("[local] Running extraction …")
    run_extract(config, log_dir)

    layers = resolve_layers(config)
    probe_types = config["probes"]["types"]
    total = len(layers) * len(probe_types)
    print(f"[local] Training {total} probe(s) ({len(layers)} layers × {len(probe_types)} types) …")

    for layer in layers:
        for probe_type in probe_types:
            run_train(config, log_dir, layer, probe_type)

    analyze(config, log_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Probe training pipeline")
    p.add_argument("--config", required=True, help="Path to YAML config file")
    p.add_argument(
        "--mode",
        choices=["extract", "train"],
        default=None,
        help="Worker mode. Omit to run as orchestrator.",
    )
    p.add_argument("--layer", type=int, default=None, help="Layer index (train mode)")
    p.add_argument("--probe-type", type=str, default=None, help="Probe type (train mode)")
    p.add_argument("--log-dir", type=str, default=None, help="Output directory")
    p.add_argument(
        "--no-slurm",
        action="store_true",
        help="Run everything locally without SLURM (orchestrator mode only)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)

    if args.log_dir:
        log_dir = Path(args.log_dir).resolve()
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        short = config["model"]["short_name"]
        log_dir = (REPO_DIR / "logs" / f"{ts}_{short}").resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "extract":
        run_extract(config, log_dir)

    elif args.mode == "train":
        if args.layer is None or args.probe_type is None:
            print("ERROR: --mode train requires --layer and --probe-type")
            sys.exit(1)
        run_train(config, log_dir, args.layer, args.probe_type)

    else:
        # Orchestrator
        if args.no_slurm:
            run_local(config, log_dir)
        else:
            asyncio.run(run_orchestrator(config, config_path, log_dir))


if __name__ == "__main__":
    main()
