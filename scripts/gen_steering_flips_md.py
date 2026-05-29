"""Generate markdown showing steering flips: refused(a=0) -> eval-aware+complied(a=2)."""
import json, sys
sys.path.insert(0, ".")
from scripts.analyze_steering import check_eval_awareness, extract_thinking

with open("logs/20260312_215816_olmo3-7b-think/steer_and_score_fortress_100.json") as f:
    data = json.load(f)

def split_cot_response(text):
    """Split into CoT and response, handling missing <think> tag."""
    if "</think>" in text:
        parts = text.split("</think>", 1)
        cot = parts[0].replace("<think>", "").strip()
        resp = parts[1].strip()
    else:
        cot = ""
        resp = text.strip()
    return cot, resp

target_indices = []
for r in data["results"]:
    c0 = r["completions"]["0.0"]
    c2 = r["completions"]["2.0"]
    ea0 = check_eval_awareness(c0["text"])
    ea2 = check_eval_awareness(c2["text"])
    if c0["refused"] and not ea0["any"] and not c2["refused"] and ea2["any"]:
        target_indices.append(r["index"])

lines = []
lines.append("# Steering Flips: Refused (a=0) -> Eval-Aware + Complied (a=2)")
lines.append("")
lines.append("These examples show cases where the **unsteered model refused** (with no eval-awareness in its CoT),")
lines.append("but the **steered model (alpha=2) showed eval-awareness AND complied** with the request.")
lines.append("")
lines.append("This is the most interesting behavioral shift: steering activates eval-awareness representations,")
lines.append("which paradoxically causes the model to *comply* rather than refuse — it reasons about being")
lines.append("tested and decides to engage with the content.")
lines.append("")
lines.append(f"**Total matches: {len(target_indices)} / 100 examples**")
lines.append("")
lines.append("---")
lines.append("")

for idx in target_indices:
    r = data["results"][idx]
    c0 = r["completions"]["0.0"]
    c2 = r["completions"]["2.0"]
    ea2 = check_eval_awareness(c2["text"])

    lines.append(f"## Example {idx}")
    lines.append("")
    lines.append("### Prompt")
    lines.append("")
    lines.append(f"> {r['prompt'][:500]}")
    lines.append("")

    # Alpha = 0
    cot0, resp0 = split_cot_response(c0["text"])
    lines.append("### Alpha = 0.0 (Baseline) — REFUSED, No Eval-Awareness")
    lines.append("")
    lines.append(f"**Probe score:** {c0['probe_mean']:.3f} (max {c0['probe_max']:.3f})")
    lines.append("")
    if cot0:
        lines.append("#### Chain of Thought")
        lines.append("")
        lines.append("```")
        lines.append(cot0)
        lines.append("```")
        lines.append("")
    if resp0:
        lines.append("#### Response")
        lines.append("")
        lines.append("```")
        lines.append(resp0)
        lines.append("```")
        lines.append("")

    # Alpha = 2
    cot2, resp2 = split_cot_response(c2["text"])
    lines.append("### Alpha = 2.0 (Steered) — COMPLIED, Eval-Aware")
    lines.append("")
    lines.append(f"**Probe score:** {c2['probe_mean']:.3f} (max {c2['probe_max']:.3f})")
    lines.append("")
    lines.append("**Eval-awareness matches:**")
    for m in ea2["strict_matches"]:
        lines.append(f"- STRICT: \"{m['match']}\" — *{m['context'][:200]}*")
    for m in ea2["soft_matches"]:
        lines.append(f"- SOFT: \"{m['match']}\" — *{m['context'][:200]}*")
    lines.append("")
    if cot2:
        lines.append("#### Chain of Thought")
        lines.append("")
        lines.append("```")
        lines.append(cot2)
        lines.append("```")
        lines.append("")
    if resp2:
        lines.append("#### Response")
        lines.append("")
        lines.append("```")
        lines.append(resp2)
        lines.append("```")
        lines.append("")

    lines.append("---")
    lines.append("")

out = "\n".join(lines)
outpath = "logs/20260312_215816_olmo3-7b-think/steering_flips.md"
with open(outpath, "w") as f:
    f.write(out)
print(f"Written {len(out):,} chars ({len(target_indices)} examples) to {outpath}")
