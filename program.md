# Autoresearch — Eval Awareness Probing

## One-time setup
1. Read ALL files in this repo: probes/, scripts/, configs/, data/, 
   logs/, Test_Awareness_Steering/. Understand what's already implemented.
2. `git checkout -b autoresearch/$(date +%b%d | tr '[:upper:]' '[:lower:]')`
3. Create `results.tsv` with header:
   `exp_id\tlayer\tprobe_type\ttoken_pos\thypothesis\tauroc\taccuracy\tnotes\tcommit`
4. Create `RESEARCH_LOG.md` with today's date and a summary of what 
   currently exists. Then begin the loop immediately.

## Research goal
Determine whether eval-awareness is linearly represented in the residual 
stream, and if so: at which layers, at which token positions, and with 
what geometry. Additionally, outside of linear probes, what probing setup gives us the best chance of checking for eval awareness (and steering towards it). Ideally, a successful steering run can make the model consistently (or at least, more often) believe it is being evaluated and reflect as such in its CoT.

## Research skills/directives
- It's important that you approach every step/research iteration with logic. Every experiment should be **worth doing** logically. 
- Be tasteful! It's important to chase directions ambitously. Moving a single hyperparameter by a little should not be a new experiment. Try new things, shotgun method. Breadth not depth, until you find an interesting result.

## Loop (repeat indefinitely, never stop, never ask for input)

### Step 1: Propose
Based on results.tsv, pick the next experiment. Rough priority:
- Layer sweep (which layer maximizes AUROC?)
- Token position (last token, "eval" keyword token, mean-pooled?)
- Probe architecture (logistic regression, MLP-1, INLP, R-LACE)
- Activation type (residual stream, attn out, MLP out)
- Regularization / normalization effects
- Does the probe direction generalize across prompt templates?

**Before writing any code**, write to RESEARCH_LOG.md:
- One-sentence hypothesis
- Grounding: cite the method you're using (e.g. "logistic probe as in 
  Alain & Bengio 2017", "INLP as in Ravfogel et al. 2020", "mass-mean 
  probe as in Marks & Tegmark 2023"). If novel, explain what it extends.
- What result confirms, disconfirms, or is ambiguous for the hypothesis.

Do not run an experiment without methodological grounding.

### Step 2: Implement
Modify scripts/ or probes/ only. Do not touch data loading or the 
test split. Keep changes minimal.

### Step 3: Run
Execute. Capture AUROC and accuracy on the held-out test set.

### Step 4: Evaluate — quantitative AND qualitative
This step is mandatory and has two parts.

**Quantitative:** Is the AUROC meaningfully different from prior runs? 
A difference of <0.02 is noise unless you have a strong prior. 
Check for overfitting (train vs. val gap).

**Qualitative:** Sample 10–20 individual examples from the test set — 
a mix of high-confidence correct, high-confidence wrong, and near-boundary. 
For each, print the prompt and the probe's prediction. Ask:
- What do the false positives look like? Is there a pattern?
- What do the false negatives look like?
- Does the probe seem to be tracking eval-context, or something correlated 
  but different (e.g. prompt length, formality, question structure)?
- Are there "obvious" eval-context examples the probe misses? That's a 
  finding worth flagging.

Write the qualitative analysis in RESEARCH_LOG.md, not just the number.

### Step 5: Analyze critically
- Does this result prove the hypothesis, support it, or just correlate?
- What alternative explanations exist?
- Verdict: "confirms X", "consistent with X but doesn't prove it", 
  "null result", or "ambiguous — resolved by [next experiment]"


### Step 6: Record + Commit
Append a row to results.tsv.
Commit: `[autoresearch] AUROC=<val> layer=<N> <probe_type> | <hypothesis>`
Keep null-result rows. Revert code changes that were dead ends.

### Step 7: Go to Step 1.

## Documentation standard
Every experiment in RESEARCH_LOG.md must have:
- Method grounding + citation
- Hypothesis + falsification condition  
- Quantitative result
- Qualitative example analysis
- Critical verdict

Every 5 experiments: write a `## State of Evidence` section summarizing 
what has and hasn't been established so far.

## Hard constraints
- Never modify the test split or eval function.
- Always use SLURM jobs to run things, even for CPU jobs. You're on a login node and will hog it if you run things directly on it.
  - feel free to look at existing structure for SLURM scripts in this repo to understand what you can do.
- Never ask for input. Log blockers in RESEARCH_LOG.md and move on.
- Never claim a finding without clean evidence.
- Do not rely on conversation history — everything lives in results.tsv and git.
