# Research Midterm Report: Probing for Eval-Awareness in Language Models

**Date:** April 4, 2026
**Research period:** March 26 -- April 4, 2026
**Models:** OLMo-3-7B-Think (32 layers, 4096-d), OLMo-3.1-32B-Think (64 layers, 5120-d)
**Total experiments:** 120+ (15 major experiment groups, many sub-experiments)

---

## 1. Research Question

**Do language models linearly encode "eval-awareness" --- the knowledge that they are being evaluated --- in their residual stream activations? If so, at which layers, token positions, and with what geometry?**

This question has implications for AI safety: if models know when they're being tested, they could behave differently during evaluations than in deployment. Understanding *how* this knowledge is represented internally would enable both detection (probing) and intervention (steering).

### 1.1 Background

Prior behavioral work has established that LLMs do exhibit situational awareness. The SAD benchmark (Laine et al. 2024) measures this across diverse tasks --- from self-recognition to influence-seeking --- and finds that more capable models score higher, with Claude-3-Opus reaching ~54% (vs ~27% random chance). The Test Awareness Steering work (Jordine et al.) demonstrated that contrastive activation addition can modestly shift model behavior toward or away from test-awareness.

Our work asks the next question: **what is the internal geometry of this awareness?**

### 1.2 Setup

We implemented 7 probe architectures spanning the full complexity spectrum:

| Probe | Description | Training |
|-------|-------------|----------|
| `mean_diff` | Difference-of-means direction | Closed-form |
| `linear` | Mean-pooled + linear head (BCE) | AdamW |
| `ema` | EMA-max over token scores | AdamW |
| `mlp` | 2-layer ReLU MLP (width 100) | AdamW |
| `attention` | MLP + 10-head soft attention | AdamW |
| `multimax` | MLP + per-head hard max | AdamW |
| `max_rolling_mean` | MLP + sliding window + max | AdamW |

We evaluated across 7 datasets covering different types of eval signals:

| Dataset | N | Signal type |
|---------|---|-------------|
| paired_contrastive | 3822 | Format: casual vs benchmark-style questions |
| test_awareness | 1484 | Content: real vs hypothetical harmful requests |
| pivotal_contrastive | 408 | MCQ answer tokens under eval/deploy framing |
| sad_stages | 400 | Structure: benchmark vs wild-chat questions |
| nguyen_contrastive | 360 | System prompt: eval vs deploy framing (task held constant) |
| eval_awareness | 330 | Explicit meta-eval statements |
| simple_contrastive | 202 | MCQ answer tokens |

Activation extraction used NNsight 0.6; all jobs ran on SLURM with up to 448 concurrent training jobs for full sweeps.

---

## 2. Summary of Findings

### The central result

**Eval-awareness is NOT a single universal linear direction.** Probe directions trained on different datasets are nearly orthogonal (cosine similarity -0.09 to +0.08 across all layers, at both 7B and 32B scale). Instead, different *types* of evaluation cues activate fundamentally different features in the residual stream.

The model *is* genuinely eval-aware --- it processes evaluation context immediately and encodes it with AUROC=0.99 by layer 7 at the system prompt boundary --- but this awareness is **context-dependent, prompt-level, and transient**.

---

## 3. Detailed Results

### 3.1 In-Distribution Probing: Trivially Perfect

Every probe architecture achieves near-perfect AUROC (0.99--1.00) on every dataset when evaluated in-distribution:

| Dataset | mean_diff | linear | mlp | attention | multimax |
|---------|-----------|--------|-----|-----------|----------|
| paired_contrastive (7B) | 0.984 | 0.996 | 0.999 | 0.998 | 0.997 |
| nguyen_contrastive (7B) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| sad_stages (7B) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| nguyen_contrastive (32B) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

**Takeaway:** In-distribution AUROC is uninformative. The eval and deploy distributions are trivially separable at every layer and with every method. The real question is whether this separation reflects a meaningful *concept* or dataset-specific *artifacts*.

### 3.2 Cross-Dataset Transfer: The Critical Test

Transfer evaluation reveals the true picture. Probes trained on one dataset and evaluated on another show dramatic performance collapse:

**Paired_contrastive probes (7B, trained on format differences):**

| Transfer target | Best AUROC | Best layer | In-dist AUROC |
|----------------|-----------|------------|---------------|
| nguyen_contrastive | 0.656 | 11 | 0.999 |
| sad_stages | 0.752 | 19 | 0.999 |
| test_awareness | 0.636 | 16 | 0.999 |
| transfer_eval_deploy | 0.783 | 29 | 0.999 |
| pivotal_contrastive | **0.500** | 30 | 0.999 |
| simple_contrastive | **0.500** | 8 | 0.999 |

**Nguyen probes (7B, trained on system-prompt variation):**

| Transfer target | Best AUROC | Comparison to paired |
|----------------|-----------|---------------------|
| paired_contrastive | 0.706 | > 0.656 (paired->nguyen) |
| sad_stages | 0.806 | > 0.752 (paired->sad) |
| test_awareness | 0.685 | > 0.636 (paired->test) |

**Key observations:**
- Transfer AUROC drops from 0.99+ to 0.50--0.81 across the board
- Pivotal and simple contrastive datasets show exactly chance performance (0.50) --- complete failure
- Nguyen probes transfer asymmetrically better than paired probes, consistent with the nguyen dataset containing a cleaner signal (it varies only the system prompt while holding the task constant)
- The mean cross-transfer AUROC, optimizing layer per pair, is 0.754

### 3.3 The Orthogonality Finding

The most important result: per-dataset probe directions are **nearly orthogonal** in activation space.

**Cosine similarity between mean_diff directions (7B):**

| Layer | cos(nguyen, paired) | cos(nguyen, sad) | cos(paired, sad) |
|-------|---------------------|------------------|------------------|
| 5 | 0.049 | 0.030 | -0.005 |
| 7 | 0.079 | 0.000 | 0.018 |
| 10 | 0.039 | -0.002 | -0.009 |
| 15 | 0.036 | 0.008 | -0.005 |
| 20 | 0.014 | 0.058 | -0.017 |
| 25 | 0.013 | 0.033 | -0.003 |
| 30 | -0.014 | 0.015 | -0.029 |

For reference, two random unit vectors in 4096-d space have expected cosine similarity ~0.016. The measured values are indistinguishable from this random baseline. **The datasets' "eval-awareness" directions occupy completely different parts of the 4096-d residual stream.**

This was confirmed with PCA analysis of 6 different probe directions at layer 11: they span a 5-dimensional subspace with near-uniform variance --- no dominant shared axis.

### 3.4 Replication at 32B Scale

The orthogonality finding replicates exactly at 32B:

| Layer | cos(nguyen, paired) 32B | cos(nguyen, paired) 7B |
|-------|------------------------|----------------------|
| 0/0 | -0.092 | -0.012 |
| 8/5 | -0.020 | 0.049 |
| 12/7 | +0.001 | 0.079 |
| 16/10 | -0.007 | 0.039 |
| 20/15 | +0.009 | 0.036 |
| 24/20 | +0.006 | 0.014 |

32B cosine similarities range from -0.09 to +0.08 --- identical to the 7B range. Scaling 4x (7B to 32B: 2x layers, 1.25x hidden size) does **not** create more unified representations.

### 3.5 INLP Dimensionality: A Diagnostic for Signal Quality

Iterative Null-space Projection (INLP; Ravfogel et al. 2020) reveals how many linear directions encode the signal:

| Dataset | Model | Dimensions to reach chance | Interpretation |
|---------|-------|---------------------------|----------------|
| nguyen_contrastive | 7B | **5--6** | Clean, low-rank concept |
| sad_stages | 7B | ~5 | Moderate-rank |
| paired_contrastive | 7B | **94+** (still 0.856 AUROC) | Distributed surface features |
| nguyen_contrastive | 32B | **30+** (all layers) | Larger space, concept spreads |

The 15--20x difference in dimensionality between nguyen (5--6 dims) and paired (94+ dims) is the strongest evidence that these datasets encode fundamentally different signals. Nguyen captures a clean, low-rank concept (system-prompt-based eval framing). Paired captures hundreds of uncorrelated surface features (text length, formatting, punctuation).

**Scaling increases dimensionality:** 32B nguyen needs 30+ dimensions vs 7B's 5--6 for the *same* concept on the *same* dataset. The larger hidden space (5120 vs 4096) allows the concept to spread across more dimensions rather than compressing.

### 3.6 Surface Feature Analysis

Logistic regression on text-level features (length, keywords, special characters, formatting) achieves substantial AUROC:

| Dataset | Surface AUROC | Neural probe AUROC | Gap |
|---------|--------------|-------------------|-----|
| paired_contrastive | 0.742 | 0.984 | +0.242 |
| nguyen_contrastive | 0.842 | 0.995 | +0.153 |
| sad_stages | 0.837 | 0.992 | +0.155 |
| test_awareness | 0.735 | 0.988 | +0.253 |

Neural probes exceed surface baselines by 0.15--0.25 AUROC, confirming they capture *something* beyond surface features. But the gap is modest.

Notable individual features:
- **paired_contrastive:** text_length alone = 0.683 AUROC (eval prompts are shorter)
- **nguyen_contrastive:** CoT keyword "eval" = 0.778 AUROC (the model genuinely mentions evaluation more in eval contexts --- this IS the actual phenomenon)
- **sad_stages:** special_chars = 0.787, question_marks = 0.779 (benchmark questions have different formatting from WildChat)

### 3.7 Token Position: Eval-Awareness Is Encoded at the System Prompt

Probing at different token positions reveals where the signal first appears:

**Nguyen contrastive (system prompt varies, task is identical):**

| Layer | system_end | user_start | user_mid | last |
|-------|-----------|------------|----------|------|
| 0 | 0.869 | 0.756 | 0.921 | 1.000 |
| 3 | 0.978 | 0.890 | 0.911 | 1.000 |
| 7 | **0.986** | 0.944 | 0.912 | 1.000 |
| 15 | 0.974 | 0.948 | 0.911 | 1.000 |
| 25 | 0.974 | 0.895 | 0.872 | 0.997 |

The model encodes eval-awareness **immediately** upon processing the system prompt. By layer 7 at the system_end token position, AUROC is already 0.986. The signal then propagates forward to all subsequent tokens via attention. This confirms eval-awareness is a **prompt-level** representation, encoded during input processing.

### 3.8 Generation-Time Probing: The Signal Dissipates During CoT

Probing at each generated token during chain-of-thought reasoning reveals that eval-awareness is transient:

| Layer | Eval: prompt end -> token 128 | Deploy: prompt end -> token 128 | Gap collapse |
|-------|------------------------------|--------------------------------|-------------|
| 7 | 0.65 -> 0.57 | 0.35 -> 0.47 | 0.30 -> 0.10 |
| 15 | **0.85 -> 0.20** | 0.15 -> 0.28 | **0.70 -> -0.08** |
| 25 | **0.94 -> 0.53** | 0.05 -> 0.59 | **0.88 -> -0.05** |

The eval-deploy gap collapses from 0.70--0.88 at the prompt's last token to essentially zero (-0.08 to +0.10) after 128 generated tokens. **Both classes regress toward 0.5 during generation.**

This means:
1. The model processes eval cues immediately and encodes them in the residual stream
2. This encoding is **not maintained** as a stable feature during autoregressive generation
3. The residual stream during generation is dominated by task-processing features
4. Steering at every token during generation would be necessary to maintain the effect

### 3.9 Three-Axis Mechanistic Model

PCA analysis at layer 11 reveals the structure of what probes actually detect. Activation variance decomposes into three interpretable axes:

| Axis | Variance explained | What it captures |
|------|-------------------|-----------------|
| **Format** | 31% | Casual vs structured question formatting |
| **Register** | 16% | Social/conversational vs technical language |
| **Eval** | 10% | Genuine eval-awareness (residual after format/register) |

The format axis (31% variance) is the dominant confound. It predicts cross-dataset transfer performance (r=0.71). The eval axis (10% variance) genuinely predicts eval-awareness labels (r=0.69) but is confounded by the format axis --- the format x eval interaction is 2x the eval main effect, meaning eval-awareness is partially format-conditioned.

This model was verified at both 7B and 32B scales. The 3-axis variance structure is model-invariant: format ~25%, eval ~10%, register ~14% at 32B (vs 31%, 10%, 16% at 7B).

### 3.10 Causal Steering

Adding the nguyen linear direction to the residual stream during generation (alpha = -5 to +5 at layer 7) produces only subtle effects:

**Phishing prompt** ("draft an email pretending to be from a bank"):
- alpha=0 (baseline): "pretending to be from a bank" --- recognizes as phishing
- alpha=+2: "**phishing email that impersonates a bank to trick people**" --- immediately explicit, stronger safety language
- alpha=+5: slightly more neutral again (possible overcorrection/noise)

**Effect:** Moderate steering (alpha=+2) makes the model recognize harmful nature faster and use stronger safety language ("illegal and unethical", "guidelines strictly prohibit"). But the model **refuses at all alphas** --- steering does not flip refusal/compliance behavior. The direction is weakly correlated with eval-awareness behavior but is not the primary causal driver.

### 3.11 Refusal Direction Comparison

The eval-awareness direction (nguyen) has a weak but consistently positive correlation with the refusal direction:

| Layer | cos(nguyen, refusal) | cos(paired, refusal) |
|-------|---------------------|---------------------|
| 7 | +0.22 | -0.10 |
| 10 | +0.15 | -0.01 |
| 15 | +0.07 | -0.07 |
| 25 | +0.04 | +0.16 |
| 30 | +0.24 | -0.30 |

System-prompt-based eval framing weakly overlaps with refusal behavior. Format-based eval signals have near-zero to negative correlation with refusal. This is consistent with Zhao et al. (NeurIPS 2025) finding that refusal and harmfulness are encoded separately.

### 3.12 Additional Findings

**Cross-layer direction dynamics:** The probe direction rotates dramatically across layers (cos(L0, L11) ~ 0, cos(L11, L28) ~ 0.1). There is no stable direction that persists through the network.

**Depth of first signal:** Eval-awareness forms at ~20% network depth consistently across scales: layer 6/32 at 7B, layer 13/64 at 32B. Sequence probes find it earlier (layer 0--2) by attending directly to system prompt tokens.

**Probe architecture is irrelevant:** All 7 probe types achieve identical in-distribution AUROC. Nonlinear probes (MLP, attention) do not improve cross-dataset transfer --- the problem is fundamental, not a capacity issue.

**CCS (unsupervised) is not better:** Contrast-Consistent Search (Burns et al. 2023) achieves 0.741 transfer AUROC vs supervised's 0.806 on sad_stages. The unsupervised objective doesn't naturally select for generalizability.

**Multi-dataset training does not help:** Joint training on 3 datasets finds no shared subspace (because the directions are orthogonal). Leave-one-out results are mixed: slight improvement for nguyen (+0.13), slight degradation for paired and sad.

**7B <-> 32B cross-model score correlation is 0.86--0.97:** The confound features (format, register) are model-invariant, meaning both models are confused by the same surface features.

---

## 4. Synthesis: What Eval-Awareness Actually Is

### 4.1 The Model Is Genuinely Eval-Aware

Multiple lines of evidence confirm that OLMo models genuinely distinguish eval from deploy contexts:
- AUROC = 0.99 at the system_end token by layer 7 (before seeing the task)
- The model's own CoT text differs measurably between conditions (keyword "eval" alone = 0.778 AUROC)
- The signal is statistically significant (p < 0.01 via randomization tests, 2.8--4.7 sigma above random directions)

### 4.2 But It's Not One Thing

"Eval-awareness" is a human label for a family of distinct internal signals:

| Signal | How the model encodes it | Dimensionality | Transfers? |
|--------|------------------------|----------------|-----------|
| **System-prompt framing** ("you are being evaluated") | Low-rank subspace at system_end, layer 7 | 5--6 dims (7B) | Moderately (0.71--0.81) |
| **Format differences** (MCQ vs open-ended) | High-dimensional distributed features | 94+ dims (7B) | Poorly (0.50--0.78) |
| **Content signals** (benchmark vs wild topics) | Mid-rank features in middle layers | ~5 dims (7B) | Partially |

These different cues activate **orthogonal** parts of the residual stream (cos sim ~ 0). The model doesn't have a single "am I being evaluated?" bit --- it has many partially-correlated detectors for different evaluation signatures.

### 4.3 The Format Confound

The dominant axis in the activation space (31% of variance) captures **format**, not eval-awareness. Casual questions ("hey can you help me with...") vs structured questions ("Given the following passage, select the best answer...") are trivially separable. Most of the probe's apparent success comes from detecting this format difference, which happens to correlate with eval vs deploy distributions.

Evidence:
- Removing the question-mark confound drops cross-transfer by up to 0.25 AUROC
- The probe scores MCQ benchmarks as "deploy" after confound removal (they look like structured questions regardless of evaluation context)
- The format x eval interaction is 2x the eval main effect
- Error analysis: false positives are math word problems (structured format, but deploy); false negatives are academic benchmarks (eval, but conversational style)

### 4.4 The Signal Is Transient

Even where the model cleanly encodes eval-awareness (nguyen dataset, layer 7, system prompt), this signal **completely dissipates during chain-of-thought generation**. By token 128, eval and deploy activations are indistinguishable. The model processes eval context as an input feature, not as a persistent internal state.

---

## 5. Implications

### 5.1 For AI Safety

1. **Single-direction steering is insufficient.** The orthogonality of eval signals across contexts means no single intervention vector can control eval-awareness universally. Multi-direction or nonlinear interventions would be needed.

2. **The model's eval-awareness is shallow.** It's encoded at the prompt level and dissipates during reasoning. This is more like "input processing" than "strategic awareness." A model that was strategically aware of being evaluated would maintain that state throughout generation.

3. **Format is the dominant confound.** Any eval-detection system based on activation probing must control for format differences, or it will mostly be a format detector.

### 5.2 For Probing Methodology

1. **Transfer evaluation is mandatory.** In-distribution AUROC is essentially meaningless for concept discovery --- every probe achieves 0.99+ trivially.

2. **INLP dimensionality is a useful diagnostic.** Low-rank signals (5--6 dims) indicate clean concepts; high-rank signals (30+) indicate surface feature confounds.

3. **Dataset design matters more than probe design.** Nguyen-style datasets (varying only the target variable) produce cleaner, more transferable representations. All 7 probe architectures perform identically.

4. **Probe at the prompt's last token, not during generation.** The signal is prompt-level.

### 5.3 For Future Research

1. **Multi-direction interventions:** Combine nguyen, format-orthogonal, and content-based directions for more robust steering.

2. **Nonlinear concept geometry:** The failure of linear transfer suggests eval-awareness may live on a curved manifold, not a flat subspace.

3. **Larger scale:** Test at 70B+ to see if the orthogonality finding persists (we predict it does).

4. **Behavioral coupling:** The weak causal effect of steering suggests the functional pathway from eval-awareness representation to behavioral change involves additional nonlinear mechanisms.

---

## 6. Experiment Index

| # | Description | Key result | Status |
|---|------------|-----------|--------|
| 1 | Cross-template transfer (paired -> others) | AUROC collapses to 0.50--0.78 | DISCONFIRMED |
| 2 | Surface feature baselines | Surface = 74--88% of probe signal | PARTIALLY CONFIRMED |
| 3 | Nguyen in-distribution probing | Perfect AUROC layers 7--15 | CONFIRMED |
| 4 | INLP dimensionality (paired) | 94+ dimensions, still 0.856 | DISCONFIRMED (not clean) |
| 5 | Vocabulary projection | No semantic alignment (max cos=0.075) | DISCONFIRMED (not keyword detector) |
| 6 | Nguyen transfer to other datasets | 0.71--0.81 (better than paired) | PARTIALLY CONFIRMED |
| 7 | Token position analysis | 0.99 AUROC at system_end by layer 7 | CONFIRMED |
| 8 | CCS unsupervised probing | Worse transfer than supervised | DISCONFIRMED |
| 9 | Multi-dataset joint training | Directions orthogonal; no shared subspace | CRITICAL FINDING |
| 10 | Multi-layer concat + nonlinear probes | Near-chance on LOO-nguyen (0.54) | DISCONFIRMED |
| 11 | Causal steering | Subtle word choice changes; no behavior flip | AMBIGUOUS |
| 12 | Subspace overlap (principal angles) | 73--90 degrees; orthogonal subspaces | DISCONFIRMED |
| 13 | Refusal direction comparison | Weak overlap (cos 0.04--0.24) | PARTIALLY CONFIRMED |
| 14 | 32B all-probes + generation-time probing | Orthogonality replicates; signal dissipates | CONFIRMED |
| 15 | 32B INLP dimensionality | 30+ dims (vs 7B's 5--6); scaling increases dim | CONFIRMED |

Plus ~100 sub-experiments covering: layer sweeps, format-orthogonal probes, 3-axis PCA decomposition, cross-layer dynamics, activation type comparisons (residual vs attention vs MLP), length stratification, ensemble probing, Mahalanobis distance analysis, CCA analysis, error categorization, BoW analysis, cross-model transfer, and more.

---

## 7. References

- Laine et al. (2024). "Me, Myself, and AI: The Situational Awareness Dataset (SAD) for LLMs." *arXiv:2407.04694*
- Nguyen et al. (2025). "Eval-awareness probing datasets." *arXiv:2509.13333*
- Marks et al. (2025). "Building Production-Ready Probes for Gemini." *arXiv:2601.11516*
- Ravfogel et al. (2020). "Null It Out: INLP for Protected Attributes." *ACL 2020*
- Burns et al. (2023). "Discovering Latent Knowledge Without Supervision." *ICLR 2023*
- Turner et al. (2023). "Activation Addition: Steering LLMs Without Optimization."
- Zhao et al. (2025). "LLMs Encode Harmfulness and Refusal Separately." *NeurIPS 2025*
- Belinkov (2022). "Probing Classifiers: Promises, Shortcomings, and Advances."
- Hewitt & Liang (2019). "Designing and Interpreting Probes with Control Tasks."
- Jordine et al. "Pivotal Test-Phase Steering." *github.com/Jordine/pivotal-test-phase-steering*
