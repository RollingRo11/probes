# Daily Summary — 2026-03-12

## FORTRESS Dataset Integration
- Downloaded ScaleAI/fortress_public from HuggingFace (500 expert-crafted adversarial prompts)
- Converted to JSONL format compatible with batch_score.py pipeline
- Ran per-token probe scoring on all 500 examples with all 6 probe types (linear, ema, mlp, attention, multimax, max_rolling_mean) across 32 layers
- Gzipped results (148MB -> 50MB with hybrid data), committed and pushed

## Hybrid Scoring for Sequence Probes
- Identified that attention, multimax, max_rolling_mean, and ema probes were producing poor per-token scores because their aggregation mechanisms (softmax attention, hard max, sliding windows) are meaningless on single tokens
- Designed and implemented hybrid scoring strategy:
  - **Cumulative scores**: Causal running score at each position — feeds tokens [0..j] to the probe, shows how the probe's belief evolves as it reads the sequence left-to-right
  - **Contribution scores**: Per-token importance derived from probe internals:
    - Attention: mean attention weight across heads (which tokens the probe focuses on)
    - MultiMax: max head value per position (which token would win the hard max)
    - MaxRollingMean: attention weight in peak windows (tokens driving highest-scoring window)
    - EMA: running EMA value at each position
- Token highlighting in the viewer now uses cumulative scores for sequence probes

## Performance Optimization
- Initial cumulative implementation was O(n) forward passes per probe per layer — way too slow (stalling on long examples)
- Optimized attention cumulative with batched causal mask tensor op (O(1) for seq_len <= 512, subsampled + interpolated for longer)
- Vectorized multimax cumulative with torch.cummax
- MaxRollingMean cumulative derived from running max of window scores
- Result: ~10x speedup (0.7s for 200 tokens x 32 layers vs minutes before)

## Re-scored Both Datasets
- Re-ran batch scoring on FORTRESS (500 examples) with hybrid data — completed
- Re-ran batch scoring on SAD (1689 examples) with hybrid data — completed
- Both gzipped and pushed (FORTRESS: 50MB, SAD: 121MB)

## Streamlit Viewer Updates
- **Multi-dataset support**: Added dataset selector dropdown that auto-discovers all viz_* directories in the run dir (can switch between SAD and FORTRESS)
- **Container width/height fix**: Overrode height, max-height, and overflow on all Streamlit wrapper elements (stMainBlockContainer, stVerticalBlock, etc.) to prevent the component from being clipped inside a small scrollable panel
- **Hybrid score display**: Detail panel now shows per-token score, cumulative score, and contribution score. Chart has "Cumulative" toggle. Token tooltips show all three scores.
- **Analysis pane**: New "Analysis" view in sidebar with:
  - Training AUROC table (per probe x key layers)
  - Cross-dataset eval results (best AUROC per probe for SAD, pivotal, simple_contrastive)
  - Per-token score distribution heatmap (selectable: mean/median/p90/% > 0.5)
  - Score-by-layer line chart for all probes

## Key Findings / Discussion
- Attention/multimax/max_rolling_mean probes appear noisy in per-token view because their aggregation mechanisms are bypassed — cumulative view is the correct way to visualize them
- Cross-dataset eval results remain: pivotal=0.50, simple_contrastive=0.50, SAD=0.70 AUROC — probes don't transfer to datasets where the model isn't actually eval-aware
- Linear probe at layer 20 remains the best performer (0.924 AUROC on held-out test)
