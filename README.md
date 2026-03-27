# Mo²BERTa — Mixture-of-Recursions on a Modernized BERT (Prototype)

Mo²BERTa adapts the **Mixture-of-Recursions (MoR)** framework (Bae _et al._ 2025) which originally designed for autoregressive decoders, to bidirectional **Masked Language Modeling**. A lightweight sigmoid router dynamically assigns each token a recursion depth (1–4 steps over a shared parameter block), allocating more compute for recursive representation refinement to tokens that need it, and less to those that don't.

The core hypothesis is that "easy" tokens (common stopwords, contextually obvious words) require fewer refinement steps, while "hard" tokens (`[MASK]` positions or morphologically complex subwords) benefit from deeper recursive processing. This allows the model to reallocate its computational budget dynamically.

> **Status:** Research prototype / Proof-of-Concept. Not intended for production use.

## Usage
*   **For model, training, inference code:** please check MoRBERT.ipynb in the repo
*   **For updated version addressing the limitation of for-loop calls and context isolation**: see [Mo²BERTa-v2-proto](https://huggingface.co/gbyuvd/Mo2BERTa-v2-proto)

## Model Details
| Field                 | Value                                                                                              |
| --------------------- | -------------------------------------------------------------------------------------------------- |
| **Architecture**      | Encoder-only Transformer with Mixture-of-Recursions (MoR)                                          |
| **Base adaptation**   | Bae _et al._ (2025), originally autoregressive; adapted here to bidirectional MLM                    |
| **Unique parameters** | ~9.6M (embedding layer excluded from count; MLM head weight-tied to embeddings)                    |
| **Effective depth**   | Variable per token: 1–4 recursions over a shared 2-layer block (~7 flat-layer equivalent in FLOPs) |
| **Training data**     | TinyStories-valid (small subset, PoC scale)                                                        |
| **Tokenizer**         | `bert-base-uncased`                                                                                |
| **Compute cap**       | 600 TFLOPs (interim results @ 100T)                                      |
| **Hardware**          | NVIDIA GeForce 930M (Compute Capability 5.0, consumer laptop GPU)                                  |
| **License**           | MIT                                                                                                |

## Architecture

### Core Design: M-Cycle Middle with Expert-Choice Routing

Mo²BERTa implements the **M-Cycle Middle** parameter-sharing strategy from the MoR paper: one unique first layer, one unique last layer, and a set of shared middle blocks applied recursively `N_recursion` times. For this prototype (`N_layers=4`, `N_recursion=4`), the shared middle consists of 2 blocks applied 4 times, producing an effective unrolled depth of ~7–8 layers while maintaining only ~9.6M unique parameters.

**Expert-choice routing** operates at the boundary between each recursion step. A linear router with sigmoid activation computes a scalar score for every currently-active token. The top-k tokens, selected according to a fixed capacity schedule that decreases from 100% → 75% → 50% → 25% of active tokens across the four recursion steps, continue to the next recursion. Tokens not selected retain their hidden state without further update and are said to have "exited early."

The update rule for continuing tokens follows Eq. 2.1 from the MoR paper:

```
H^(r+1)_t = g_t^r * f(H^(r)_t, Φ') + H^(r)_t   if selected
           = H^(r)_t                               otherwise
```

where `g_t^r` is the continuous sigmoid score and `f(·, Φ')` is the shared recursive block. The continuous gate weight means gradient flows through `g_t * f(H, Φ')` directly such that no Straight-Through Estimator is needed. The discrete top-k selection has no gradient and does not require one.

```
Total params  : 9.69M
Unique params : 9.69M  (weight-tied mlm_head not double-counted)
  Embedding         : 7.85M
  Unique tfm blocks : 0.92M
  Shared tfm blocks : 0.92M  (×4 recursions)
  MLM head          : 0.00M  (weight-tied → 0 unique params; counted in embedding)
  Routers           : 0.8K
```
### Key Differences from the Original MoR Paper

The original MoR paper (Bae _et al._ 2025) targets autoregressive causal language modeling on Llama-style decoders. This adaptation makes the following deliberate changes for the encoder setting:

*   **Bidirectional attention.** The causal mask is removed entirely. All tokens attend to all other currently-active tokens at each recursion depth. No auxiliary router is needed to prevent causality leakage because there is no causality constraint in an encoder.

*   **Post-block routing.** The original paper routes on the hidden state *before* the block is applied (the router predicts whether a token needs more processing). This implementation routes on the hidden state *after* the block, so the gate decision reflects the refined representation. In the encoder setting this is the appropriate framing: the router asks "is this token's context already rich enough to stop?" rather than "does this token need processing?"

*   **MLM objective.** The autoregressive next-token prediction loss is replaced with BERT-style masked language modeling: 15% of tokens masked per batch (80% `[MASK]`, 10% random, 10% unchanged), with cross-entropy loss over masked positions only.

*   **No KV sharing variant.** The paper proposes a second KV strategy ("recursive KV sharing") that reuses KV pairs from recursion 1 for exited tokens at deeper steps. This implementation uses only the primary recursion-wise caching strategy. The MoR paper's own ablations show KV sharing slightly degrades performance under expert-choice routing, so this omission is intentional.

*   **Auxiliary loss coefficient.** The BCE auxiliary loss weight is set to α=0.1 (versus the paper's α=0.01). This was found empirically to produce faster router calibration (bimodal score distributions emerging within the first 200 training steps) on the small-scale TinyStories setting but a more thorough sweep is needed.

### Modern Encoder Enhancements

Both Mo²BERTa and all baselines use the following components, so they cancel as experimental confounds:

*   **Grouped-Query Attention (GQA):** 8 query heads, 4 KV heads (2:1 ratio), reducing KV memory pressure.
*   **Rotary Positional Embeddings (RoPE):** Applied to queries and keys. During sparse gather-scatter attention steps, RoPE frequencies are indexed by **original global token positions** (not re-indexed 0..k positions), preserving positional integrity when tokens are processed out of dense order.
*   **Weight-Tied MLM Head:** The output projection shares weights with the input token embedding layer, keeping the unique parameter count honest across all compared models.

## Baseline Architectures

Two baselines are constructed for isoFLOP comparison, both using identical GQA, RoPE, and weight-tying configurations:

*   **IsoParam Baseline (Vanilla L4).** A flat Transformer with 4 unique layers and the same parameter count as Mo²BERTa (~9.6M unique). Tests whether recursive depth itself provides value beyond simply having the same number of parameters.

*   **IsoDepth Baseline (Vanilla L7).** A flat Transformer with ~7 unique layers, matching Mo²BERTa's effective FLOP budget. The depth is computed from Mo²BERTa's capacity schedule: recursion step j processes a fraction (N_r − j + 1)/N_r of tokens, reducing both attention (quadratic in active sequence length) and MLP (linear in active token count) costs relative to a full dense pass. The L7 baseline represents an equivalent total floating-point operation count for a dense model. Tests whether adaptive depth allocation is more efficient than uniformly applying the same compute budget to all tokens.

## Training

>
> The name Mo²BERTa is a post-train naming, so in this report (esp. in plots) the model is still referred to as MoR-BERT
>

- **Dataset:** `TinyStories-valid` tokenized with `bert-base-uncased`
- **Masking:** Standard BERT MLM, 15% of tokens masked per sequence
- **Optimizer:** AdamW (lr=3e-4, weight_decay=0.01, β=(0.9, 0.95))
- **Batch size:** 8 sequences × 128 tokens
- **LR schedule:** Constant (no warmup or decay), intentional simplification for PoC; see Limitations
- **Gradient clipping:** 1.0 (global norm)
- **Stopping criterion:** Fixed compute budget (TFLOP cap @ 100 & 600T), not steps or epochs
- **Validation trigger:** Every 50 TFLOPs of cumulative compute

## Evaluation Results (600 TFLOP Final)

All three models trained to the same 600T cumulative FLOP budget. Validation on 20 batches of held-out TinyStories-valid. **Best checkpoint metrics** are reported (not final-step), as all models exhibit late-stage oscillation under the constant LR (see the Overfitting note below).

### Best Checkpoint Metrics

| Metric                    |  Mo²BERTa  | IsoParam (L4) | IsoDepth (L7) |
| ------------------------- | :--------: | :-----------: | :-----------: |
| **Best Val Loss**         | **1.8127** |    1.9328     |    1.8310     |
| **Best Val Acc (MLM)**    |   65.79%   |    62.54%     |  **67.19%**   |
| **Best checkpoint @ (T)** |    500T    |     500T      |     600T      |
| **Unique parameters**     |   9.69M    |     9.69M     |    11.07M     |
| **Tokens seen at best**   |   8.35M    |     8.60M     |     9.04M     |

### Training Regime Analysis

The full 600T run reveals a regime-dependent picture rather than a single winner:

![image](https://cdn-uploads.huggingface.co/production/uploads/667da868d653c0b02d6a2399/r_-cLhzR9J6zb0Uj8mmtL.png)

Based on the IsoFLOP triangulation plots, 

| Regime              | Leader                                               | Interpretation (Val Loss & Val Acc)                                                                                                                                                                                                                    |
| :------------------ | :--------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **0–200T (early)**  | **Mo²BERTa**                                         | **Val Loss**: Mo²BERTa starts lower (~3.5 vs ~3.9) and maintains advantage to ~150T.<br>**Val Acc**: Leads consistently. Adaptive routing provides better sample efficiency per FLOP.                                                                  |
| **200–500T (mid)**  | **Contested**<br>(IsoDepth early /<br>Mo²BERTa late) | **Val Loss**: IsoDepth dips lowest around 250T (~2.1), then Mo²BERTa takes lead at 500T (~1.81).<br>**Val Acc**: IsoDepth leads 200–300T; Mo²BERTa leads 350–500T (peaking at 65.79%).<br>*Oscillation dominates; no clear single winner throughout.*  |
| **500–600T (late)** | **IsoDepth**                                         | **Val Loss**: IsoDepth improves (1.89 → 1.83) while Mo²BERTa degrades (1.81 → 1.95).<br>**Val Acc**: IsoDepth continues rising to 67.19%; Mo²BERTa drops to ~63% (overfitting).<br>Larger parameter capacity (L7) prevents saturation on limited data. |
| **Per-parameter**   | **Mo²BERTa > IsoParam**                              | At identical parameter counts (L4), recursion (Mo²BERTa) consistently outperforms flat execution (IsoParam) on both metrics across all regimes.                                                                                                        |

- **Mo²BERTa vs IsoParam** is the cleanest comparison, identical parameter count, identical FLOP budget, same components. Mo²BERTa leads throughout on both loss and accuracy, peaking at 1.8127 / 65.79% vs IsoParam's 1.9328 / 62.54% at 500T. This is the core PoC claim: recursive adaptive depth outperforms flat execution at equal parameter count.

- **Mo²BERTa vs IsoDepth** is more nuanced. IsoDepth has 14% more unique parameters and eventually surpasses Mo²BERTa in accuracy (67.19% vs 65.79%) while seeing fewer tokens (9.04M vs 10.02M). However, Mo²BERTa matches IsoDepth's loss (1.8127 vs 1.8310) with fewer parameters, and leads clearly in the early compute regime. The late IsoDepth advantage is consistent with its larger parameter capacity not yet saturated on the small dataset.

- **The data efficiency panel** (val loss vs tokens seen) shows Mo²BERTa reaches low loss faster per token than both baselines in the 6 & 8M token range, consistent with the router concentrating compute on hard tokens from early in training.

### Note on Late-Stage Oscillation

Mo²BERTa shows more pronounced val loss oscillation after 500T than the flat baselines:

```
MoR  500T: 1.8127 → 550T: 1.8658 → 600T: 1.9508  (degrading)
IsoD 500T: 1.8869 → 550T: 1.8849 → 600T: 1.8310  (still improving)
```

This reflects two compounding factors: (1) the constant LR provides no annealing, so all models oscillate at late stages, and (2) Mo²BERTa's higher effective capacity per parameter causes it to overfit the small TinyStories corpus (~199K tokens) faster. By the 600T endpoint, each unique token has been seen roughly 50 times by Mo²BERTa. A trapezoidal LR schedule or early stopping at 500T would be appropriate in a production setting.

### 100 TFLOP Interim Results (for reference)

|Metric|Mo²BERTa|IsoParam (L4)|IsoDepth (L7)|
|---|:-:|:-:|:-:|
|Best Val Loss|**2.8097**|2.8797|2.8335|
|Best Val Acc (MLM)|**50.69%**|46.42%|49.74%|
|Tokens seen D|1.67M|1.72M|1.51M|

At 100T, Mo²BERTa leads on both metrics. The +4.27 point accuracy gap over IsoParam at equal parameter count is the clearest early signal that adaptive depth allocation is working.

## Routing Behavior Analysis 

> **Note on the uniform exit distribution panel (25%/25%/25%/25%):** This is by architectural design, not a finding. The capacity schedule hardcodes the _fraction_ of tokens exiting at each step (25% of the remaining pool per router). The uniform bars simply confirm the schedule is being respected. What the schedule does _not_ determine is _which_ tokens exit, that is what the router learns, and is what the depth gap and heatmap panels show. A token-choice routing variant (no fixed budget) would produce a learned, non-uniform exit distribution.

### Step 400
![image](https://cdn-uploads.huggingface.co/production/uploads/667da868d653c0b02d6a2399/fzJLOzcqqsTrThIIzz1LG.png)

Routing analysis at step 400 confirms the router has learned semantically meaningful token-difficulty discrimination without any explicit token-type supervision.

**[MASK] tokens receive substantially deeper processing.** Across a full batch, `[MASK]` tokens achieve a mean recursion depth of **3.87 out of 4** (n=147), while non-`[MASK]` tokens average **2.27** (n=877). This 1.60-recursion gap representing ~70% additional compute directed to masked positions emerges purely from the MLM training signal.

**The router is well-calibrated and decisive.** Score distributions for all three routers (operating on 128, 96, and 64 active tokens respectively) are strongly bimodal, with mass concentrated near 0.0 and 1.0 and minimal density in the 0.2–0.8 range. This matches the paper's reported behavior for expert-choice routing with auxiliary loss, and indicates the router is making confident discrete-like decisions rather than hedging. This bimodality is present from step ~200 onward.

**Routing tracks semantic salience, not just mask position.** In qualitative inspection, tokens that stay active deepest are not only `[MASK]` positions but also morphologically complex subwords (e.g., `##yin`) and thematically loaded content words in context (`piracy`, `dangerous`). Common punctuation and function words exit earliest. This suggests the router has learned a proxy for contextual difficulty rather than a simple token-identity rule.

### Trained
#### At Inference (Post-Training, TinyStories Val Sample)

Routing behavior on a held-out val sequence after the full 600T run:

- `[MASK]` tokens: mean depth **3.38 / 4** (n=21)
- Non-`[MASK]` tokens: mean depth **2.33 / 4** (n=107)

The depth gap (1.05 recursions) is smaller than at training time (1.60). This is expected: during training the aux loss actively pushes the router toward bimodal confidence, while at inference the router makes live decisions without that gradient pressure. The gap is nonetheless consistent and meaningful.

Router score distributions remain strongly bimodal at inference, confirming the routing behavior is genuinely learned and not an artifact of the training objective alone.

![output](https://cdn-uploads.huggingface.co/production/uploads/667da868d653c0b02d6a2399/zeGD189yymGQomBkYG_qe.png)

#### MLM Prediction Quality (Same Inference Sample)

18 masked positions evaluated. Selected examples:

|Position|True token|Rank|Notes|
|---|---|---|---|
|13|`gave`|**1**|Correct, confident (10.71 vs 7.55 next)|
|16|`big`|**1**|Correct, semantically tight (`small`, `little` as alternatives)|
|70|`from`|**1**|Correct, high margin (12.63 vs 7.88)|
|100|`##t`|**1**|Correct subword completion, very high confidence (17.11)|
|7|`to`|2|`on` ranked first, both valid prepositions in context|
|111|`man`|2|`boy` ranked first, semantically close|
|28|`balanced`|miss|Rare word, `are` at rank 1; model hasn't seen `balanced` enough in TinyStories (n=49)|
|30|`tray`|miss|`table` at rank 1, reasonable guess, `tray` is low-frequency (n=180)|
|0|`cat`|miss|`her` at rank 1, insufficient left context at position 0|

Overall: true token in top-5 for ~12/18 positions (~67%), rank-1 accuracy ~8/18 (~44%). Misses cluster on rare or domain-specific words (`balanced`, `tray`) and position-0 tokens with no left context; both expected failure modes for a tiny model trained only on simple stories.

## Known Limitations and Unoptimized Components

### Compute and Scale

This is a PoC trained on a consumer laptop GPU (NVIDIA 930M, Compute Capability 5.0) with a maximum compute budget of 600 TFLOPs and approximately 11M tokens seen. No claims about scalability, generalization to other domains, or comparison to full-scale pretrained encoders are made or implied. The MoR paper's main results operate at 135M–1.7B parameters across hundreds of billions of tokens; this prototype is smaller in both dimensions.

### Constant Learning Rate

All three models (Mo²BERTa and both baselines) use a constant learning rate of 3e-4 with no warmup or cosine decay. The MoR paper uses a trapezoidal schedule (linear warmup → constant → linear decay). This simplification means the absolute loss values are not directly comparable to the paper's reported numbers. The equal treatment of all three models preserves the *relative* comparison validity, and the argument that "MoR wins even without specialized LR tuning" is a conservative one, but a full reproduction would require schedule matching.

### Python Loop Gather-Scatter (Wall-Clock vs. FLOP Efficiency)

The attention gather-scatter operation which involves selecting active tokens, running sparse attention over them, and scattering results back is implemented as a **Python-level per-batch-item loop** in `BidirectionalGQA._attn_skip`. This is theoretically correct (the FLOP counts are accurate for the active token subsets) but does **not** translate to wall-clock speedup on current GPU hardware. The reason is that PyTorch's CUDA kernels are optimized for dense, uniform-shape tensor operations; variable-length sparse loops fragment GPU parallelism and introduce Python interpreter overhead.

Realizing the theoretical throughput gains would require custom CUDA kernels (e.g., variable-length FlashAttention, as used in the MoR paper's throughput experiments with continuous depth-wise batching). This is out of scope for a PoC on consumer hardware. The FLOP accounting in the training logs reflects the theoretical active-token compute, not the wall-clock cost.

### MLP Gather-Scatter

Similarly, the MLP gather-scatter (`TransformerBlock._forward_skip`) uses `torch.where` with a flat gather over `(active_b, active_t)` indices. This is more GPU-friendly than the attention loop (the gathered tensor is processed as a single dense matmul), but still incurs indexing overhead that a fused kernel would eliminate.

### Context Isolation in Recursive Attention

Because attention at recursion depth r operates only over tokens still active at depth r, tokens that exit early stop contributing their Keys and Values to the attention context of deeper tokens. This means, for example, a function word that exits at recursion 1 does not appear in the KV context for a `[MASK]` token still processing at recursion 3. The unique final layer (which operates on all tokens with `active_mask=None`) provides one round of global mixing after all recursions complete, partially mitigating this, but the fundamental limitation remains: iterative co-refinement between easy and hard tokens is reduced compared to a dense model.

A potential future improvement: exited tokens could contribute frozen Keys and Values (from their exit-step representations) to deeper recursion steps while skipping their Query computation, preserving context density while still saving the dominant FLOP cost (QKV projections and attention over the active set).

### No LM Head Temperature / Calibration

The MLM accuracy metric uses argmax over raw logits. No temperature calibration or probability normalization beyond softmax is applied. The auxiliary BCE loss on router scores could in principle shift the output distribution slightly; this has not been analyzed.

### Training Data Domain

The model is trained exclusively on TinyStories-valid, a synthetic dataset of simple English children's stories. Performance on technical, formal, scientific, or multilingual text is expected to be poor. The dataset also has limited vocabulary diversity and sentence complexity relative to pretraining corpora used by production encoders.

## What This PoC Does and Does Not Prove

**Supported claims:**

- The MoR mechanism transfers from autoregressive decoders to bidirectional encoders without fundamental architectural barriers.
- Expert-choice routing with BCE auxiliary loss produces well-calibrated, bimodal, semantically meaningful routing within ~200 training steps on small-scale data.
- Mo²BERTa matches a larger IsoDepth baseline (14% more parameters) on val loss at best checkpoint, with fewer parameters.
- The router learns to allocate more recursive depth to harder tokens ([MASK] positions and complex subwords) using only the MLM training signal with no token-type supervision.

**Not supported / out of scope:**

- Wall-clock inference speedup (requires custom CUDA kernels, not implemented here).
- Scaling behavior (one model size, one dataset, one compute budget).
- Comparison to production encoders (BERT-base, RoBERTa, DeBERTa, etc.).
- Generalization beyond the TinyStories domain.
- Optimal hyperparameter configuration (constant LR, fixed α=0.1, no ablations over N_recursion or capacity schedule shape).
- Late-stage training stability (constant LR is a known limitation)

## Known TODOs / Future Work

**Architecture**
- [x] Frozen-KV variant: exited tokens contribute KV but skip Q projection  -> [Mo²BERTa-v2-proto](https://huggingface.co/gbyuvd/Mo2BERTa-v2-proto)
- [ ] Test token-choice routing variant for comparison

**Performance**
- [ ] Replace Python gather-scatter loop with variable-length FlashAttention kernel 
  - [x] We can't do this at the time being, but decided to do padded batched SDPA   -> [Mo²BERTa-v2-proto](https://huggingface.co/gbyuvd/Mo2BERTa-v2-proto)
- [ ] Benchmark wall-clock throughput vs. dense baseline after optimization

**Experiments**
- [x] 600 TFLOP final PoC run
- [x] 100 TFLOP run with optimized model
- [x] 600 TFLOP run with optimized model and baselines -> [Mo²BERTa-v2-proto](https://huggingface.co/gbyuvd/Mo2BERTa-v2-proto)
- [ ] LR schedule matching (trapezoidal warmup-decay per paper)
- [ ] Scale up: larger model size, fuller dataset

**Code**
- [ ] Modularize into model.py / train.py / router.py / dataset.py
- [ ] Add config dataclass to replace module-level constants
- [ ] Unit tests for router capacity schedule and FLOP estimator


## Citation

If you use this prototype or the encoder adaptation strategy in your research, please cite the original MoR paper and this repository:


```bibtex
@misc{bae2025mixtureofrecursionslearningdynamicrecursive,
      title={Mixture-of-Recursions: Learning Dynamic Recursive Depths for Adaptive Token-Level Computation}, 
      author={Sangmin Bae and Yujin Kim and Reza Bayat and Sungnyun Kim and Jiyoun Ha and Tal Schuster and Adam Fisch and Hrayr Harutyunyan and Ziwei Ji and Aaron Courville and Se-Young Yun},
      year={2025},
      eprint={2507.10524},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2507.10524}, 
}
```

```bibtex
@software{mo2berta_proto,
  author  = {GP Bayu},
  title   = {{Mo²BERTa}: Mixture-of-Recursions for Bidirectional MLM},
  url     = {https://huggingface.co/gbyuvd/Mo2BERTa-proto},
  version = {0.1},
  year    = {2026},
}
```

```bibtex
@misc{eldan2023tinystoriessmalllanguagemodels,
      title={TinyStories: How Small Can Language Models Be and Still Speak Coherent English?}, 
      author={Ronen Eldan and Yuanzhi Li},
      year={2023},
      eprint={2305.07759},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2305.07759}, 
}
```
## Contact

For questions about this prototype implementation, please open an issue in the source repository.
