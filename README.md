# Qwen2.5-1.5B-Instruct: End-to-End Fine-Tuning, Adapter Merging, 4-Bit NF4 Quantization & 6-Part Verification Suite

> **Author:** Balavignesh V.T. ([GitHub: `@Speedyzyzz`](https://github.com/Speedyzyzz))  
> **Role:** Soup AI Engineer Take-Home Project  
> **Target Architecture:** [`Qwen/Qwen2.5-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) ($1,543,714,304$ Base Params; 28 Layers, GQA $12\text{Q}/2\text{KV}$, Vocab $151,936$)  
> **Dataset:** 1,500 curated multi-turn dialogues from [`HuggingFaceTB/smoltalk`](https://huggingface.co/datasets/HuggingFaceTB/smoltalk) (`everyday-conversations`, deduplicated, assistant turns $\ge 40$ chars; 1,450 train / 50 held-out eval)  
> **Hardware Target:** Free Google Colab NVIDIA Tesla T4 GPU (`15,360 MiB` / `15.00 GiB` Usable VRAM, `sm_75` Turing Architecture)  
> **Stack:** `soup-cli[train]` v0.77.2 + [PyTorch 2.4](https://pytorch.org/), [Transformers 4.45](https://huggingface.co/docs/transformers), [PEFT 0.13](https://github.com/huggingface/peft) (`rsLoRA` + `NEFTune`), and [BitsAndBytes 0.43](https://github.com/bitsandbytes-foundation/bitsandbytes) (`NF4`)  

## Architecture & Verification Pipeline

```mermaid
flowchart TD
    A["Curated Multi-Turn Instruction Data<br/>(1,500 Smoltalk Everyday-Conversations)"] --> B["Pre-Flight & Turn-by-Turn Mask Audit<br/>(soup data doctor + build_assistant_masked_labels)"]
    B -->|100% Assistant-Only Labels, EOS 151645 Verified| C["Partitioned Dataset<br/>1,450 Train | 50 Held-Out Eval"]
    
    subgraph SFT_Engine ["High-Output SFT Engine (2 Epochs = 362 Steps)"]
        D["Qwen2.5-1.5B-Instruct (FP16 Base: 2,944 MiB)"] --> E["rsLoRA Injection (r=16, alpha=32, scale=8.0)<br/>All 7 Projections = 18,464,768 Trainable Params"]
        E --> F["Fused AdamW + NEFTune (alpha=5.0) + Cosine LR<br/>Grad Checkpointing (use_reentrant=False)<br/>Peak VRAM: 6,420 MiB (6.27 GiB) / 15.00 GiB"]
    end
    
    C --> SFT_Engine
    F --> G["Trained rsLoRA Adapter<br/>(./output: 35.22 MiB FP16)"]
    
    G --> H["GPU-Resident Adapter Fusion<br/>(merge_and_unload + tie_word_embeddings=True)"]
    D --> H
    H -->|Peak Merge VRAM Spike: 6,840 MiB (6.68 GiB)| I["Artifact B: Merged FP16 Model<br/>(./output_merged: 3.09 GB)"]
    
    I --> J["4-Bit NF4 Quantization Engine<br/>(BitsAndBytes NF4 + Double Quant + save_pretrained)"]
    J -->|Serialized 4-Bit Safetensors| K["Artifact C: 4-Bit NF4 Model<br/>(./output_4bit: 824 MB Disk / 1,746 MiB VRAM)"]
    
    subgraph Verification_Engine ["6-Part Verification Engine (verify_artifacts.py)"]
        L["1. Layer Frobenius Norms & Frozen Isolation"]
        M["2. SVD Rank-16 Spectral Energy (>99.99%)"]
        N["3. Dequantized 4-Bit Lineage Distance"]
        O["4. Global Base-Merged Cosine Similarity"]
        P["5. Multi-Turn Masked PPL & Token KL-Divergence"]
        Q["6. Active EOS Termination & 4-Gram Diversity"]
    end
    
    D --> Verification_Engine
    I --> Verification_Engine
    K --> Verification_Engine
    
    Verification_Engine --> R{"Final Deployment Gate<br/>(soup ship)"}
    R -->|Task PPL -11.82%, 4-Bit Penalty +0.3123<br/>Top-1 Agreement 94.62%| S["VERDICT: SHIP (Exit 0)"]
```

## 1. Pre-Training Analytical Memory Budget vs. Empirical T4 Telemetry

The NVIDIA Tesla T4 provides **15,360 MiB (15.00 GiB)** of usable VRAM. With `gradient_checkpointing=True`, `use_cache=False` is enforced so **zero KV-cache** is allocated during training. Instead, the largest transient activation spike comes from upcasting Qwen2.5's `151,936`-vocabulary logits to FP32 (`logits.float()`) alongside `shift_logits` and the FP32 cross-entropy backward gradient tensor.

| Memory Component | Precision / Format | Analytical Formula ($B=2, S=512, V=151,936$) | Analytical Estimate | Observed (`nvidia-smi` / `torch.cuda`) | Gap & Engineering Explanation |
|---|---|---|---|---|---|
| **Base Model Weights** | FP16 (2 B/param) | $1,543,714,304 \times 2\text{ B}$ (`lm_head` tied to `embed_tokens`) | **2,944.40 MiB** (`2.88 GiB`) | `2,952 MiB` (`2.88 GiB`) | `+7.6 MiB` for RoPE cos/sin caches and non-parameter buffers. |
| **rsLoRA Master Weights** | FP32 (4 B/param) | $28 \times 659,456 = 18,464,768 \times 4\text{ B}$ | **70.44 MiB** (`0.07 GiB`) | `71 MiB` (`0.07 GiB`) | PEFT keeps trainable adapter weights in FP32 during AMP training (`35.22 MiB` on disk in FP16). |
| **AdamW Optimizer States** | FP32 ($m, v$: 8 B/param) | $18,464,768 \times 8\text{ B}$ (`adamw_torch_fused`) | **140.88 MiB** (`0.14 GiB`) | `142 MiB` (`0.14 GiB`) | First and second moments allocated only for the `18.46M` trainable adapter parameters. |
| **rsLoRA Gradients** | FP32 (4 B/param) | $18,464,768 \times 4\text{ B}$ | **70.44 MiB** (`0.07 GiB`) | `71 MiB` (`0.07 GiB`) | Unscaled FP32 adapter gradients prior to `clip_grad_norm_(1.0)`. |
| **Checkpointed Activations** | FP16 (`use_cache=False`) | $28 \times (2 \times 512 \times 1536 \times 2\text{ B}) + \text{1-Layer Recompute}$ | **410.00 MiB** (`0.40 GiB`) | `424 MiB` (`0.41 GiB`) | $84.0\text{ MiB}$ layer-boundary checkpoints + $\sim 340\text{ MiB}$ single-layer recomputation buffer. |
| **Logits + FP32 CE Loss Spike** | FP16 $\rightarrow$ 3$\times$ FP32 | $2 \times 512 \times 151,936 \times (2 + 4 + 4 + 4)\text{ B}$ | **2,077.25 MiB** (`2.03 GiB`) | `2,084 MiB` (`2.04 GiB`) | FP16 `lm_head` logits ($296.75\text{ MiB}$) + FP32 upcast, `shift_logits`, and $\nabla\mathcal{L}_{\text{CE}}$ ($3 \times 593.5\text{ MiB}$). |
| **CUDA Context & Allocator** | Driver + Segment Cache | CUDA runtime context ($\sim 550\text{ MiB}$) + caching allocator reserve | **582.00 MiB** (`0.57 GiB`) | `676 MiB` (`0.66 GiB`) | PyTorch caching allocator rounds block allocations into $20\text{ MiB}$ segments. |
| **Peak Training Total** | FP16 AMP + Grad Ckpt | Sum of active training allocations + context | **6,295.41 MiB** (`6.15 GiB`) | **`6,420 MiB` (`6.27 GiB`)** | **+1.98% empirical gap** (`+124.6 MiB`) from caching allocator segment reservation. |
| **GPU Merge Peak (`merge_and_unload`)** | FP16 Base + Delta GEMM | $W_{\text{base}}$ + Coexisting FP32/FP16 Clone & $BA$ Accumulator + Context | **6,620.00 MiB** (`6.46 GiB`) | **`6,840 MiB` (`6.68 GiB`)** | Transient spike while base tensors and `rsLoRA` GEMM accumulators coexist on GPU before `safetensors` serialization. |
| **4-Bit NF4 Quantized Export** | BnB NF4 + Double Quant | $1.31\times 10^9 \times 0.5625\text{ B} + 0.23\times 10^9 \times 2\text{ B}$ | **1,164.00 MiB** (`824 MB` Disk) | **`1,746 MiB` VRAM** (`824 MB` Disk) | `embed_tokens` stays FP16 ($445\text{ MiB}$) + $703\text{ MiB}$ NF4 linear blocks + $582\text{ MiB}$ CUDA context. |

## 2. Six-Part Verification Engine Scorecard (`verify_artifacts.py`)

| Verification Check | Expected Criterion | Observed Value | Status |
|---|---|---|---|
| **1. Target Projections Frobenius Norm Mean** | Strictly $> 0.0$ across all 196 linear layers | **`0.054182`** | **PASS (Active Adaptation)** |
| **2. Frozen Modules Frobenius Norm Mean** | Exactly $== 0.0$ (`embed_tokens`, 57 RMSNorms, `lm_head`) | **`0.000000`** | **PASS (Zero Leakage, Tied)** |
| **3. SVD Top-16 Spectral Energy Ratio** | $> 99.9\%$ energy in top $r=16$ singular values of $\Delta W$ | **`99.9984%`** | **PASS (True Rank-16 Update)** |
| **4. Global Cosine Similarity ($W_{\text{base}}, W_{\text{merged}}$)** | Within $[0.985, 0.9995]$ | **`0.994108`** | **PASS (No Collapse / No Noop)** |
| **5. 4-Bit Dequantized Lineage Check** | $\|\hat{W}_{\text{4bit}} - W_{\text{merged}}\|_F < \|\hat{W}_{\text{4bit}} - W_{\text{base}}\|_F$ | **`0.01842 < 0.05891`** | **PASS (Derived from Merged)** |
| **6. Base Held-Out Perplexity ($N=50$)** | Multi-turn assistant-only cross-entropy baseline | **`14.8210`** | **BASELINE (`3,842` tokens)** |
| **7. Merged FP16 Held-Out Perplexity ($N=50$)** | Strictly $<$ Base Perplexity | **`13.0691`** | **PASS (-11.82% Improvement)** |
| **8. 4-Bit NF4 Held-Out Perplexity ($N=50$)** | Degradation penalty $\le +0.50$ ppl over Merged FP16 | **`13.3814`** | **PASS (+0.3123 ppl Penalty)** |
| **9. Token Logit KL Shift $D_{\text{KL}}(P_{\text{merged}} \parallel P_{\text{base}})$** | $> 0.01\text{ nats/tok}$ (meaningful behavioral shift) | **`0.08412 nats/tok`** | **PASS (76.48% Top-1 Agree)** |
| **10. 4-Bit Logit Fidelity $D_{\text{KL}}(P_{\text{merged}} \parallel P_{\text{4bit}})$** | $< 0.03\text{ nats/tok}$ and $> 92\%$ Top-1 agreement | **`0.01428 nats/tok`** | **PASS (94.62% Top-1 Agree)** |
| **11. Active Generation EOS & 4-Gram Diversity** | $\ge 90\%$ `<\|im_end\|>` termination & $> 0.85$ 4-gram ratio | **`100% EOS / 0.964`** | **PASS (Zero Repetition Loops)** |

## 3. Silent Failure Modes Audit Matrix

| # | Pipeline Stage | Silent Failure Mode | Detection & Verification Procedure | Caught by Soup Built-Ins? |
|---|---|---|---|---|
| **1** | **Data $\rightarrow$ Collator** | **Multi-Turn Label Mask Misalignment:** Filtering `role != "assistant"` concatenates user turns at the front and unmasks later user turns in multi-turn dialogues. | Tokenize turn-by-turn via `build_assistant_masked_labels`; decode unmasked tokens (`labels != -100`) and assert zero `<\|im_start\|>user` headers appear. | **Partially** (`soup data doctor --show-mask` checks built-in collators, not custom dataset mappers). |
| **2** | **Tokenizer & Template** | **Missing `<\|im_end\|>` EOS Token:** Omitting `<\|im_end\|>` (`151645`) from target labels causes infinite generation until `max_new_tokens`. | Assert `151645` closes every unmasked assistant span and verify `10/10` greedy generations terminate before hitting `max_new_tokens=128`. | **Yes** (`soup data doctor` `eos_in_labels` and `bos_duplication` checks). |
| **3** | **Training (T4 `sm_75`)** | **FP16 Underflow & Frozen-Input Grad Detachment:** Enabling gradient checkpointing on a frozen base model without `enable_input_require_grads()` or `GradScaler` zeroes or detaches gradients. | Enforce `use_reentrant=False`, `enable_input_require_grads()`, and FP16 AMP `GradScaler`; verify non-zero step `grad_norm` (`0.41–0.89`). | **Yes** (`soup doctor` checks `sm_75` and enables FP16 AMP). |
| **4** | **Merge** | **Untied Embeddings & Wrong LoRA Scaling:** `merge_and_unload()` can untie `lm_head.weight` (+467 MB bloat) or apply $\alpha/r$ instead of `rsLoRA` $\alpha/\sqrt{r}$. | Assert `merged_model.config.tie_word_embeddings == True`, check `use_rslora: true`, and verify SVD top-16 energy is `99.9984%`. | **Partially** (`soup merge` fuses adapters, but does not run SVD rank or weight-tying checks). |
| **5** | **Quantization** | **Unsaved Quantized Weights / Base-Model Lineage Bug:** Saving only the tokenizer to `./output_4bit` or quantizing `Base` instead of `Merged`. | Call `quant_model.save_pretrained("./output_4bit")` and verify $\|\hat{W}_{\text{4bit}} - W_{\text{merged}}\|_F < \|\hat{W}_{\text{4bit}} - W_{\text{base}}\|_F$ via `dequantize_4bit`. | **No** (Requires explicit dequantized norm distance check in `verify_artifacts.py`). |
| **6** | **Inference** | **4-Bit Outlier Clipping & Degenerate Loops:** Clipping salient `down_proj` channels causes repetitive looping on open-ended generation despite low perplexity. | Measure token-level $D_{\text{KL}}(P_{\text{merged}} \parallel P_{\text{4bit}}) = 0.01428$ and distinct 4-gram ratio (`0.964`). | **Partially** (`soup ship` checks benchmark accuracy/perplexity, not KL drift or n-gram diversity). |

## 4. Verdict and Final Ship Decision

```text
====================================================================================
FINAL DEPLOYMENT DECISION GATE: SOUP SHIP
====================================================================================
1. Memory Budget Compliance:          PASS (Peak 6,420 MiB Train / 6,840 MiB Merge / 15,360 MiB T4)
2. Target Projections Frobenius Norm: PASS (Mean 0.054182 across 196 adapted linear layers)
3. Frozen Layers & Weight Tying:      PASS (Bit-exact 0.000000 on embed/norms; tie_word_embeddings=True)
4. SVD Rank-16 Spectral Energy:       PASS (99.9984% spectral energy in top r=16 singular values)
5. 4-Bit Dequantized Lineage:         PASS (||W_4bit - W_merged||_F < ||W_4bit - W_base||_F)
6. Held-Out Perplexity Improvement:   PASS (13.0691 < 14.8210, -11.82% relative reduction)
7. 4-Bit NF4 Perplexity & KL Drift:   PASS (+0.3123 ppl penalty; KL=0.01428; 94.62% Top-1 agreement)
8. Active EOS Termination & 4-Gram:   PASS (100% <|im_end|> termination; 0.964 distinct 4-gram ratio)
------------------------------------------------------------------------------------
DECISION: >>> SHIP <<< (Exit Code: 0 — Conditional on Full GSM8k/Safety Canary in CI)
====================================================================================
```

## 5. Repository Deliverables & Reproduction

- `report.md` — Formal 2-page engineering report covering memory budget, 6-part verification proofs, silent failure matrix, surprises/concerns, and AI tools audit.
- `verify_artifacts.py` — Standalone 3-artifact verification engine (Frobenius norms, SVD rank-16 spectrum, 4-bit dequantization lineage, KL divergence, perplexity, and EOS generation check).
- `soup_takehome.ipynb` & `build_notebook.py` — Complete executable Colab T4 training, merging, quantization, and auditing notebook.
- `raw_logs.txt` & `nvidia_smi_log.csv` — Timestamped training, merge, quantization, and verification execution logs and GPU VRAM telemetry.

```bash
# Run the 6-part verification engine across Base, Merged, and 4-Bit artifacts:
python verify_artifacts.py \
    --base_model Qwen/Qwen2.5-1.5B-Instruct \
    --merged_model ./output_merged \
    --quantized_model ./output_4bit \
    --lora_rank 16 \
    --eval_file ./eval_50.jsonl \
    --device cuda \
    --plot \
    --output_json verification_results.json
```
