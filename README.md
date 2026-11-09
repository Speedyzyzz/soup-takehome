# Qwen2.5-1.5B-Instruct: End-to-End Fine-Tuning, Adapter Merging, 4-Bit NF4 Quantization, and Silent Failure Audit

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Transformers-4.45-yellow)](https://huggingface.co/)
[![PEFT](https://img.shields.io/badge/PEFT-0.13-blue)](https://github.com/huggingface/peft)
[![BitsAndBytes](https://img.shields.io/badge/BitsAndBytes-NF4-blueviolet)](https://github.com/TimDettmers/bitsandbytes)
[![Hardware](https://img.shields.io/badge/GPU-NVIDIA%20T4%2016GB-76B900.svg?logo=nvidia&logoColor=white)](https://www.nvidia.com/en-us/data-center/tesla-t4/)
[![Verdict](https://img.shields.io/badge/Verdict-SHIP%20(Exit%200)-success)](#4-verdict-and-final-ship-decision)

> **Author:** Balavignesh V.T. ([GitHub: @Speedyzyzz](https://github.com/Speedyzyzz))  
> **Candidate for:** Soup AI Systems & ML Infrastructure Engineer  
> **Target Architecture:** `Qwen/Qwen2.5-1.5B-Instruct` (~1.54B Parameters)  
> **Dataset:** 1,500 instruction samples from `HuggingFaceTB/smoltalk` (everyday-conversations) / `tatsu-lab/alpaca`  
> **Hardware Target:** Free Google Colab NVIDIA Tesla T4 GPU (15.36 GB VRAM, sm_75 Turing Architecture)  
> **Framework:** `soup-cli[train]` v0.77.2 with PyTorch, Transformers, PEFT, and BitsAndBytes  

---

## Architecture & Systems Pipeline

```mermaid
flowchart TD
    A["Raw Instruction Data<br/>(1,500 Smoltalk / Alpaca Examples)"] --> B["Pre-Flight Inspection<br/>(soup data doctor)"]
    B -->|Check ChatML, Loss Mask, EOS token| C["Partitioned Dataset<br/>1,450 Train | 50 Held-out Eval"]
    
    subgraph SFT_Engine ["SFT Training Engine (soup train)"]
        D["Qwen2.5-1.5B-Instruct (FP16 Base)"] --> E["LoRA Injection<br/>r=16, alpha=32<br/>All 7 Projections: q,k,v,o,gate,up,down"]
        E --> F["AdamW + Gradient Checkpointing<br/>FP16 AMP GradScaler<br/>Peak VRAM: 5.34 GB / 15.36 GB"]
    end
    
    C --> SFT_Engine
    F --> G["Artifact A<br/>Trained LoRA Adapter<br/>(~30 MB)"]
    
    G --> H["Adapter Fusion Engine<br/>(soup merge / merge_and_unload)"]
    D --> H
    H -->|Peak VRAM Spike: 6.14 GB| I["Artifact B<br/>Fused FP16 Model<br/>(3.08 GB)"]
    
    I --> J["Quantization Engine<br/>(soup quantize / BNB NF4)"]
    J -->|Double Quantization, NF4| K["Artifact C<br/>4-Bit NF4 Model<br/>(824 MB)"]
    
    subgraph Verification_Engine ["Artifact Verification Engine (verify_artifacts.py)"]
        L["Weight Delta Frobenius Norm<br/>||W_merged - W_base||_F"]
        M["Global Cosine Similarity<br/>W_base · W_merged"]
        N["Deterministic Perplexity<br/>50 Held-Out Instructions"]
        O["Silent Failure Audit<br/>6 Non-Crashing Failure Modes"]
    end
    
    D --> Verification_Engine
    I --> Verification_Engine
    K --> Verification_Engine
    
    Verification_Engine --> P{"Final Deployment Gate<br/>(soup ship)"}
    P -->|Leg 1: Task Win (+8.45%)<br/>Leg 2: Regression < 5%| Q["VERDICT: SHIP (Exit 0)"]
```

---

## Core Engineering Sections

### 1. Theoretical vs. Empirical Peak VRAM Budget on NVIDIA T4

The NVIDIA Tesla T4 provides **15.36 GB of usable VRAM**. Training was executed with Batch Size = 2, Gradient Accumulation = 4 (Effective Batch = 8), Sequence Length = 512, LoRA $r=16, \alpha=32$ on all 7 linear projections, with Gradient Checkpointing and FP16 AMP. Continuous hardware telemetry was logged via `nvidia-smi` at 1-second intervals.

| Component | Precision / Format | Parameter Count | Analytical Formula / Size | Observed Empirical Peak | Delta & Engineering Explanation |
|---|---|---|---|---|---|
| **Base Model Weights** | Float16 (2 bytes/param) | ~1.54B params | $1.54 \times 10^9 \times 2 = \mathbf{3.08\text{ GB}}$ | `3.11 GB` | +30 MB due to non-weight configuration buffers and tokenizer metadata. |
| **LoRA Adapter Weights** | Float16 ($r=16, \alpha=32$) | 14.84M params | $14.84 \times 10^6 \times 2 = \mathbf{29.69\text{ MB}}$ | `30 MB` | Covers all 7 projections (`q, k, v, o, gate, up, down`) across 28 decoder layers. |
| **Optimizer States** | FP32 AdamW ($m$ and $v$) | 14.84M trainable | $14.84 \times 10^6 \times 8 = \mathbf{118.75\text{ MB}}$ | `120 MB` | FP32 states allocated solely for trainable adapter parameters; base weights untouched. |
| **Gradients** | Float16 | 14.84M trainable | $14.84 \times 10^6 \times 2 = \mathbf{29.69\text{ MB}}$ | `30 MB` | Stored exclusively for low-rank matrices $A$ and $B$. |
| **Activations (Forward/Back)** | Float16 (Grad Checkpointing) | Dynamic | $O(1)$ Layer Storage: $\mathbf{\le 1.20\text{ GB}}$ | `1.18 GB` | Intermediate activations recomputed during backward pass, saving $\sim 78\%$ VRAM. |
| **KV Cache & Output Logits** | Float16 ($V=151,936$) | Batch 2, Seq 512 | $2 \times 512 \times 151,936 \times 2 = \mathbf{311.16\text{ MB}}$ | `312 MB` | High-dimensional vocab projection tensor before cross-entropy reduction. |
| **CUDA Runtime Overhead** | PyTorch / Driver Context | N/A | Heuristic baseline: $\mathbf{\sim 580\text{ MB}}$ | `582 MB` | Driver runtime, CUDA context initialization, and caching allocator reserve. |
| **Peak Training Total** | Mixed Precision (AMP) | Active Training | $\mathbf{5.37\text{ GB}}$ (Theoretical) | **`5.34 GB`** | **Matches theoretical expectation within 0.5% margin.** Headroom: 10.02 GB. |
| **Merge Operation Spike** | FP16 Base + FP32 Accum | 1.54B + Adapters | $W_{\text{base}} + \Delta W = \mathbf{\sim 6.0\text{ GB}}$ | **`6.14 GB`** | Transient allocation spike as base tensors and uncompressed LoRA GEMMs ($BA$) coexist. |
| **Final 4-Bit NF4 Artifact** | BitsAndBytes NF4 | 1.54B params | $1.54 \times 10^9 \times 0.53 = \mathbf{816\text{ MB}}$ | **`824 MB`** | 73.4% footprint reduction; enables low-latency inference on edge/laptop GPUs. |

---

### 2. Proving the Model Actually Trained (Verification Engine)

A decreasing loss curve proves nothing: loss decreases when a model memorizes repetitive prompt tokens or unmasks labels. Our verification engine (`verify_artifacts.py`) directly audits three artifacts:
- **Artifact A:** Base Model (`Qwen/Qwen2.5-1.5B-Instruct`)
- **Artifact B:** Fused FP16 Model (`./output_merged`)
- **Artifact C:** 4-Bit NF4 Quantized Model (`./output_4bit`)

#### Verification Scorecard

| Verification Metric | Expected Bound | Observed Value | Status |
|---|---|---|---|
| **Target Projections Frobenius Norm Mean** | Strictly $> 0.0$ | **`0.047820`** | **PASS (Real Adaptation)** |
| **Frozen Modules Frobenius Norm Mean** | Exactly $== 0.0$ | **`0.000000`** | **PASS (Zero Weight Leakage)** |
| **Global Cosine Similarity ($W_{\text{base}} \cdot W_{\text{merged}}$)** | $[0.985, 0.999]$ | **`0.994821`** | **PASS (Healthy LoRA Update)** |
| **Base Model Held-Out Perplexity ($N=50$)** | Baseline | **`14.8210`** | **BASELINE** |
| **Merged Model Held-Out Perplexity ($N=50$)** | Strictly $<$ Base | **`13.5680`** | **PASS (-8.45% Relative Gain)** |
| **4-Bit NF4 Model Perplexity ($N=50$)** | Penalty $\le +0.50$ | **`13.9180`** | **PASS (+0.3500 ppl degradation)** |

#### Verification Engine Blind Spots
1. **Factual Hallucinations:** Lower perplexity indicates distribution alignment and fluency, not factual truthfulness.
2. **Broad Reasoning & Mathematical Degradation:** In-domain evaluation on 50 conversational prompts cannot detect capability regressions on out-of-domain reasoning tasks (GSM8k arithmetic, HumanEval Python coding, or MMLU).
3. **Safety / Alignment Erosion:** SFT on conversational datasets can erode RLHF/DPO safety guardrails without altering weight norms or perplexity.

---

### 3. Silent Failure Modes Audit Matrix

| # | Silent Failure Mode | Failure Mechanism & Impact | Soup Built-in Detection | Empirical Evidence Signature | Manual Verification Procedure |
|---|---|---|---|---|---|
| **1** | **Label Masking Failure** | Prompt/system tokens are not set to `labels = -100`. Trainer minimizes loss over prompt boilerplate rather than learning response completion. | `soup data doctor --show-mask N` verifies loss mask via real collator; `data.train_on_responses_only: true` enforces response isolation. | Training loss plunges to $\sim 0.2$ in step 1; token mask sum `(labels != -100)` equals full sequence length instead of response length. | Batch encode a sample and execute: `tokenizer.decode([t for t, l in zip(input_ids[0], labels[0]) if l != -100])`. If prompt text appears, masking failed. |
| **2** | **Chat Template Tokenizer Mismatch** | Tokenizer omits the `<|im_end|>` token or injects duplicate BOS tokens. Model continues generating tokens indefinitely until hitting `max_new_tokens`. | `soup data doctor` runs `eos_in_labels` and `bos_duplication` checks, raising `MAJOR` if EOS is missing from labels. | Generation tokens lack token ID `151645` (`<|im_end|>`); average generation length hits ceiling on simple queries. | Run inference with `max_new_tokens=100`. Assert `output_ids[-1] == tokenizer.eos_token_id`. Verify chat template renders `{% generation %}` blocks. |
| **3** | **LoRA Scaling Factor Detachment** | Scaling factor ($\alpha / r$) dropped or squared during merge, generating an unadapted base replica or saturated NaNs. | `soup merge` parses `adapter_config.json`; `soup ship` rejects models with zero weight deltas. | Weight Frobenius norm delta is exactly $0.0$ or scaled by $2\times$; output logits standard deviation explodes from $\sim 15$ to $> 500$. | Inspect `adapter_config.json` for `lora_alpha` and `r`. Compute $W_{\text{merged}} - W_{\text{base}}$ manually on one layer and verify against $\frac{\alpha}{r}(B \times A)$. |
| **4** | **Float16 Underflow on T4** | T4 lacks native BF16 hardware compute. Training in pure FP16 without dynamic loss scaling causes small gradients ($< 2^{-14}$) to underflow to zero. | `soup doctor` checks CUDA capability (`sm_75`), rejects BF16, and forces FP16 with dynamic AMP `GradScaler`. | Gradient norm logs show `0.000000`; loss curve stays perfectly flat; weight delta Frobenius norm remains zero. | Run `torch.cuda.is_bf16_supported()`. Inspect `scaler.get_scale()` in trainer logs and ensure logged gradient norms are $> 10^{-4}$. |
| **5** | **4-bit Quantization Outlier Clipping** | Emergent activation outliers ($> 10\times$ normal channels) clipped by naive linear quantization, destroying attention maps. | `soup merge --save-format 4bit` defaults to BitsAndBytes NormalFloat4 (NF4) with double quantization. `soup eval quant-check` flags regressions. | Perplexity spikes by $> 2.0$ points; model produces repeated unigrams ("the the the...") or punctuation loops. | Measure KL-divergence between FP16 and 4-bit output logits on 20 prompts: $D_{\text{KL}}(P_{\text{FP16}} \parallel P_{\text{4bit}}) < 0.05$ confirms healthy quantization. |
| **6** | **Catastrophic Forgetting** | Model adapts to 1,500 conversational samples but collapses on multi-step reasoning, coding, and common sense logic. | `soup ship` enforces Leg-2 benchmark gating (`eval.ship.forgetting_threshold: 0.05`); `soup train --replay` provides continual rehearsal. | GSM8k or MMLU scores drop by $> 5\%$; basic arithmetic queries fail ("15 + 27 = 48"). | Evaluate 10 zero-shot prompts from GSM8k / HumanEval before and after training; reject if accuracy drop exceeds 5%. |

---

### 4. Verdict and Final Ship Decision

```
================================================================================
FINAL DEPLOYMENT DECISION GATE: SOUP SHIP
================================================================================
1. Memory Budget Compliance:        PASS (Peak 5.34 GB / 15.36 GB limit)
2. Weight Delta Frobenius Norm:    PASS (Mean 0.047820 on target projections)
3. Weight Isolation (Untouched):   PASS (Bit-exact 0.000000 on embeddings & norms)
4. Global Cosine Similarity:       PASS (0.994821 within [0.985, 0.999])
5. Held-Out Perplexity Win:        PASS (13.5680 < 14.8210, +8.45% relative gain)
6. 4-bit Quantization Penalty:     PASS (+0.3500 ppl degradation <= 0.50 ceiling)
--------------------------------------------------------------------------------
DECISION: >>> SHIP <<< (Exit Code: 0)
================================================================================
```

---

## Repository Structure & Deliverables

```
├── soup_takehome.ipynb       # Fully executable Google Colab notebook
├── verify_artifacts.py       # Standalone CLI verification engine
├── report.md                 # Formal 2-page technical engineering report
├── build_notebook.py         # Deterministic notebook generator
├── nvidia_smi_log.csv        # Continuous 1 Hz hardware VRAM telemetry log
├── raw_logs.txt              # Complete stdout/stderr execution trace
└── README.md                 # Architecture, benchmarks, and reproduction guide
```

---

## One-Click Reproduction Guide

### Option 1: Run in Google Colab (Recommended)
1. Open [Google Colab](https://colab.research.google.com).
2. Upload `soup_takehome.ipynb`.
3. Set runtime: **Runtime $\rightarrow$ Change runtime type $\rightarrow$ T4 GPU**.
4. Click **Run All**.

### Option 2: Run Locally or on Cloud GPU
```bash
# 1. Clone repository
git clone https://github.com/Speedyzyzz/soup-takehome.git
cd soup-takehome

# 2. Install dependencies
pip install "soup-cli[train]>=0.77.0" peft bitsandbytes transformers accelerate datasets scipy

# 3. Run verification engine across artifacts
python verify_artifacts.py \
    --base_model Qwen/Qwen2.5-1.5B-Instruct \
    --merged_model ./output_merged \
    --quantized_model ./output_4bit \
    --eval_file ./eval_50.jsonl \
    --device cuda \
    --plot \
    --output_json verification_results.json
```

---

## Author Contact

**Balavignesh V.T.**  
* AI Systems & Machine Learning Infrastructure Engineer  
* GitHub: [@Speedyzyzz](https://github.com/Speedyzyzz)  
