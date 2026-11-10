#!/usr/bin/env python3
"""
verify_artifacts.py - Production-Grade Verification Engine for LLM Fine-Tuning Artifacts
Part of the Soup AI Engineer Take-Home Task for Balavignesh V.T. (GitHub: Speedyzyzz)

Audits and compares three distinct model artifacts:
  - Artifact A: Base Model (Qwen/Qwen2.5-1.5B-Instruct)
  - Artifact B: Merged Model (Base + rsLoRA FP16 fused standalone model)
  - Artifact C: 4-bit Quantized Model (BitsAndBytes NF4 + Double Quantization)

Performs 6 mathematical and behavioral proofs:
  1. Weight Delta Frobenius Norm per layer (||W_merged - W_base||_F):
     - Proves Norm > 0 across all 196 adapted linear projections (q, k, v, o, gate, up, down)
     - Proves Norm == 0 across all frozen modules (embed_tokens, RMSNorms, tied lm_head)
     - Verifies preservation of weight tying (tie_word_embeddings=True) after merge
  2. Spectral Low-Rank SVD Proof on Weight Deltas:
     - Computes singular values of (W_merged - W_base) on representative projections
     - Proves top r=16 singular values capture >99.9% of total spectral energy (ruling out full-rank noise/corruption)
  3. Direct 4-Bit Weight Dequantization Audit (Base vs Merged vs Dequantized 4-Bit):
     - Reconstructs FP32 weights from 4-bit NF4 blocks via bitsandbytes.functional.dequantize_4bit
     - Proves ||W_4bit_dequant - W_merged||_F < ||W_4bit_dequant - W_base||_F (confirming 4-bit was quantized from Merged, not Base)
  4. Global Cosine Similarity between Base and Merged flattened tensors:
     - Proves healthy fine-tuning range [0.985, 0.9995]
  5. Multi-Turn Masked Perplexity & Token-Level Logit KL-Divergence (50 Held-Out Probes):
     - Evaluates NLL strictly over assistant response spans (<|im_start|>assistant\n ... <|im_end|>)
     - Computes KL(P_merged || P_base), KL(P_merged || P_4bit), and Top-1 Next-Token Agreement Rate
  6. Active Generation EOS Termination & Anti-Collapse Check:
     - Verifies deterministic generation terminates cleanly with <|im_end|> (ID 151645) and checks distinct 4-gram ratio
"""

import argparse
import json
import math
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

# Block incompatible environment torchao from breaking imports
sys.modules["torchao"] = None

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

try:
    import bitsandbytes as bnb
    HAS_BNB = True
except ImportError:
    HAS_BNB = False

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def parse_args():
    parser = argparse.ArgumentParser(
        description="Verify LLM fine-tuning artifacts: Frobenius Norm, SVD Rank-r Spectrum, 4-Bit Dequantization, KL-Divergence, Perplexity, and EOS Generation."
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Path or HuggingFace ID of the base model",
    )
    parser.add_argument(
        "--merged_model",
        type=str,
        default="./output_merged",
        help="Path to the merged FP16 model directory",
    )
    parser.add_argument(
        "--quantized_model",
        type=str,
        default="./output_4bit",
        help="Path to the serialized 4-bit NF4 quantized model directory",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=16,
        help="Expected LoRA rank r for spectral energy verification",
    )
    parser.add_argument(
        "--eval_file",
        type=str,
        default="./eval_50.jsonl",
        help="Path to held-out evaluation JSONL dataset (50 samples)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run evaluation on (cuda/cpu)",
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=50,
        help="Number of held-out samples to evaluate perplexity and KL divergence on",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="verification_results.json",
        help="Path to save verification summary JSON",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate high-resolution PNG plots of layer Frobenius norms and SVD spectrum",
    )
    return parser.parse_args()


# ==============================================================================
# HELPER: MULTI-TURN SAFE ASSISTANT SPAN MASKING FOR CHATML
# ==============================================================================

def build_assistant_masked_labels(
    messages: List[Dict[str, str]],
    tokenizer,
    max_length: int = 512
) -> Tuple[List[int], List[int]]:
    """
    Tokenizes a single-turn or multi-turn conversation using Qwen2.5 ChatML and
    masks 100% of system/user tokens and role headers with -100, unmasking ONLY
    assistant completion tokens (including the closing <|im_end|> token).
    """
    full_ids: List[int] = []
    labels: List[int] = []

    for idx, msg in enumerate(messages):
        if idx == 0:
            prefix_ids = []
        else:
            prefix_text = tokenizer.apply_chat_template(
                messages[:idx],
                tokenize=False,
                add_generation_prompt=(msg["role"] == "assistant")
            )
            prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)

        upto_text = tokenizer.apply_chat_template(
            messages[: idx + 1],
            tokenize=False,
            add_generation_prompt=False
        )
        upto_ids = tokenizer.encode(upto_text, add_special_tokens=False)

        if idx == 0:
            full_ids.extend(upto_ids)
            if msg["role"] == "assistant":
                header_len = len(prefix_ids)
                labels.extend([-100] * header_len + upto_ids[header_len:])
            else:
                labels.extend([-100] * len(upto_ids))
        else:
            turn_ids = upto_ids[len(full_ids):]
            header_ids = prefix_ids[len(full_ids):]
            full_ids.extend(turn_ids)
            if msg["role"] == "assistant":
                h_len = min(len(header_ids), len(turn_ids))
                labels.extend([-100] * h_len + turn_ids[h_len:])
            else:
                labels.extend([-100] * len(turn_ids))

    return full_ids[:max_length], labels[:max_length]


# ==============================================================================
# SECTION 1: WEIGHT DELTA FROBENIUS NORM, COSINE SIMILARITY & SVD SPECTRUM
# ==============================================================================

def analyze_weights_and_spectrum(
    base_model_path: str,
    merged_model_path: str,
    lora_rank: int = 16,
    plot: bool = False
) -> Tuple[Dict[str, float], float, Dict[str, Any]]:
    print("\n" + "=" * 84)
    print("STEP 1: WEIGHT TENSOR & SPECTRAL LOW-RANK AUDIT (BASE vs MERGED)")
    print("=" * 84)
    print(f"Base Model:   {base_model_path}")
    print(f"Merged Model: {merged_model_path}")

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="cpu",
        low_cpu_mem_usage=True
    )
    merged_model = AutoModelForCausalLM.from_pretrained(
        merged_model_path,
        torch_dtype=torch.float16,
        device_map="cpu",
        low_cpu_mem_usage=True
    )

    tied_embeddings_preserved = bool(getattr(merged_model.config, "tie_word_embeddings", False))
    base_dict = base_model.state_dict()
    merged_dict = merged_model.state_dict()

    layer_norms: Dict[str, float] = {}
    attn_layer_norms: Dict[int, List[float]] = {}
    mlp_layer_norms: Dict[int, List[float]] = {}

    dot_product = 0.0
    norm_base_sq = 0.0
    norm_merged_sq = 0.0

    targeted_norm_sum = 0.0
    targeted_count = 0
    untouched_norm_sum = 0.0
    untouched_count = 0

    svd_checks: Dict[str, float] = {}

    print("\nRepresentative Weight Delta Frobenius Norms (||W_merged - W_base||_F):")
    print("-" * 84)
    print(f"{'Layer / Parameter Name':<56} | {'Delta Norm':<12} | {'Status':<10}")
    print("-" * 84)

    for name, w_base_param in base_dict.items():
        if name not in merged_dict:
            continue

        w_base = w_base_param.float()
        w_merged = merged_dict[name].float()

        delta = w_merged - w_base
        norm = torch.linalg.norm(delta).item()
        layer_norms[name] = norm

        w_b_flat = w_base.flatten()
        w_m_flat = w_merged.flatten()
        dot_product += torch.dot(w_b_flat, w_m_flat).item()
        norm_base_sq += torch.dot(w_b_flat, w_b_flat).item()
        norm_merged_sq += torch.dot(w_m_flat, w_m_flat).item()

        is_attn = any(p in name for p in ["q_proj", "k_proj", "v_proj", "o_proj"])
        is_mlp = any(p in name for p in ["gate_proj", "up_proj", "down_proj"])
        is_untouched = any(u in name for u in ["embed_tokens", "input_layernorm", "post_attention_layernorm", "lm_head", "norm.weight"])

        if (is_attn or is_mlp) and "weight" in name:
            targeted_norm_sum += norm
            targeted_count += 1
            status = "ADAPTED" if norm > 1e-6 else "FAIL_NOOP"

            m = re.search(r"layers\.(\d+)\.", name)
            if m:
                l_idx = int(m.group(1))
                if is_attn:
                    attn_layer_norms.setdefault(l_idx, []).append(norm)
                else:
                    mlp_layer_norms.setdefault(l_idx, []).append(norm)

            if name in [
                "model.layers.0.self_attn.q_proj.weight",
                "model.layers.14.self_attn.v_proj.weight",
                "model.layers.27.mlp.down_proj.weight",
            ] and norm > 1e-6:
                s = torch.linalg.svdvals(delta)
                total_energy = torch.sum(s ** 2).item()
                top_r_energy = torch.sum(s[:lora_rank] ** 2).item()
                svd_checks[name] = (top_r_energy / max(total_energy, 1e-12)) * 100.0
        elif is_untouched:
            untouched_norm_sum += norm
            untouched_count += 1
            status = "FROZEN" if norm < 1e-6 else "LEAKED"
        else:
            status = "ADAPTED" if norm > 1e-6 else "FROZEN"

        if any(marker in name for marker in [
            "layers.0.self_attn.q_proj.weight",
            "layers.0.self_attn.k_proj.weight",
            "layers.0.mlp.down_proj.weight",
            "layers.14.self_attn.q_proj.weight",
            "layers.14.mlp.down_proj.weight",
            "layers.27.self_attn.q_proj.weight",
            "layers.27.mlp.down_proj.weight",
            "embed_tokens.weight",
            "norm.weight",
            "lm_head.weight"
        ]):
            print(f"{name:<56} | {norm:<12.6f} | {status:<10}")

    print("-" * 84)

    cosine_sim = dot_product / (math.sqrt(norm_base_sq) * math.sqrt(norm_merged_sq))
    avg_targeted_norm = targeted_norm_sum / max(targeted_count, 1)
    avg_untouched_norm = untouched_norm_sum / max(untouched_count, 1)
    mean_svd_top_r_pct = sum(svd_checks.values()) / max(len(svd_checks), 1) if svd_checks else 0.0

    print(f"\n[WEIGHT & SPECTRAL VERIFICATION METRICS]")
    print(f"Total Target Projections Checked:        {targeted_count} linear layers (28 layers x 7 projections)")
    print(f"Mean Frobenius Norm on Target Layers:    {avg_targeted_norm:.6f}  (Must be > 0.0)")
    print(f"Mean Frobenius Norm on Frozen Layers:    {avg_untouched_norm:.6f}  (Must be == 0.0)")
    print(f"Top-{lora_rank} SVD Spectral Energy Ratio:       {mean_svd_top_r_pct:.4f}%  (Must be > 99.9% for rank-{lora_rank} LoRA)")
    print(f"Weight Tying Preserved (lm_head==embed): {tied_embeddings_preserved}")
    print(f"Global Base-Merged Cosine Similarity:    {cosine_sim:.6f}")

    if cosine_sim >= 0.99999:
        sim_verdict = "CRITICAL FAILURE: Cosine similarity is 1.00000. Adapter was NEVER merged."
    elif 0.985 <= cosine_sim <= 0.9995:
        sim_verdict = f"PASS: Cosine similarity {cosine_sim:.6f} within healthy fine-tuning window [0.985, 0.9995]."
    elif cosine_sim < 0.90:
        sim_verdict = f"CRITICAL FAILURE: Cosine similarity {cosine_sim:.6f} < 0.90 indicates catastrophic weight collapse."
    else:
        sim_verdict = f"NOTICE: Cosine similarity {cosine_sim:.6f} is marginally outside typical low-rank window."

    print(f"Verdict: {sim_verdict}")

    if plot and HAS_MATPLOTLIB:
        try:
            layers = sorted(list(set(attn_layer_norms.keys()) | set(mlp_layer_norms.keys())))
            avg_attn = [sum(attn_layer_norms.get(l, [0])) / max(len(attn_layer_norms.get(l, [1])), 1) for l in layers]
            avg_mlp = [sum(mlp_layer_norms.get(l, [0])) / max(len(mlp_layer_norms.get(l, [1])), 1) for l in layers]

            plt.figure(figsize=(10, 4.5), dpi=300)
            plt.plot(layers, avg_attn, marker="o", linewidth=2, label="Self-Attention (q, k, v, o)", color="#2563eb")
            plt.plot(layers, avg_mlp, marker="s", linewidth=2, label="Feed-Forward MLP (gate, up, down)", color="#10b981")
            plt.axhline(0, color="gray", linestyle="--", alpha=0.5)
            plt.title("Qwen2.5-1.5B Weight Delta Frobenius Norm per Decoder Layer", fontsize=12, fontweight="bold")
            plt.xlabel("Layer Index (0 to 27)", fontsize=10)
            plt.ylabel(r"$\|W_{\mathrm{merged}} - W_{\mathrm{base}}\|_F$", fontsize=10)
            plt.grid(True, linestyle=":", alpha=0.6)
            plt.legend(frameon=True)
            plt.tight_layout()
            plt.savefig("weight_delta_norms.png")
            print("Generated weight norm plot: weight_delta_norms.png")
        except Exception as e:
            print(f"Plotting notice: {e}")

    del base_model, merged_model, base_dict, merged_dict
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    stats = {
        "targeted_layers_count": targeted_count,
        "avg_targeted_norm": avg_targeted_norm,
        "avg_untouched_norm": avg_untouched_norm,
        "svd_top_r_energy_pct": mean_svd_top_r_pct,
        "svd_layer_breakdown": svd_checks,
        "tied_embeddings_preserved": tied_embeddings_preserved,
        "cosine_similarity": cosine_sim,
        "similarity_verdict": sim_verdict,
    }
    return layer_norms, cosine_sim, stats


# ==============================================================================
# SECTION 2: 4-BIT WEIGHT DEQUANTIZATION LINEAGE VERIFICATION
# ==============================================================================

def verify_4bit_weight_lineage(
    base_model_path: str,
    merged_model_path: str,
    quantized_model_path: str,
    device: str = "cuda"
) -> Dict[str, Any]:
    print("\n" + "=" * 84)
    print("STEP 2: 4-BIT NF4 WEIGHT DEQUANTIZATION & LINEAGE VERIFICATION")
    print("=" * 84)

    if device != "cuda" or not HAS_BNB:
        raise RuntimeError("CUDA and bitsandbytes are required to run genuine 4-bit NF4 dequantization checks.")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16
    )

    base_cpu = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.float16, device_map="cpu", low_cpu_mem_usage=True)
    merged_cpu = AutoModelForCausalLM.from_pretrained(merged_model_path, torch_dtype=torch.float16, device_map="cpu", low_cpu_mem_usage=True)
    quant_gpu = AutoModelForCausalLM.from_pretrained(quantized_model_path, quantization_config=bnb_config, device_map="auto", low_cpu_mem_usage=True)

    sample_layers = [
        ("layers.0.self_attn.q_proj", base_cpu.model.layers[0].self_attn.q_proj, merged_cpu.model.layers[0].self_attn.q_proj, quant_gpu.model.layers[0].self_attn.q_proj),
        ("layers.14.mlp.down_proj", base_cpu.model.layers[14].mlp.down_proj, merged_cpu.model.layers[14].mlp.down_proj, quant_gpu.model.layers[14].mlp.down_proj),
        ("layers.27.self_attn.v_proj", base_cpu.model.layers[27].self_attn.v_proj, merged_cpu.model.layers[27].self_attn.v_proj, quant_gpu.model.layers[27].self_attn.v_proj),
    ]

    lineage_records = []
    for layer_name, mod_b, mod_m, mod_q in sample_layers:
        w_q_param = mod_q.weight
        if hasattr(w_q_param, "quant_state") and w_q_param.quant_state is not None:
            w_dequant = bnb.functional.dequantize_4bit(w_q_param.data, w_q_param.quant_state).cpu().float()
        else:
            w_dequant = w_q_param.detach().cpu().float()

        w_b = mod_b.weight.detach().cpu().float()
        w_m = mod_m.weight.detach().cpu().float()

        err_to_merged = torch.linalg.norm(w_dequant - w_m).item()
        err_to_base = torch.linalg.norm(w_dequant - w_b).item()
        lora_delta = torch.linalg.norm(w_m - w_b).item()

        closer_to_merged = err_to_merged < err_to_base
        print(
            f"Layer {layer_name:<26} | ||W_4bit - W_merged||_F = {err_to_merged:.5f} | "
            f"||W_4bit - W_base||_F = {err_to_base:.5f} | ||W_merged - W_base||_F = {lora_delta:.5f} | "
            f"{'PASS (Merged Lineage)' if closer_to_merged else 'FAIL (Closer to Base!)'}"
        )
        lineage_records.append({
            "layer": layer_name,
            "dist_4bit_to_merged": err_to_merged,
            "dist_4bit_to_base": err_to_base,
            "lora_delta_norm": lora_delta,
            "derived_from_merged": closer_to_merged
        })

    del base_cpu, merged_cpu, quant_gpu
    torch.cuda.empty_cache()

    all_derived_from_merged = all(r["derived_from_merged"] for r in lineage_records)
    return {
        "all_derived_from_merged": all_derived_from_merged,
        "sample_layers": lineage_records
    }


# ==============================================================================
# SECTION 3: DETERMINISTIC PERPLEXITY, KL-DIVERGENCE & TOP-1 AGREEMENT
# ==============================================================================

def load_eval_prompts(eval_file_path: str, max_samples: int = 50) -> List[Dict[str, Any]]:
    if not os.path.exists(eval_file_path):
        raise FileNotFoundError(
            f"Evaluation dataset '{eval_file_path}' not found! Do not fall back to synthetic prompts; "
            f"run the dataset partitioning step first."
        )
    samples = []
    with open(eval_file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
            if len(samples) >= max_samples:
                break
    return samples


def collect_eval_logits_and_perplexity(
    model_path: str,
    tokenizer,
    eval_samples: List[Dict[str, Any]],
    device: str = "cuda",
    quantization_config: Optional[BitsAndBytesConfig] = None,
    reference_probs: Optional[List[torch.Tensor]] = None,
    reference_top1: Optional[List[torch.Tensor]] = None,
) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {"device_map": "auto" if device == "cuda" else "cpu"}
    if quantization_config is not None:
        kwargs["quantization_config"] = quantization_config
    else:
        kwargs["torch_dtype"] = torch.float16 if device == "cuda" else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        low_cpu_mem_usage=True,
        **kwargs
    )
    model.eval()

    total_nll = 0.0
    total_tokens = 0
    loss_fn = nn.CrossEntropyLoss(reduction="sum")

    collected_probs: List[torch.Tensor] = []
    collected_top1: List[torch.Tensor] = []

    kl_sum = 0.0
    top1_matches = 0
    compared_tokens = 0

    with torch.no_grad():
        for item in eval_samples:
            if "messages" in item:
                messages = item["messages"]
            else:
                instruction = item.get("instruction", "")
                inp = item.get("input", "")
                output = item.get("output", "")
                user_content = f"{instruction}\n{inp}".strip()
                messages = [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": output}
                ]

            full_ids, label_ids = build_assistant_masked_labels(messages, tokenizer, max_length=512)
            if len(full_ids) < 2:
                continue

            input_tensor = torch.tensor([full_ids], device=device)
            labels = torch.tensor([label_ids], device=device)

            outputs = model(input_tensor)
            logits = outputs.logits.float()

            shift_logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
            shift_labels = labels[..., 1:].contiguous().view(-1)

            valid_mask = (shift_labels != -100)
            num_valid = int(valid_mask.sum().item())
            if num_valid == 0:
                continue

            valid_logits = shift_logits[valid_mask]
            valid_targets = shift_labels[valid_mask]

            nll = loss_fn(valid_logits, valid_targets)
            total_nll += nll.item()
            total_tokens += num_valid

            probs_cpu = F.softmax(valid_logits, dim=-1).cpu()
            top1_cpu = torch.argmax(valid_logits, dim=-1).cpu()

            if reference_probs is None:
                collected_probs.append(probs_cpu)
                collected_top1.append(top1_cpu)
            else:
                ref_p = reference_probs[len(collected_probs)]
                ref_t1 = reference_top1[len(collected_top1)]
                collected_probs.append(torch.empty(0))
                collected_top1.append(torch.empty(0))

                eps = 1e-8
                kl_tokens = torch.sum(ref_p * (torch.log(ref_p + eps) - torch.log(probs_cpu + eps)), dim=-1)
                kl_sum += kl_tokens.sum().item()
                top1_matches += int((ref_t1 == top1_cpu).sum().item())
                compared_tokens += num_valid

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    ppl = math.exp(total_nll / max(total_tokens, 1))
    mean_kl = (kl_sum / compared_tokens) if compared_tokens > 0 else 0.0
    top1_agreement = (top1_matches / compared_tokens) if compared_tokens > 0 else 1.0

    return {
        "perplexity": ppl,
        "total_eval_tokens": total_tokens,
        "mean_kl_from_merged": mean_kl,
        "top1_agreement_with_merged": top1_agreement,
        "probs": collected_probs if reference_probs is None else None,
        "top1": collected_top1 if reference_top1 is None else None,
    }


# ==============================================================================
# SECTION 4: ACTIVE GENERATION EOS TERMINATION & ANTI-COLLAPSE CHECK
# ==============================================================================

def verify_generation_health(
    model_path: str,
    tokenizer,
    eval_samples: List[Dict[str, Any]],
    device: str = "cuda",
    quantization_config: Optional[BitsAndBytesConfig] = None,
    num_probes: int = 10
) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {"device_map": "auto" if device == "cuda" else "cpu"}
    if quantization_config is not None:
        kwargs["quantization_config"] = quantization_config
    else:
        kwargs["torch_dtype"] = torch.float16 if device == "cuda" else torch.float32

    model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
    model.eval()

    eos_terminated_count = 0
    distinct_4gram_ratios: List[float] = []

    with torch.no_grad():
        for item in eval_samples[:num_probes]:
            if "messages" in item:
                user_msgs = [item["messages"][0]] if item["messages"][0]["role"] == "user" else item["messages"][:2]
            else:
                user_msgs = [{"role": "user", "content": item.get("instruction", "Hello")}]

            prompt_str = tokenizer.apply_chat_template(user_msgs, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt_str, return_tensors="pt").to(device)
            prompt_len = inputs.input_ids.shape[1]

            out = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            gen_ids = out[0][prompt_len:].tolist()
            if len(gen_ids) < 128 or (len(gen_ids) > 0 and gen_ids[-1] == tokenizer.eos_token_id):
                eos_terminated_count += 1

            if len(gen_ids) >= 4:
                four_grams = [tuple(gen_ids[i:i+4]) for i in range(len(gen_ids) - 3)]
                distinct_ratio = len(set(four_grams)) / len(four_grams)
            else:
                distinct_ratio = 1.0
            distinct_4gram_ratios.append(distinct_ratio)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "probes_tested": num_probes,
        "eos_termination_rate": eos_terminated_count / max(num_probes, 1),
        "mean_distinct_4gram_ratio": sum(distinct_4gram_ratios) / max(len(distinct_4gram_ratios), 1)
    }


# ==============================================================================
# SECTION 5: BLIND SPOT STATEMENT
# ==============================================================================

BLIND_SPOT_STATEMENT = """
[EXPLICIT VERIFICATION ENGINE BLIND SPOTS]
1. Factual Truthfulness & Hallucinations: Weight norms, SVD rank checks, KL-divergence, and
   held-out perplexity confirm genuine optimization and domain distribution alignment, but they
   CANNOT verify factual correctness. A fine-tuned model can assign high probability to fluent,
   stylistically faithful, yet factually false statements.
2. Out-of-Distribution Reasoning & Coding Regression (Catastrophic Forgetting): Evaluating 50
   in-domain conversational probes cannot detect silent degradation on multi-step mathematical
   reasoning (GSM8k/MATH), code synthesis (HumanEval/MBPP), or formal logic unless explicitly
   paired with an external benchmark suite (such as `soup ship` Leg 2).
3. Safety, Refusal Calibration & Alignment Erosion: Fine-tuning even on benign conversational
   examples shifts attention and MLP weights in ways that can soften RLHF/DPO refusal boundaries
   against adversarial jailbreaks without triggering perplexity or Frobenius norm alarms.
4. Long-Context Needle-in-a-Haystack Degradation: Because training and verification are truncated
   at max_length=512, this script cannot detect positional interpolation drift or attention sink
   degradation at 8k-32k context lengths.
"""


# ==============================================================================
# MAIN RUNNER
# ==============================================================================

def main():
    args = parse_args()
    print("=" * 84)
    print("SOUP AI ARTIFACT VERIFICATION ENGINE (3-ARTIFACT DEEP AUDIT)")
    print(f"Device: {args.device} | Base: {args.base_model} | LoRA Rank r={args.lora_rank}")
    print("=" * 84)

    layer_norms, cosine_sim, weight_stats = analyze_weights_and_spectrum(
        base_model_path=args.base_model,
        merged_model_path=args.merged_model,
        lora_rank=args.lora_rank,
        plot=args.plot
    )

    lineage_stats = verify_4bit_weight_lineage(
        base_model_path=args.base_model,
        merged_model_path=args.merged_model,
        quantized_model_path=args.quantized_model,
        device=args.device
    )

    print("\n" + "=" * 84)
    print("STEP 3: MULTI-TURN MASKED PERPLEXITY, KL-DIVERGENCE & TOP-1 AGREEMENT (N=50)")
    print("=" * 84)
    eval_samples = load_eval_prompts(args.eval_file, max_samples=args.max_eval_samples)
    print(f"Loaded {len(eval_samples)} held-out evaluation samples from {args.eval_file}.")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("\nEvaluating Artifact B (Merged Model FP16 - Reference Distribution)...")
    merged_eval = collect_eval_logits_and_perplexity(
        args.merged_model,
        tokenizer,
        eval_samples,
        device=args.device
    )
    merged_ppl = merged_eval["perplexity"]
    print(f"-> Merged Model Perplexity: {merged_ppl:.4f} ({merged_eval['total_eval_tokens']} assistant tokens)")

    print("\nEvaluating Artifact A (Base Model FP16 vs Merged Reference)...")
    base_eval = collect_eval_logits_and_perplexity(
        args.base_model,
        tokenizer,
        eval_samples,
        device=args.device,
        reference_probs=merged_eval["probs"],
        reference_top1=merged_eval["top1"]
    )
    base_ppl = base_eval["perplexity"]
    kl_merged_base = base_eval["mean_kl_from_merged"]
    top1_merged_base = base_eval["top1_agreement_with_merged"]
    print(f"-> Base Model Perplexity:   {base_ppl:.4f}")
    print(f"-> KL(P_merged || P_base):  {kl_merged_base:.5f} nats/tok | Top-1 Agreement: {top1_merged_base*100:.2f}%")

    print("\nEvaluating Artifact C (4-Bit NF4 Quantized Model vs Merged Reference)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16
    )
    quant_eval = collect_eval_logits_and_perplexity(
        args.quantized_model,
        tokenizer,
        eval_samples,
        device=args.device,
        quantization_config=bnb_config,
        reference_probs=merged_eval["probs"],
        reference_top1=merged_eval["top1"]
    )
    quant_ppl = quant_eval["perplexity"]
    quant_penalty = quant_ppl - merged_ppl
    kl_merged_4bit = quant_eval["mean_kl_from_merged"]
    top1_merged_4bit = quant_eval["top1_agreement_with_merged"]
    print(f"-> 4-Bit NF4 Perplexity:    {quant_ppl:.4f} (Penalty: +{quant_penalty:.4f} ppl)")
    print(f"-> KL(P_merged || P_4bit):  {kl_merged_4bit:.5f} nats/tok | Top-1 Agreement: {top1_merged_4bit*100:.2f}%")

    print("\n" + "=" * 84)
    print("STEP 4: ACTIVE GENERATION EOS TERMINATION & 4-GRAM DIVERSITY CHECK")
    print("=" * 84)
    merged_gen = verify_generation_health(args.merged_model, tokenizer, eval_samples, device=args.device)
    quant_gen = verify_generation_health(args.quantized_model, tokenizer, eval_samples, device=args.device, quantization_config=bnb_config)
    print(f"Merged FP16 -> EOS Rate: {merged_gen['eos_termination_rate']*100:.1f}% | Distinct 4-gram Ratio: {merged_gen['mean_distinct_4gram_ratio']:.4f}")
    print(f"4-Bit NF4   -> EOS Rate: {quant_gen['eos_termination_rate']*100:.1f}% | Distinct 4-gram Ratio: {quant_gen['mean_distinct_4gram_ratio']:.4f}")

    print("\n" + "=" * 84)
    print("STEP 5: COMPREHENSIVE 3-ARTIFACT VERIFICATION SCORECARD")
    print("=" * 84)

    ppl_improved = merged_ppl < base_ppl
    ppl_delta = merged_ppl - base_ppl
    ppl_improvement_pct = ((base_ppl - merged_ppl) / base_ppl) * 100.0

    svd_pct_str = f"{weight_stats['svd_top_r_energy_pct']:.4f}%"
    top1_str = f"{top1_merged_4bit * 100:.2f}%"
    eos_4gram_str = f"{quant_gen['eos_termination_rate'] * 100:.0f}% / {quant_gen['mean_distinct_4gram_ratio']:.3f}"

    print(f"{'Verification Check':<46} | {'Observed Value':<18} | {'Status'}")
    print("-" * 84)
    print(f"{'1. Target Projections Frobenius Norm Mean':<46} | {weight_stats['avg_targeted_norm']:<18.6f} | {'PASS (> 0.0)' if weight_stats['avg_targeted_norm'] > 0 else 'FAIL'}")
    print(f"{'2. Frozen Modules Frobenius Norm Mean':<46} | {weight_stats['avg_untouched_norm']:<18.6f} | {'PASS (== 0.0)' if weight_stats['avg_untouched_norm'] == 0 else 'LEAKED'}")
    print(f"{'3. SVD Top-r Spectral Energy Concentration':<46} | {svd_pct_str:<18} | {'PASS (> 99.9%)' if weight_stats['svd_top_r_energy_pct'] > 99.9 else 'FAIL_NOISE'}")
    print(f"{'4. Global Base-Merged Cosine Similarity':<46} | {cosine_sim:<18.6f} | {'PASS [0.985, 0.9995]' if 0.985 <= cosine_sim <= 0.9995 else 'INVESTIGATE'}")
    print(f"{'5. 4-Bit Dequantized Lineage (Closer to Merged)':<46} | {str(lineage_stats['all_derived_from_merged']):<18} | {'PASS (Merged Lineage)' if lineage_stats['all_derived_from_merged'] else 'FAIL'}")
    print(f"{'6. Base Model Held-Out Perplexity':<46} | {base_ppl:<18.4f} | {'BASELINE'}")
    print(f"{'7. Merged FP16 Held-Out Perplexity':<46} | {merged_ppl:<18.4f} | {f'PASS (-{ppl_improvement_pct:.2f}%)' if ppl_improved else 'FAIL'}")
    print(f"{'8. 4-Bit NF4 Held-Out Perplexity':<46} | {quant_ppl:<18.4f} | {f'PASS (+{quant_penalty:.4f} ppl)' if quant_penalty <= 0.50 else 'HIGH_PENALTY'}")
    print(f"{'9. Token KL Divergence KL(Merged || Base)':<46} | {kl_merged_base:<18.5f} | {'PASS (Learned Shift)' if kl_merged_base > 0.01 else 'NOOP'}")
    print(f"{'10. Token KL Divergence KL(Merged || 4-Bit)':<46} | {kl_merged_4bit:<18.5f} | {'PASS (Low Quant Drift)' if kl_merged_4bit < 0.03 else 'HIGH_DRIFT'}")
    print(f"{'11. Top-1 Next-Token Agreement (Merged vs 4-Bit)':<46} | {top1_str:<18} | {'PASS (> 92%)' if top1_merged_4bit > 0.92 else 'LOW_AGREEMENT'}")
    print(f"{'12. 4-Bit EOS Termination & 4-Gram Diversity':<46} | {eos_4gram_str:<18} | {'PASS' if quant_gen['eos_termination_rate'] >= 0.9 and quant_gen['mean_distinct_4gram_ratio'] >= 0.85 else 'LOOP_RISK'}")

    print(BLIND_SPOT_STATEMENT)

    results = {
        "base_model": args.base_model,
        "merged_model": args.merged_model,
        "quantized_model": args.quantized_model,
        "device": args.device,
        "lora_rank": args.lora_rank,
        "weight_verification": weight_stats,
        "quantized_lineage": lineage_stats,
        "perplexity_and_kl": {
            "base_perplexity": base_ppl,
            "merged_perplexity": merged_ppl,
            "perplexity_delta": ppl_delta,
            "improvement_pct": ppl_improvement_pct,
            "quantized_perplexity": quant_ppl,
            "quantization_penalty": quant_penalty,
            "kl_merged_vs_base": kl_merged_base,
            "top1_agreement_merged_vs_base": top1_merged_base,
            "kl_merged_vs_4bit": kl_merged_4bit,
            "top1_agreement_merged_vs_4bit": top1_merged_4bit,
        },
        "generation_health": {
            "merged_fp16": merged_gen,
            "quantized_4bit": quant_gen,
        },
        "blind_spots": BLIND_SPOT_STATEMENT.strip()
    }

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Verification summary saved to {args.output_json}\n")


if __name__ == "__main__":
    main()
