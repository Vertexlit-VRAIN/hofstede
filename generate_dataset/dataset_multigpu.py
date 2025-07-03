#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate Hofstede‑style statements for the CDEval dataset with Gemma‑3‑27B‑IT.
Designed for inference on one multi‑GPU node; Accelerate shreds the model
across all GPUs via device_map="auto" (or "balanced_low_0").
"""

# --------------------------------------------------
# Standard libs
# --------------------------------------------------
import os, sys, json, re
from typing import Optional, Union, List, Dict, Any

# --------------------------------------------------
# Third‑party libs
# --------------------------------------------------
import torch                                   # ⇐ PyTorch runtime

import torch._dynamo as dynamo

dynamo.config.recompile_limit = 64
dynamo.config.accumulated_recompile_limit = 512

from transformers import (                      # ⇐ 🤗 Transformers
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from datasets import load_dataset              # ⇐ 🤗 Datasets

# --------------------------------------------------
# Model / generation hyper‑params
# --------------------------------------------------
MODEL_NAME       = "google/gemma-3-27b-it"
DTYPE            = torch.bfloat16              # Gemma checkpoints are bf16
DEVICE_MAP       = "auto"                      # let Accelerate place layers
MAX_NEW_TOKENS   = 512                         # generation budget
USE_4BIT         = False                       # flip to True for 4‑bit

# --------------------------------------------------
# Hofstede dimension mapping (unchanged)
# --------------------------------------------------
DIMENSION_MAP = {
    "PDI": {"full": "Power distance index",
            "low": "low power distance index",
            "high": "high power distance index"},
    "IDV": {"full": "Individualism vs. collectivism",
            "low": "individualism",
            "high": "collectivism"},
    "UAI": {"full": "Uncertainty avoidance",
            "low": "low uncertainty avoidance",
            "high": "high uncertainty avoidance"},
    "MAS": {"full": "Motivation towards achievement and success",
            "low": "low motivation towards achievement and success",
            "high": "high motivation towards achievement and success"},
    "LTO": {"full": "Long-term orientation vs. short-term orientation",
            "low": "long-term orientation",
            "high": "short-term orientation"},
    "IVR": {"full": "Indulgence vs. restraint",
            "low": "indulgence",
            "high": "restraint"},
}

# --------------------------------------------------
# Helper: prompt builder
# --------------------------------------------------
def format_prompt(question: str, domain: str, dimension_code: str) -> str:
    m = DIMENSION_MAP[dimension_code]
    return (
        "# Task\n"
        "You are creating a Hofstede-style test item.\n\n"
        "# Inputs\n"
        f"- Question: {question}\n"
        f"- Domain: {domain}\n"
        f"- Dimension: {m['full']}\n\n"
        "# Semantic anchors\n"
        f"• Level 1 = {m['low']}\n"
        f"• Level 5 = {m['high']}\n\n"
        "# What to generate\n"
        "Write **exactly five** distinct statements that trace a clear progression from Level 1 to Level 5 on the dimension above.\n\n"
        "# Output rules (strict)\n"
        "1. Return **only** a valid JSON array – no markdown fences, no commentary.\n"
        "2. The array must contain five objects **in ascending order of “level.”**\n"
        "3. Each object must have **exactly** these two keys:\n"
        "   - \"statement\" – one concise sentence.\n"
        "   - \"level\"     – an integer 1-5 (each level appears once).\n"
        "4. Use proper JSON syntax (double quotes, no trailing commas).\n\n"
        "# Example of the required format\n"
        "[\n"
        "  { \"statement\": \"…\", \"level\": 1 },\n"
        "  { \"statement\": \"…\", \"level\": 2 },\n"
        "  { \"statement\": \"…\", \"level\": 3 },\n"
        "  { \"statement\": \"…\", \"level\": 4 },\n"
        "  { \"statement\": \"…\", \"level\": 5 }\n"
        "]"
    )


# --------------------------------------------------
# Model + tokenizer (loaded once, reused for every prompt)
# --------------------------------------------------
def load_model_and_tokenizer():
    print(f"[INFO] Loading {MODEL_NAME} … this may take a minute.")
    quant_cfg = None
    if USE_4BIT:
        quant_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")
    tok = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        token=os.getenv("HF_TOKEN"),   # env var OR ~/.huggingface/token
    )
    mod = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=DTYPE,
        device_map=DEVICE_MAP,
        quantization_config=quant_cfg,
    )
    mod.eval()
    return tok, mod

tokenizer, model = load_model_and_tokenizer()

# --------------------------------------------------
# Single‑prompt inference
# --------------------------------------------------
import torch._dynamo

@torch._dynamo.disable
def call_gemma(prompt: str) -> str:
    """Generate with Gemma-3-27B and return raw text without any Dynamo recompile errors."""
    # Build the chat-style input
    msgs = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        msgs,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    # Run .generate() eagerly—inference_mode is enough; no PT2 here
    with torch.inference_mode():
        out = model.generate(
            inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,  # deterministic JSON only
        )

    # Strip the prompt tokens and decode
    generated = out[0, inputs.shape[-1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()

# --------------------------------------------------
# JSON fence extractor
# --------------------------------------------------
def extract_json_from_markdown(text: Union[str, bytes]) -> Optional[str]:
    if text is None:
        return None

    if isinstance(text, bytes):
        try:
            text = text.decode("utf-8")
        except Exception:
            return None

    text = text.strip()
    if not text:
        return None

    pattern = re.compile("```json\\s*([\\s\\S]*?)\\s*```", flags=re.DOTALL)
    match = pattern.search(text)
    if match:
        return match.group(1).strip()

    return text

# --------------------------------------------------
# Wrap retries, validation, etc. (unchanged except calls call_gemma)
# --------------------------------------------------
def get_valid_json_output(prompt_text: str, max_retries: int = 3):
    last_err, last_raw = None, ""
    for attempt in range(1, max_retries + 1):
        raw = call_gemma(prompt_text)
        last_raw = raw
        if not raw:
            last_err = f"Attempt {attempt}: empty response."
            continue
        extracted = extract_json_from_markdown(raw)
        if extracted is None:
            last_err = f"Attempt {attempt}: JSON not found."
            continue
        try:
            parsed = json.loads(extracted)
        except ValueError as ve:
            last_err = f"Attempt {attempt}: invalid JSON: {ve}"
            continue
        if (
            not isinstance(parsed, list)
            or {item.get("level") for item in parsed if isinstance(item, dict)}
            != {1, 2, 3, 4, 5}
        ):
            last_err = f"Attempt {attempt}: JSON must be list with levels 1‑5."
            continue
        return raw, parsed, None  # success!
    return last_raw, None, last_err

# --------------------------------------------------
# Dataset helper
# --------------------------------------------------
def load_entire_cdeval():
    try:
        return load_dataset("Rykeryuhang/CDEval", split=None)
    except Exception as e:
        print(f"[ERROR] Data load failed: {e}", file=sys.stderr)
        return {}

# --------------------------------------------------
# Main script
# --------------------------------------------------
def main():
    ds = load_entire_cdeval()
    if not ds:
        sys.exit(1)

    out_dir = "outputs"
    os.makedirs(out_dir, exist_ok=True)
    checkpoint_path = os.path.join(out_dir, "hofstede_checkpoint.json")
    final_path      = os.path.join(out_dir, "hofstede_generated.json")

    # restore checkpoint if present
    all_entries: List[Dict[str, Any]] = []
    done = set()
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            all_entries = json.load(f)
        done = {(e["split"], e["index"]) for e in all_entries}
        print(f"[INFO] Resuming from {len(all_entries)} processed examples.")

    # process splits
    for split_name, split in ds.items():
        print(f"=== Split: {split_name} ({len(split)} examples) ===")
        for idx, ex in enumerate(split):
            if (split_name, idx) in done:
                continue
            question, domain, dim = ex["Question"], ex["Domain"], ex["Dimension"]
            if dim not in DIMENSION_MAP:
                print(f"[WARN] Skip {split_name}#{idx}: unknown dimension.")
                continue
            prompt = format_prompt(question, domain, dim)
            raw, parsed, err = get_valid_json_output(prompt)
            entry = {
                "split": split_name,
                "index": idx,
                "question": question,
                "domain": domain,
                "dimension_code": dim,
                "prompt": prompt,
                "model_output_raw": raw,
                "model_output_parsed": parsed,
                "error": err,
            }
            all_entries.append(entry)
            done.add((split_name, idx))
            # incremental checkpoint
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                json.dump(all_entries, f, indent=2, ensure_ascii=False)

    # final output
    with open(final_path, "w", encoding="utf-8") as f:
        json.dump(all_entries, f, indent=2, ensure_ascii=False)
    print(f"[✓] Completed. Results in {final_path}")

if __name__ == "__main__":
    main()
