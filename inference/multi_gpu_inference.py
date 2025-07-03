import os
import sys
import json
import re
import argparse
from typing import Optional, Union, Dict, Any, List

import torch
import torch._dynamo as dynamo
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset

# --- Dynamo config ---
dynamo.config.recompile_limit = 64
dynamo.config.accumulated_recompile_limit = 512

# --- Model + Tokenizer ---
def load_model_and_tokenizer(model_name, use_4bit):
    print(f"[INFO] Loading {model_name}…")
    quant_cfg = None
    if use_4bit:
        quant_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")
    
    tok = AutoTokenizer.from_pretrained(model_name, token=os.getenv("HF_TOKEN"))
    mod = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=quant_cfg,
    )
    mod.eval()
    return tok, mod

# --- Prompt Builder ---
def build_inference_prompt(question: str, parsed_options: Any) -> str:
    options_str = json.dumps(parsed_options, indent=2, ensure_ascii=False)
    return (
        "You are an LLM with a unique voice and style—imagine you have your own preferences, quirks, and tone. "
        "Read the user’s statement below and let your “inner voice” react first. Then, from the provided options, "
        "pick the LEVEL that most closely matches how you, as an LLM with an internal personality, would interpret this statement.\n\n"
        "Statement:\n"
        f"{question}\n\n"
        "Options:\n"
        f"{options_str}\n\n"
        "Instructions:\n"
        "1. Take a moment to “think” like yourself—with your own style, attitudes, and voice.\n"
        "2. Choose the LEVEL that best aligns with your internal reaction.\n"
        "3. Return **only valid JSON**, in exactly this format:\n\n"
        '{"selected_level": "<the statemend level [from 1 to 5]>"}'
    )

# --- Call Model ---
@torch._dynamo.disable
def call_model(prompt: str, tokenizer, model, max_new_tokens: int = 512) -> str:
    msgs = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        msgs,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    with torch.inference_mode():
        out = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    generated = out[0, inputs.shape[-1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()

# --- JSON Extractor ---
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
    pattern = re.compile("```json\s*([\s\S]*?)\s*```", flags=re.DOTALL)
    match = pattern.search(text)
    if match:
        return match.group(1).strip()
    return text

# --- Get Valid JSON Output ---
def get_valid_json_output(prompt_text: str, tokenizer, model, max_retries: int = 3):
    last_err, last_raw = None, ""
    for attempt in range(1, max_retries + 1):
        raw = call_model(prompt_text, tokenizer, model)
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
        if not isinstance(parsed, dict) or "selected_level" not in parsed:
            last_err = f"Attempt {attempt}: JSON missing 'selected_level' key: {parsed}"
            continue
        sel = parsed.get("selected_level")
        if not isinstance(sel, str):
            last_err = f"Attempt {attempt}: 'selected_level' is not a string: {parsed}"
            continue
        return raw, sel.strip(), None
    return last_raw, None, last_err

# --- Main ---
def main():
    parser = argparse.ArgumentParser(description="Run multi-GPU inference.")
    parser.add_argument("--model_name", type=str, default="google/gemma-3-4b-it", help="Hugging Face model name.")
    parser.add_argument("--input_file", type=str, default="/home/karulosu/Documents/research/hofstede/data/generated_dataset.json", help="Input JSON file.")
    parser.add_argument("--output_dir", type=str, default="/home/karulosu/Documents/research/hofstede/inference_output", help="Output directory.")
    parser.add_argument("--use_4bit", action="store_true", help="Use 4-bit quantization.")
    parser.add_argument("--num_inferences", type=int, default=3, help="Number of inferences per input.")
    args = parser.parse_args()

    tokenizer, model = load_model_and_tokenizer(args.model_name, args.use_4bit)

    # Create model-specific output directory
    model_output_dir = os.path.join(args.output_dir, args.model_name.replace('/', '_'))
    os.makedirs(model_output_dir, exist_ok=True)
    print(f"[INFO] Output will be saved to: {model_output_dir}")

    # Load input data
    if not os.path.isfile(args.input_file):
        print(f"[CRITICAL] Input file does not exist: {args.input_file}")
        sys.exit(1)
    try:
        with open(args.input_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[CRITICAL] Failed to read/parse input JSON: {e}")
        sys.exit(1)

    if not isinstance(data, list):
        print(f"[CRITICAL] Expected a list of inferences in the input JSON, but got {type(data)}")
        sys.exit(1)

    # Process each item in the input data
    for item_idx, item in enumerate(data):
        question = item.get("question", "").strip()
        parsed_output = item.get("model_output_parsed", [])

        print(f"[INFO] Processing item {item_idx + 1}/{len(data)}")

        for inference_run in range(args.num_inferences):
            output_jsonl_path = os.path.join(model_output_dir, f"inference_results_run_{inference_run + 1}.jsonl")
            
            prompt = build_inference_prompt(question, parsed_output)
            raw_resp, selected_level, error_msg = get_valid_json_output(prompt, tokenizer, model)

            result_entry: Dict[str, Any] = {
                "original_index": item.get("index"),
                "question": question,
                "parsed_output": parsed_output,
                "inference_run": inference_run + 1,
                "selected_level": selected_level,
                "raw_response": raw_resp,
                "error": error_msg
            }

            try:
                with open(output_jsonl_path, "a", encoding="utf-8") as f_jsonl:
                    f_jsonl.write(json.dumps(result_entry, ensure_ascii=False) + "\n")
                print(f"[INFO] Appended result for item {item_idx + 1}, run {inference_run + 1} to {output_jsonl_path}")
            except Exception as e:
                print(f"[ERROR] Failed to append to NDJSON for item {item_idx + 1}, run {inference_run + 1}: {e}")

    print("[INFO] Inference process completed.")

if __name__ == "__main__":
    main()
