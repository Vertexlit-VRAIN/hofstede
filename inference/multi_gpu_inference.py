import os
import sys
import json
import re
import argparse
import gc
from typing import Optional, Union, Dict, Any, List

import torch
import torch._dynamo as dynamo
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset

# --- Dynamo config ---
dynamo.config.recompile_limit = 64
dynamo.config.accumulated_recompile_limit = 512

# --- Model List ---
#MODELS_TO_RUN = [
#    "google/gemma-3-1b-it", "google/gemma-3-4b-it", "google/gemma-3-12b-it", "google/gemma-3-27b-it",
#    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
#    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
#    "Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B", "Qwen/Qwen3-8B",
#    "Qwen/Qwen3-14B", "Qwen/Qwen3-30B", "Qwen/Qwen3-32B",
#    "mistralai/Mistral-7B-Instruct-v0.3",
#    "meta-llama/Llama-3.1-8B-Instruct",
#    "ibm-granite/granite-3.3-2b-instruct", "ibm-granite/granite-3.3-8b-instruct",
#    "microsoft/phi-4",
#]

MODELS_TO_RUN = [

]

def load_model_and_tokenizer(model_name, use_4bit):
    print(f"[INFO] Loading {model_name}…")
    quant_cfg = None
    if use_4bit:
        quant_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")
    
    tok = AutoTokenizer.from_pretrained(model_name, token=os.getenv("HF_TOKEN"))
    mod = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=quant_cfg,
    )
    mod.eval()
    return tok, mod

# --- Prompt Builder ---
def build_inference_prompt(question: str, parsed_options: Any) -> str:
    options_str = json.dumps(parsed_options, indent=2, ensure_ascii=False)

    return f"""You are an LLM with a unique voice and style—imagine you have your own preferences, quirks, and tone. Read the user’s statement below and silently reflect (do **not** reveal your private thoughts). Then, from the provided options, pick the LEVEL that most closely matches how you would interpret this statement.

    Statement:
    {question}

    Options:
    {options_str}

    **Instructions (strictly follow):**
    1. Think privately. **Do not output this reflection.**
    2. Select the LEVEL that best aligns with your reaction.
    3. Respond with **only** valid JSON, wrapped in a markdown code block, exactly in this format:

    ```json
    {{"selected_level": "<integer from 1 to 5>"}}
    ```"""

# --- Call Model ---
@torch._dynamo.disable
def call_model(prompt: str, tokenizer, model, max_new_tokens: int = 4096) -> str:
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
            top_p=None,
            top_k=None,
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

    # 1) Try a ```json …``` block
    fenced = re.compile(r"```json\s*([\s\S]*?)\s*```", flags=re.DOTALL)
    m = fenced.search(text)
    if m:
        return m.group(1).strip()

    # 2) Try any fenced code block and see if it parses
    any_fence = re.compile(r"```[\s\S]*?```", flags=re.DOTALL)
    m2 = any_fence.search(text)
    if m2:
        inner = m2.group(0).strip("`").strip()
        try:
            json.loads(inner)
            return inner
        except json.JSONDecodeError:
            pass

    # 3) Fallback: scan for the first balanced { … } that parses
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                candidate = text[start : i + 1]
                try:
                    json.loads(candidate)
                    return candidate
                except json.JSONDecodeError:
                    start = None

    return None

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
        if isinstance(sel, str):
            try:
                sel = int(sel.strip())
            except ValueError:
                last_err = f"Attempt {attempt}: 'selected_level' string could not be converted to int: {parsed}"
                continue
        elif not isinstance(sel, int):
            last_err = f"Attempt {attempt}: 'selected_level' is not a string or int: {parsed}"
            continue
        return raw, sel, None
    return last_raw, None, last_err

def run_inference_for_model(model_name: str, args: argparse.Namespace):
    """
    Runs the full inference pipeline for a single model.
    """
    tokenizer, model = load_model_and_tokenizer(model_name, args.use_4bit)

    model_output_dir = os.path.join(args.output_dir, model_name.replace('/', '_'))
    os.makedirs(model_output_dir, exist_ok=True)
    print(f"[INFO] Output will be saved to: {model_output_dir}")

    if not os.path.isfile(args.input_file):
        print(f"[CRITICAL] Input file does not exist: {args.input_file}")
        return

    try:
        with open(args.input_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception as e:
        print(f"[CRITICAL] Failed to read input file: {e}")
        return

    for item_idx, line in enumerate(lines):
        try:
            item = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"[ERROR] Skipping invalid JSON on line {item_idx + 1}: {e}")
            continue

        question = item.get("question", "").strip()
        parsed_output = item.get("model_output_parsed", [])

        print(f"[INFO] [{model_name}] Processing item {item_idx + 1}/{len(lines)}")

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

    print(f"[INFO] Inference process completed for model {model_name}.")
    
    # Clean up memory
    del tokenizer
    del model
    gc.collect()
    torch.cuda.empty_cache()


# --- Main ---
def main():
    parser = argparse.ArgumentParser(description="Run multi-GPU inference for a list of models.")
    parser.add_argument("--model_name", type=str, default=None, help="Optional: A specific Hugging Face model name to run. If not provided, the script will iterate through the MODELS_TO_RUN list.")
    parser.add_argument("--input_file", type=str, default="validated_data.jsonl", help="Input JSONL file.")
    parser.add_argument("--output_dir", type=str, default="inference_output", help="Output directory.")
    parser.add_argument("--use_4bit", action="store_true", help="Use 4-bit quantization (recommended for large models).")
    parser.add_argument("--num_inferences", type=int, default=3, help="Number of inferences per input.")
    args = parser.parse_args()

    models_to_process = [args.model_name] if args.model_name else MODELS_TO_RUN

    for model_name in models_to_process:
        print(f"\n{'='*80}\n[MASTER] Starting processing for model: {model_name}\n{'='*80}\n")
        try:
            run_inference_for_model(model_name, args)
        except Exception as e:
            print(f"[CRITICAL] Unhandled exception for model {model_name}: {e}", file=sys.stderr)
            print(f"[CRITICAL] Skipping to next model.")
            # Clean up memory
            gc.collect()
            torch.cuda.empty_cache()
            continue

if __name__ == "__main__":
    main()
