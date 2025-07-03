import os
import sys
import json
import re
import requests
from datasets import load_dataset
from typing import Optional, Union, List, Dict, Any

# ------------------------------------------------------------------------------
# Mapping from the dataset’s dimension code -> (full phrase, low_label, high_label)
# ------------------------------------------------------------------------------

DIMENSION_MAP = {
    "PDI": {
        "full": "Power distance index",
        "low": "low power distance index",
        "high": "high power distance index"
    },
    "IDV": {
        "full": "Individualism vs. collectivism",
        "low": "individualism",
        "high": "collectivism"
    },
    "UAI": {
        "full": "Uncertainty avoidance",
        "low": "low uncertainty avoidance",
        "high": "high uncertainty avoidance"
    },
    "MAS": {
        "full": "Motivation towards achievement and success",
        "low": "low motivation towards achievement and success",
        "high": "high motivation towards achievement and success"
    },
    "LTO": {
        "full": "Long-term orientation vs. short-term orientation",
        "low": "long-term orientation",
        "high": "short-term orientation"
    },
    "IVR": {
        "full": "Indulgence vs. restraint",
        "low": "indulgence",
        "high": "restraint"
    }
}


def load_entire_cdeval():
    """
    Load all available splits ('train', 'validation', 'test') from Rykeryuhang/CDEval.
    Returns a DatasetDict mapping split_name -> Dataset.
    """
    try:
        return load_dataset("Rykeryuhang/CDEval", split=None)
    except Exception as e:
        print(f"[ERROR] Failed to load CDEval dataset: {e}", file=sys.stderr)
        return {}


def format_prompt(question: str, domain: str, dimension_code: str) -> str:
    """
    Build the Hofstede prompt given question, domain, and dimension_code.

    Uses DIMENSION_MAP to fill in the full phrase, low_label, and high_label.
    """
    mapping = DIMENSION_MAP[dimension_code]
    full_phrase = mapping["full"]
    low_label = mapping["low"]
    high_label = mapping["high"]

    template = (
        "Prepare a Hofstede test question and answer:\n\n"
        "Question: {question}\n\n"
        "Domain: {domain}\n"
        "Dimension: {full_phrase}\n\n"
        "Instruction: Provide five statements that span levels 1 to 5 on the domain theme. Return "
        "the result in JSON format, where each item has\n"
        "  - \"statement\": <the sentence>\n"
        "  - \"level\": <an integer from 1 to 5>\n\n"
        "- Level 1 is {low_label}\n"
        "- Level 5 is {high_label}"
    )

    return template.format(
        question=question,
        domain=domain,
        full_phrase=full_phrase,
        low_label=low_label,
        high_label=high_label
    )


def call_ollama_http(
    prompt: str,
    model_name: str = "gemma3:4b",
    ollama_host: str = "http://localhost:11434"
) -> str:
    """
    Send a POST to Ollama’s HTTP API at /api/generate with JSON:
      {
        "model": <model_name>,
        "prompt": <prompt>,
        "stream": false
      }

    Returns the raw `response` field as a string (which may include Markdown
    fences like ```json ... ```). If the server errors or returns invalid JSON,
    returns an empty string.
    """
    url = f"{ollama_host}/api/generate"
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False
    }

    try:
        resp = requests.post(url, json=payload, timeout=300)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"[ERROR] Ollama HTTP request failed: {e}", file=sys.stderr)
        return ""

    try:
        data = resp.json()
    except ValueError:
        print(f"[ERROR] Could not decode JSON response from Ollama: {resp.text}", file=sys.stderr)
        return ""

    if not isinstance(data, dict) or "response" not in data:
        print(f"[WARNING] Unexpected Ollama JSON format: {json.dumps(data, indent=2)}", file=sys.stderr)
        return ""

    raw_response = data["response"]
    if raw_response is None:
        return ""
    if not isinstance(raw_response, str):
        raw_response = str(raw_response)
    return raw_response.strip()


def extract_json_from_markdown(text: Union[str, bytes]) -> Optional[str]:
    """
    Given a string (or bytes) that may contain a fenced ```json ... ``` block,
    extract the inner JSON. If no fence is found, assume the entire `text` is
    valid JSON. Return None if parsing fails or if input is empty.
    """
    if text is None:
        return None

    if isinstance(text, bytes):
        try:
            text = text.decode("utf-8")
        except Exception:
            return None

    if not isinstance(text, str):
        text = str(text)

    text = text.strip()
    if text == "":
        return None

    code_block_pattern = re.compile(
        r"```json\s*(\[\s*\{.*?\}\s*\])\s*```",
        flags=re.DOTALL
    )

    m = code_block_pattern.search(text)
    if m:
        return m.group(1).strip()

    return text


def get_valid_json_output(
    prompt_text: str,
    max_retries: int = 3
) -> (str, Optional[List[Dict[str, Any]]], Optional[str]):
    """
    Calls `call_ollama_http` repeatedly (up to `max_retries`) to fetch a raw string,
    extracts JSON from any ```json … ``` fence, then parses + validates it.

    Returns a tuple:
      ( raw_output, parsed_list_or_None, error_msg_or_None )

    - raw_output:    the exact string that Ollama returned (could be Markdown‐wrapped)
    - parsed_list:   a List[{"statement": str, "level": int}, …] if successful, else None
    - error_msg:     a descriptive error if it failed, else None
    """
    last_error: Optional[str] = None
    last_raw: str = ""

    for attempt in range(1, max_retries + 1):
        raw = call_ollama_http(prompt_text)
        last_raw = raw
        if not raw:
            last_error = f"Attempt {attempt}: empty response from Ollama."
            continue

        # 1) Extract JSON‐looking text
        extracted = extract_json_from_markdown(raw)
        if extracted is None:
            last_error = f"Attempt {attempt}: could not find JSON content in response."
            continue

        # 2) Parse it as JSON
        try:
            parsed = json.loads(extracted)
        except ValueError as ve:
            snippet = extracted[:200] + ("…" if len(extracted) > 200 else "")
            last_error = (
                f"Attempt {attempt}: invalid JSON payload (first 200 chars):\n"
                f"{snippet}\nError: {ve}"
            )
            continue

        # 3) Validate that it's a list of dicts with "statement" and integer "level"
        if not isinstance(parsed, list):
            last_error = f"Attempt {attempt}: top‐level JSON is not a list (got {type(parsed)})."
            continue

        seen_levels = set()
        ok = True
        for idx, item in enumerate(parsed):
            if not isinstance(item, dict):
                last_error = f"Attempt {attempt}: item {idx} is not a dict (got {type(item)})."
                ok = False
                break
            if "statement" not in item or "level" not in item:
                last_error = (
                    f"Attempt {attempt}: missing 'statement' or 'level' in item {idx}: {item}"
                )
                ok = False
                break
            if not isinstance(item["level"], int):
                last_error = (
                    f"Attempt {attempt}: 'level' is not an int in item {idx}: {item}"
                )
                ok = False
                break
            seen_levels.add(item["level"])

        if not ok:
            continue

        # 4) Ensure exactly levels {1,2,3,4,5}
        required = {1, 2, 3, 4, 5}
        if seen_levels != required:
            last_error = (
                f"Attempt {attempt}: JSON contained levels {sorted(seen_levels)}, "
                f"but expected exactly {sorted(required)}."
            )
            continue

        # Success: return raw + parsed + no error
        return raw, parsed, None

    # If we exit the loop, all attempts failed
    return last_raw, None, last_error


def main():
    """
    1. Load all splits of the Rykeryuhang/CDEval dataset.
    2. Attempt to load any existing checkpoint (partial results).
    3. For each example in each split:
       - Skip if already in checkpoint.
       - Otherwise:
         - Read 'Question', 'Domain', 'Dimension' (a code string).
         - Map the code to its full phrase and labels via DIMENSION_MAP.
         - Build the prompt with format_prompt().
         - Call get_valid_json_output() to retrieve raw and parsed JSON.
         - Append this entry to both:
           a) the in‐memory list `all_entries`
           b) the checkpoint file (overwrite it each time so it remains up to date)
    4. After processing all examples, write the final complete JSON array to outputs/hofstede_generated.json.
    """
    print("Loading all splits of Rykeryuhang/CDEval…")
    dataset_splits = load_entire_cdeval()
    if not dataset_splits:
        print("[ERROR] No dataset splits loaded; exiting.", file=sys.stderr)
        sys.exit(1)

    output_dir = "outputs"
    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception as e:
        print(f"[ERROR] Cannot create output directory '{output_dir}': {e}", file=sys.stderr)
        sys.exit(1)

    checkpoint_path = os.path.join(output_dir, "hofstede_checkpoint.json")
    final_output_path = os.path.join(output_dir, "hofstede_generated.json")

    # Load checkpoint if it exists
    all_entries: List[Dict[str, Any]] = []
    processed_keys = set()  # set of ("split", index) tuples that have been done
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as cp_file:
                all_entries = json.load(cp_file)
            for entry in all_entries:
                key = (entry.get("split"), entry.get("index"))
                processed_keys.add(key)
            print(f"[INFO] Loaded checkpoint with {len(all_entries)} entries.")
        except Exception as e:
            print(f"[ERROR] Failed to load checkpoint: {e}", file=sys.stderr)
            print("[INFO] Starting from scratch.")
            all_entries = []
            processed_keys.clear()

    # Iterate over splits and examples
    for split_name, split_dataset in dataset_splits.items():
        print(f"Processing split '{split_name}' ({len(split_dataset)} examples)…")
        for idx, example in enumerate(split_dataset):
            key = (split_name, idx)
            if key in processed_keys:
                # This example was already processed in a previous run
                continue

            try:
                question = example.get("Question")
                domain = example.get("Domain")
                dimension_code = example.get("Dimension")

                # Validate fields
                if question is None or domain is None or dimension_code is None:
                    print(f"[WARN] Skipping {split_name}#{idx}: missing Question/Domain/Dimension")
                    continue
                if not isinstance(dimension_code, str) or dimension_code not in DIMENSION_MAP:
                    print(f"[WARN] Skipping {split_name}#{idx}: unexpected dimension '{dimension_code}'")
                    continue

                prompt_text = format_prompt(question, domain, dimension_code)

                print(f"- Generating & parsing JSON for {split_name}#{idx} (Dimension={dimension_code}) …")
                raw_output, parsed_output, error_msg = get_valid_json_output(prompt_text, max_retries=3)

                entry = {
                    "split": split_name,
                    "index": idx,
                    "question": question,
                    "domain": domain,
                    "dimension_code": dimension_code,
                    "prompt": prompt_text,
                    "model_output_raw": raw_output,
                    "model_output_parsed": parsed_output,
                    "error": error_msg
                }

            except Exception as e:
                print(f"[ERROR] Unexpected failure on {split_name}#{idx}: {e}", file=sys.stderr)
                entry = {
                    "split": split_name,
                    "index": idx,
                    "question": example.get("Question"),
                    "domain": example.get("Domain"),
                    "dimension_code": example.get("Dimension"),
                    "prompt": format_prompt(
                        example.get("Question", ""),
                        example.get("Domain", ""),
                        example.get("Dimension", "") if isinstance(example.get("Dimension"), str) else ""
                    ),
                    "model_output_raw": "",
                    "model_output_parsed": None,
                    "error": f"Unexpected exception: {e}"
                }

            # Append the new entry to our in-memory list
            all_entries.append(entry)
            processed_keys.add(key)

            # Immediately write out the updated checkpoint
            try:
                with open(checkpoint_path, "w", encoding="utf-8") as cp_file:
                    json.dump(all_entries, cp_file, indent=2, ensure_ascii=False)
            except Exception as e:
                print(f"[ERROR] Failed to write checkpoint after {split_name}#{idx}: {e}", file=sys.stderr)
                # We choose to continue anyway, so that even if checkpoint writing fails once,
                # we still keep processing. The final write will hopefully succeed.

    # After all examples have been processed, write the final JSON array
    try:
        with open(final_output_path, "w", encoding="utf-8") as fout:
            json.dump(all_entries, fout, indent=2, ensure_ascii=False)
        print(f"\nFinished. Final output saved to: {final_output_path}")
    except Exception as e:
        print(f"[ERROR] Failed to write final output: {e}", file=sys.stderr)
        print(f"[INFO] Checkpoint remains at: {checkpoint_path}")


if __name__ == "__main__":
    main()

