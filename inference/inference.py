import os
import sys
import json
import re
import requests
import logging
from typing import Optional, Union, Dict, Any

# ------------------------------------------------------------------------------
# Configure logger
# ------------------------------------------------------------------------------
logger = logging.getLogger("inference_logger")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    "[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
handler.setFormatter(formatter)
logger.addHandler(handler)

# ------------------------------------------------------------------------------
# Configuration variables (replace paths/model name as needed)
# ------------------------------------------------------------------------------
INPUT_JSON_PATH     = "./outputs/hofstede_checkpoint.json"   # Path to JSON file with inferences
OUTPUT_DIR          = "inference_output"              # Folder where both files will be stored
OUTPUT_JSONL_NAME   = "all_inference_results.jsonl"           # NDJSON (line-delimited) filename
FINAL_JSON_NAME     = "all_inference_results.json"            # Final combined JSON array filename
OLLAMA_MODEL_NAME   = "gemma3:4b"                             # Ollama model name
OLLAMA_HOST         = "http://localhost:11434"                # Ollama HTTP host

# Construct full paths inside OUTPUT_DIR
OUTPUT_JSONL_PATH = os.path.join(OUTPUT_DIR, OUTPUT_JSONL_NAME)
FINAL_JSON_PATH   = os.path.join(OUTPUT_DIR, FINAL_JSON_NAME)

# ------------------------------------------------------------------------------
def call_ollama_http(
    prompt: str,
    model_name: str = OLLAMA_MODEL_NAME,
    ollama_host: str = OLLAMA_HOST
) -> str:
    """
    Send a POST to Ollama’s HTTP API at /api/generate with JSON:
        {
          "model": <model_name>,
          "prompt": <prompt>,
          "stream": false
        }

    Returns the raw `response` string (which may include Markdown fences like ```json ... ```).
    If the server errors or returns invalid JSON, returns an empty string.
    """
    url = f"{ollama_host}/api/generate"
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False
    }

    logger.debug(f"Sending request to Ollama at {url} with payload:\n{json.dumps(payload, indent=2)}")
    try:
        resp = requests.post(url, json=payload, timeout=300)
        resp.raise_for_status()
        logger.debug(f"Ollama HTTP status code: {resp.status_code}")
    except requests.RequestException as e:
        logger.error(f"Ollama HTTP request failed: {e}")
        return ""

    try:
        data = resp.json()
        logger.debug(f"Ollama raw JSON response:\n{json.dumps(data, indent=2)}")
    except ValueError:
        logger.error(f"Could not decode JSON response from Ollama: {resp.text}")
        return ""

    if not isinstance(data, dict) or "response" not in data:
        logger.warning(f"Unexpected Ollama JSON format: {json.dumps(data, indent=2)}")
        return ""

    raw_response = data["response"]
    if raw_response is None:
        logger.warning("Ollama returned a null response field.")
        return ""
    if not isinstance(raw_response, str):
        raw_response = str(raw_response)

    logger.debug(f"Ollama raw response (string):\n{raw_response}\n")
    return raw_response.strip()

# ------------------------------------------------------------------------------
def extract_json_from_markdown(text: Union[str, bytes]) -> Optional[str]:
    """
    Given a string (or bytes) that may contain a fenced ```json ... ``` block,
    extract the inner JSON. If no fence is found, assume the entire `text` is valid JSON.
    Return None if input is empty or extraction fails.
    """
    if text is None:
        return None

    if isinstance(text, bytes):
        try:
            text = text.decode("utf-8")
        except Exception as e:
            logger.error(f"Failed to decode bytes to UTF-8: {e}")
            return None

    if not isinstance(text, str):
        text = str(text)

    text = text.strip()
    if not text:
        logger.warning("Empty text received for JSON extraction.")
        return None

    code_block_pattern = re.compile(
        r"```json\s*(\{.*?\})\s*```",
        flags=re.DOTALL
    )
    m = code_block_pattern.search(text)
    if m:
        extracted = m.group(1).strip()
        logger.debug(f"Extracted JSON from markdown block:\n{extracted}\n")
        return extracted

    logger.debug("No fenced JSON block found; treating entire text as JSON.")
    return text

# ------------------------------------------------------------------------------
def get_selected_level(
    prompt_text: str,
    model_name: str = OLLAMA_MODEL_NAME,
    max_retries: int = 3
) -> (str, Optional[str], Optional[str]):
    """
    Calls `call_ollama_http` up to `max_retries` times, extracts JSON, and returns:
      - raw_output: exact string returned by Ollama
      - selected_level: parsed string under "selected_level", or None if failure
      - error_msg: descriptive error if parsing/validation fails, else None
    """
    last_error: Optional[str] = None
    last_raw: str = ""

    for attempt in range(1, max_retries + 1):
        logger.debug(f"Attempt {attempt}/{max_retries} to get selected_level from Ollama.")
        raw = call_ollama_http(prompt_text, model_name=model_name)
        last_raw = raw

        if not raw:
            last_error = f"Attempt {attempt}: empty response from Ollama."
            logger.warning(last_error)
            continue

        extracted = extract_json_from_markdown(raw)
        if extracted is None:
            last_error = f"Attempt {attempt}: could not find JSON content in response."
            logger.warning(last_error)
            continue

        try:
            parsed = json.loads(extracted)
            logger.debug(f"Parsed JSON on attempt {attempt}:\n{json.dumps(parsed, indent=2)}")
        except ValueError as ve:
            snippet = extracted[:200] + ("…" if len(extracted) > 200 else "")
            last_error = (
                f"Attempt {attempt}: invalid JSON payload (first 200 chars):\n"
                f"{snippet}\nError: {ve}"
            )
            logger.error(last_error)
            continue

        if not isinstance(parsed, dict) or "selected_level" not in parsed:
            last_error = f"Attempt {attempt}: JSON missing 'selected_level' key: {parsed}"
            logger.warning(last_error)
            continue

        sel = parsed.get("selected_level")
        if not isinstance(sel, str):
            last_error = f"Attempt {attempt}: 'selected_level' is not a string: {parsed}"
            logger.warning(last_error)
            continue

        logger.info(f"Successfully obtained selected_level on attempt {attempt}: {sel}")
        return raw, sel.strip(), None

    logger.error(f"All {max_retries} attempts failed. Last error: {last_error}")
    return last_raw, None, last_error

# ------------------------------------------------------------------------------
def build_inference_prompt(question: str, parsed_options: Any) -> str:
    """
    Build a prompt that elicits the model's 'internal personality' before classification.
    `parsed_options` is a Python list-of-dicts (JSON-serializable).
    """
    options_str = json.dumps(parsed_options, indent=2, ensure_ascii=False)
    prompt = (
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
    logger.debug(f"Built inference prompt:\n{prompt}\n")
    return prompt

# ------------------------------------------------------------------------------
def process_inference_file(
    input_path: str = INPUT_JSON_PATH,
    output_dir: str = OUTPUT_DIR,
    jsonl_path: str = OUTPUT_JSONL_PATH,
    final_json_path: str = FINAL_JSON_PATH,
    model_name: str = OLLAMA_MODEL_NAME
):
    """
    Reads a JSON list from `input_path` where each item has:
      - index
      - question
      - domain
      - dimension_code
      - model_output_parsed (list of {statement, level})

    For each item:
      1. Build a prompt via build_inference_prompt().
      2. Call get_selected_level().
      3. Immediately append the result to `jsonl_path` (newline-delimited).
    After all items, reconstruct a final combined JSON array at `final_json_path`.
    """
    if not os.path.isfile(input_path):
        logger.critical(f"Input file does not exist: {input_path}")
        sys.exit(1)

    # Ensure output directory exists
    try:
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Ensured output directory exists: {output_dir}")
    except Exception as e:
        logger.critical(f"Could not create output directory '{output_dir}': {e}")
        sys.exit(1)

    logger.info(f"Reading input file: {input_path}")
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.critical(f"Failed to read/parse input JSON: {e}")
        sys.exit(1)

    if not isinstance(data, list):
        logger.critical(f"Expected a list of inferences in the input JSON, but got {type(data)}")
        sys.exit(1)

    # Initialize (or truncate) the NDJSON file
    try:
        open(jsonl_path, "w", encoding="utf-8").close()
        logger.info(f"Initialized NDJSON file at: {jsonl_path}")
    except Exception as e:
        logger.error(f"Could not initialize NDJSON file: {e}")
        sys.exit(1)

    for item in data:
        idx = item.get("index")
        question = item.get("question", "").strip()
        domain = item.get("domain", "").strip()
        dimension_code = item.get("dimension_code", "").strip()
        parsed_output = item.get("model_output_parsed", [])

        logger.info(f"Processing inference index={idx}, dimension={dimension_code}")

        prompt = build_inference_prompt(question, parsed_output)
        raw_resp, selected_level, error_msg = get_selected_level(prompt, model_name=model_name)

        result_entry: Dict[str, Any] = {
            "index": idx,
            "question": question,
            "domain": domain,
            "dimension_code": dimension_code,
            "parsed_output": parsed_output,
            "selected_level": selected_level,
            "error": error_msg
        }

        try:
            with open(jsonl_path, "a", encoding="utf-8") as f_jsonl:
                f_jsonl.write(json.dumps(result_entry, ensure_ascii=False) + "\n")
            logger.debug(f"Appended result for index={idx} to {jsonl_path}")
        except Exception as e:
            logger.error(f"[Index {idx}] Failed to append to NDJSON: {e}")

    # Reconstruct final JSON array from NDJSON
    try:
        combined = []
        with open(jsonl_path, "r", encoding="utf-8") as f_jsonl:
            for line in f_jsonl:
                line = line.strip()
                if not line:
                    continue
                combined.append(json.loads(line))
        with open(final_json_path, "w", encoding="utf-8") as f_final:
            json.dump(combined, f_final, indent=2, ensure_ascii=False)
        logger.info(f"Reconstructed final JSON array to {final_json_path} (total {len(combined)} entries)")
    except Exception as e:
        logger.error(f"Failed to generate final JSON array: {e}")

# ------------------------------------------------------------------------------
if __name__ == "__main__":
    process_inference_file()
