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
# Configuration variables (replace paths/model names as needed)
# ------------------------------------------------------------------------------
INPUT_JSON_PATH   = "./outputs/hofstede_checkpoint.json"  # Path to JSON file with inferences
OUTPUT_DIR        = "multi_inference_results"                    # Base folder where model subfolders will be created
NUM_RUNS          = 3                                            # <<< CHANGED: Number of times to run inference for each model

# List of Ollama models (up to 20 or more as desired)
OLLAMA_MODELS     = [
    "gemma3:4b",
    "deepseek-r1:7b",
]

OPENAI_MODELS     = [
    # "gpt-3.5-turbo",
    # "gpt-4",
]

OLLAMA_HOST       = "http://localhost:11434"
USE_OPENAI        = False

# ------------------------------------------------------------------------------
def call_ollama_http(
    prompt: str,
    model_name: str,
    ollama_host: str = OLLAMA_HOST
) -> str:
    """
    Send a POST to Ollama’s HTTP API at /api/generate with JSON:
      {
        "model": <model_name>,
        "prompt": <prompt>,
        "stream": false
      }
    Returns the raw `response` string (which may include chain-of-thought, etc.).
    On error or invalid JSON, returns an empty string.
    """
    url = f"{ollama_host}/api/generate"
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False
    }

    logger.debug(f"[Ollama:{model_name}] Sending request to {url} with payload:\n{json.dumps(payload, indent=2)}")
    try:
        resp = requests.post(url, json=payload, timeout=300)
        resp.raise_for_status()
    except requests.RequestException as e:
        logger.error(f"[Ollama:{model_name}] HTTP request failed: {e}")
        return ""

    try:
        data = resp.json()
    except ValueError:
        logger.error(f"[Ollama:{model_name}] Could not decode JSON response: {resp.text}")
        return ""

    if not isinstance(data, dict) or "response" not in data:
        logger.warning(f"[Ollama:{model_name}] Unexpected JSON format: {json.dumps(data, indent=2)}")
        return ""

    raw_response = data["response"]
    if raw_response is None:
        logger.warning(f"[Ollama:{model_name}] Response field was null.")
        return ""
    if not isinstance(raw_response, str):
        raw_response = str(raw_response)

    logger.debug(f"[Ollama:{model_name}] Raw response:\n{raw_response}\n")
    return raw_response.strip()

# ------------------------------------------------------------------------------
def call_openai_api(
    prompt: str,
    model_name: str
) -> str:
    """
    Call OpenAI chat completion API with given prompt and model_name.
    Returns the raw text response (which may include chain-of-thought).
    On error, returns empty string.
    Requires environment variable OPENAI_API_KEY.
    """
    try:
        import openai
    except ImportError:
        logger.error("OpenAI library not installed. Cannot call OpenAI API.")
        return ""

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("Environment variable OPENAI_API_KEY not set.")
        return ""
    openai.api_key = api_key

    try:
        response = openai.ChatCompletion.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        text = response.choices[0].message.content
        logger.debug(f"[OpenAI:{model_name}] Raw response:\n{text}\n")
        return text.strip()
    except Exception as e:
        logger.error(f"[OpenAI:{model_name}] API call failed: {e}")
        return ""

# ------------------------------------------------------------------------------
def extract_json_from_markdown(text: Union[str, bytes]) -> Optional[str]:
    """
    Extract the JSON object that contains "selected_level" from any surrounding text.
    1. If a fenced ```json { … } ``` block is present, extract that inner object.
    2. Otherwise, find the first { … } substring that contains "selected_level" and return it.
    Return None if no such JSON object is found.
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

    # 1) Check for fenced ```json { … } ```
    fence_pattern = re.compile(r"```json\s*(\{.*?\})\s*```", flags=re.DOTALL)
    m = fence_pattern.search(text)
    if m:
        candidate = m.group(1).strip()
        # Ensure it contains "selected_level"
        if '"selected_level"' in candidate:
            logger.debug(f"Extracted JSON from fenced block:\n{candidate}\n")
            return candidate

    # 2) Otherwise, search for any { … } that contains "selected_level"
    #    We match minimal braces that include the key:
    brace_pattern = re.compile(r"\{[^{}]*\"selected_level\"[^{}]*\}", flags=re.DOTALL)
    m2 = brace_pattern.search(text)
    if m2:
        candidate = m2.group(0).strip()
        logger.debug(f"Extracted JSON containing selected_level:\n{candidate}\n")
        return candidate

    logger.debug("No JSON object with 'selected_level' detected in the text.")
    return None

# ------------------------------------------------------------------------------
def get_selected_level(
    prompt_text: str,
    model_name: str,
    use_openai: bool = False,
    max_retries: int = 3
) -> (str, Optional[str], Optional[str]):
    """
    Attempts up to max_retries to call the chosen model (Ollama or OpenAI),
    then extracts the JSON object containing "selected_level" and returns:
      - raw_response: exact full string (with any extra text)
      - selected_level: value under "selected_level", or None if parsing fails
      - error_msg: descriptive error if parsing/validation fails, else None
    """
    last_error: Optional[str] = None
    last_raw: str = ""

    for attempt in range(1, max_retries + 1):
        logger.debug(f"[{model_name}] Attempt {attempt}/{max_retries} to get selected_level.")
        if use_openai:
            raw = call_openai_api(prompt_text, model_name)
        else:
            raw = call_ollama_http(prompt_text, model_name)
        last_raw = raw

        if not raw:
            last_error = f"Attempt {attempt}: empty response."
            logger.warning(f"[{model_name}] {last_error}")
            continue

        # Extract JSON object containing "selected_level"
        extracted = extract_json_from_markdown(raw)
        if extracted is None:
            last_error = f"Attempt {attempt}: no JSON with 'selected_level' found."
            logger.warning(f"[{model_name}] {last_error}")
            continue

        # Parse that JSON substring
        try:
            parsed = json.loads(extracted)
            logger.debug(f"[{model_name}] Parsed JSON:\n{json.dumps(parsed, indent=2)}")
        except ValueError as ve:
            snippet = extracted[:200] + ("…" if len(extracted) > 200 else "")
            last_error = (
                f"Attempt {attempt}: invalid JSON payload (first 200 chars):\n"
                f"{snippet}\nError: {ve}"
            )
            logger.error(f"[{model_name}] {last_error}")
            continue

        # Validate "selected_level" key
        if not isinstance(parsed, dict) or "selected_level" not in parsed:
            last_error = f"Attempt {attempt}: missing 'selected_level' key: {parsed}"
            logger.warning(f"[{model_name}] {last_error}")
            continue

        sel = parsed.get("selected_level")
        if not isinstance(sel, str):
            last_error = f"Attempt {attempt}: 'selected_level' is not a string: {parsed}"
            logger.warning(f"[{model_name}] {last_error}")
            continue

        logger.info(f"[{model_name}] Obtained selected_level on attempt {attempt}: {sel}")
        return raw, sel.strip(), None

    logger.error(f"[{model_name}] All {max_retries} attempts failed. Last error: {last_error}")
    return last_raw, None, last_error

# ------------------------------------------------------------------------------
def build_inference_prompt(question: str, parsed_options: Any) -> str:
    """
    Build the prompt that elicits the model's 'internal personality' before classification.
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
def process_for_model(
    model_name: str,
    use_openai: bool,
    input_path: str,
    output_dir: str,
    run_index: int  # <<< CHANGED: Added run_index parameter
):
    """
    Runs the inference pipeline for a single model for a specific run,
    storing results in a subfolder under output_dir named after the model.
    """
    safe_model_name = model_name.replace("/", "_").replace(":", "_")
    model_folder = os.path.join(output_dir, safe_model_name)
    
    # <<< CHANGED: Use run_index to create unique filenames for each run
    jsonl_path = os.path.join(model_folder, f"results_run_{run_index}.jsonl")
    final_json_path = os.path.join(model_folder, f"results_run_{run_index}.json")

    # Ensure model-specific folder exists
    try:
        os.makedirs(model_folder, exist_ok=True)
        logger.info(f"[{model_name}] [Run {run_index}] Created/verified model folder: {model_folder}")
    except Exception as e:
        logger.critical(f"[{model_name}] [Run {run_index}] Could not create folder '{model_folder}': {e}")
        return

    # Read input data
    if not os.path.isfile(input_path):
        logger.error(f"[{model_name}] [Run {run_index}] Input file does not exist: {input_path}")
        return

    try:
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"[{model_name}] [Run {run_index}] Failed to read/parse input JSON: {e}")
        return

    if not isinstance(data, list):
        logger.error(f"[{model_name}] [Run {run_index}] Expected a list of inferences, got {type(data)}")
        return

    # Initialize/truncate NDJSON file
    try:
        open(jsonl_path, "w", encoding="utf-8").close()
        logger.info(f"[{model_name}] [Run {run_index}] Initialized NDJSON at: {jsonl_path}")
    except Exception as e:
        logger.error(f"[{model_name}] [Run {run_index}] Could not initialize NDJSON file: {e}")
        return

    # Process each inference
    for item in data:
        idx = item.get("index")
        question = item.get("question", "").strip()
        domain = item.get("domain", "").strip()
        dimension_code = item.get("dimension_code", "").strip()
        parsed_output = item.get("model_output_parsed", [])

        logger.info(f"[{model_name}] [Run {run_index}] Processing index={idx}, dimension={dimension_code}")

        prompt = build_inference_prompt(question, parsed_output)
        raw_resp, selected_level, error_msg = get_selected_level(
            prompt, model_name, use_openai
        )

        # <<< CHANGED: Add run_index to the result dictionary
        result_entry: Dict[str, Any] = {
            "run_index": run_index,
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
            logger.debug(f"[{model_name}] [Run {run_index}] Appended result for index={idx}")
        except Exception as e:
            logger.error(f"[{model_name}] [Run {run_index}] [Index {idx}] Failed to append to NDJSON: {e}")

    # Reconstruct final JSON array
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
        logger.info(f"[{model_name}] [Run {run_index}] Final JSON written to {final_json_path} (total {len(combined)} entries)")
    except Exception as e:
        logger.error(f"[{model_name}] [Run {run_index}] Failed to generate final JSON array: {e}")

# ------------------------------------------------------------------------------
def main():
    # Ensure base output directory exists
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        logger.info(f"Created/verified base output directory: {OUTPUT_DIR}")
    except Exception as e:
        logger.critical(f"Could not create base output directory '{OUTPUT_DIR}': {e}")
        sys.exit(1)
    
    for ollama_model in OLLAMA_MODELS:
        for run_idx in range(1, NUM_RUNS + 1):
            logger.info(f"--- Starting Run {run_idx}/{NUM_RUNS} for model {ollama_model} ---")
            process_for_model(
                model_name=ollama_model,
                use_openai=False,
                input_path=INPUT_JSON_PATH,
                output_dir=OUTPUT_DIR,
                run_index=run_idx
            )

    if USE_OPENAI:
        for openai_model in OPENAI_MODELS:
            for run_idx in range(1, NUM_RUNS + 1):
                logger.info(f"--- Starting Run {run_idx}/{NUM_RUNS} for model {openai_model} ---")
                process_for_model(
                    model_name=openai_model,
                    use_openai=True,
                    input_path=INPUT_JSON_PATH,
                    output_dir=OUTPUT_DIR,
                    run_index=run_idx
                )

# ------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
