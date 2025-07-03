import gradio as gr
import json
import os
import argparse
from functools import partial
from threading import Lock

# --- Nombres de las carpetas ---
INPUT_FOLDER = "inputs"
OUTPUT_FOLDER = "outputs"

# --- Variables globales que se configurarán al inicio ---
INPUT_FILE = None
OUTPUT_FILE = None

DIMENSION_MAP = {
    "PDI": {"full": "Power distance index", "low": "low power distance index", "high": "high power distance index"},
    "IDV": {"full": "Individualism vs. collectivism", "low": "individualism", "high": "collectivism"},
    "UAI": {"full": "Uncertainty avoidance", "low": "low uncertainty avoidance", "high": "high uncertainty avoidance"},
    "MAS": {"full": "Motivation towards achievement and success", "low": "low motivation towards achievement and success", "high": "high motivation towards achievement and success"},
    "LTO": {"full": "Long-term orientation vs. short-term orientation", "low": "long-term orientation", "high": "short-term orientation"},
    "IVR": {"full": "Indulgence vs. restraint", "low": "indulgence", "high": "restraint"},
}

# --- State Management & Data ---
lock = Lock()
current_index = 0
ALL_DATA = []

def load_progress():
    global current_index
    try:
        if os.path.exists(OUTPUT_FILE):
            with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
                current_index = sum(1 for _ in f)
        else:
            current_index = 0
    except FileNotFoundError:
        current_index = 0

def get_item(index):
    if 0 <= index < len(ALL_DATA):
        return ALL_DATA[index]
    return None

def format_progress_bar(current, total):
    if total == 0: return "No data loaded."
    if current >= total: return f"Completed {total}/{total} (100%)"
    percent = (current / total)
    bar = '█' * int(30 * percent) + '─' * (30 - int(30 * percent))
    return f"{bar} {current}/{total} ({percent:.0%})"

def save_result(evaluation_status, item_data, s1, s2, s3, s4, s5):
    global current_index
    with lock:
        # This check prevents stale updates and is still necessary.
        if not item_data or item_data.get('index') != current_index:
             print(f"Warning: Stale update detected. Browser index: {item_data.get('index')}, Server index: {current_index}. Ignoring.")
             return load_next_item()
        
        result_item = item_data.copy()
        result_item["evaluation"] = evaluation_status

        # --- LOGIC CORRECTION ---
        # Only overwrite the statements if the evaluation is 'bad' or a 'fix'.
        # If the evaluation is 'good', we preserve the original statements from `item_data`,
        # ignoring any edits made in the UI textboxes.
        if evaluation_status != "good":
            result_item["model_output_parsed"] = [
                {"statement": s1, "level": 1}, {"statement": s2, "level": 2},
                {"statement": s3, "level": 3}, {"statement": s4, "level": 4},
                {"statement": s5, "level": 5},
            ]
        # If evaluation_status is "good", we do nothing, and result_item
        # retains the original 'model_output_parsed' from the state.
        # --- END CORRECTION ---
        
        with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
            f.write(json.dumps(result_item) + '\n')
            
        current_index += 1
        return load_next_item()

def load_next_item():
    item = get_item(current_index)
    total_items = len(ALL_DATA)
    progress_bar_text = format_progress_bar(current_index, total_items)
    
    if item is None: # Completion State
        return ("", "", "All items have been evaluated!", gr.update(visible=False), "## All Done!",
                (gr.update(value="", interactive=False),) * 5, (gr.update(visible=False),) * 3,
                progress_bar_text, None)

    # Add the index to the item dictionary before returning it.
    item['index'] = current_index

    domain = item.get("domain", "N/A")
    dim_code = item.get("dimension_code", "N/A")
    dim_info = DIMENSION_MAP.get(dim_code, {"full": "Unknown", "low": "N/A", "high": "N/A"})
    dimension_text = f"{dim_info['full']} (Low: {dim_info['low']}, High: {dim_info['high']})"
    question = item.get("question", "N/A")
    
    if item.get("error") is not None: # Format Error Mode
        raw_output_update = gr.update(value=item.get("model_output_raw", ""), visible=True)
        statements_title_update = gr.update(value="### Manually Enter Corrected Statements Below")
        statement_updates = (gr.update(value="", interactive=True),) * 5
        button_updates = (gr.update(visible=False), gr.update(visible=False), gr.update(visible=True))
    else: # Normal Mode
        raw_output_update = gr.update(visible=False)
        statements_title_update = gr.update(value="### Model Output (Editable)")
        parsed = item.get("model_output_parsed", [])
        statements = [p.get("statement", "") for p in parsed]
        statements.extend([""] * (5 - len(statements)))
        statement_updates = tuple(gr.update(value=s, interactive=True) for s in statements)
        button_updates = (gr.update(visible=True), gr.update(visible=True), gr.update(visible=False))

    return (domain, dimension_text, question, raw_output_update, statements_title_update
            ) + statement_updates + button_updates + (progress_bar_text, item)

def get_original_statement(item_data, statement_index):
    if item_data and item_data.get("model_output_parsed") and len(item_data["model_output_parsed"]) > statement_index:
        return item_data["model_output_parsed"][statement_index].get("statement", "")
    return ""

def create_app(part_id):
    with gr.Blocks(theme=gr.themes.Default(), title=f"Hofstede Evaluator (Part {part_id})") as app:
        current_item_data = gr.State()
        gr.Markdown(f"# Hofstede Model Output Evaluator (Part {part_id})")
        
        with gr.Row():
            domain_display = gr.Textbox(label="Domain", interactive=False)
            dimension_display = gr.Textbox(label="Dimension", interactive=False, scale=2)
        
        question_display = gr.Textbox(label="Question", interactive=False, lines=2)
        raw_output_display = gr.Code(label="Raw Model Output (Read-Only)", language="json", interactive=False, visible=False)
        statements_title = gr.Markdown("### Model Output (Editable)")

        all_statements, reset_buttons = [], []
        for i in range(5):
            with gr.Row():
                statement_box = gr.Textbox(label=f"Level {i+1} Statement", interactive=True, scale=20, lines=1)
                reset_button = gr.Button("🔄", scale=1, min_width=10, variant="secondary")
            all_statements.append(statement_box)
            reset_buttons.append(reset_button)
        
        with gr.Row():
            accept_btn = gr.Button("Accept (Good)", variant="primary")
            bad_btn = gr.Button("Save Changes (Bad)", variant="stop")
            fix_error_btn = gr.Button("Save Manual Fix (Format Error)", variant="secondary")

        progress_bar = gr.Label(label="Overall Progress")
        
        for i in range(5):
            reset_buttons[i].click(
                fn=partial(get_original_statement, statement_index=i),
                inputs=[current_item_data],
                outputs=[all_statements[i]]
            )
        
        main_buttons = [accept_btn, bad_btn, fix_error_btn]
        outputs = ([domain_display, dimension_display, question_display, raw_output_display, statements_title] + 
                   all_statements + main_buttons + [progress_bar, current_item_data])
        
        main_inputs = [current_item_data] + all_statements

        accept_btn.click(fn=partial(save_result, "good"), inputs=main_inputs, outputs=outputs)
        bad_btn.click(fn=partial(save_result, "bad"), inputs=main_inputs, outputs=outputs)
        fix_error_btn.click(fn=partial(save_result, "format_error_corrected"), inputs=main_inputs, outputs=outputs)

        app.load(load_next_item, outputs=outputs)
    return app

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Hofstede evaluation GUI for a specific data part.")
    parser.add_argument("part_id", type=int, choices=range(1, 6), help="The ID of the data part to evaluate (1-5)")
    args = parser.parse_args()
    part_id = args.part_id

    os.makedirs(INPUT_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    INPUT_FILE = os.path.join(INPUT_FOLDER, f"part_{part_id}.json")
    OUTPUT_FILE = os.path.join(OUTPUT_FOLDER, f"evaluated_data_part_{part_id}.jsonl")

    print(f"--- Starting Evaluation for Part {part_id} ---")
    print(f"Input folder:  {INPUT_FOLDER}")
    print(f"Output folder: {OUTPUT_FOLDER}")
    
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            ALL_DATA = json.load(f)
    except FileNotFoundError:
        print(f"\nERROR: Input file not found: '{INPUT_FILE}'")
        print("Please make sure the file exists in the 'inputs' directory.")
        exit(1)
    except json.JSONDecodeError:
        print(f"\nERROR: Could not parse '{INPUT_FILE}'. Please ensure it is valid JSON.")
        exit(1)
        
    load_progress()
    app = create_app(part_id)
    app.launch()
