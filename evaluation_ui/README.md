# Hofstede Model Output Evaluator

This tool provides a simple web interface for evaluating, correcting, and saving model outputs related to the Hofstede cultural dimensions model. It is designed to process data in parts, saving progress automatically.

## Setup and Installation

### 1. Prerequisites
- Python 3.7 or higher

### 2. Install Dependencies
You only need to install the `gradio` library. Open your terminal and run:
```bash
pip install gradio
```

### 3. Directory Structure
Before running the script, your project folder must be organized as follows. You will need to create the `inputs` directory manually.

```
your_project_folder/
├── main.py      <-- The script itself
└── inputs/
    ├── part_1.json   <-- Your input data file
    ├── part_2.json
    └── ...
```
- The `outputs/` directory will be created automatically by the script when you save your first item.

## How to Run the Application

1.  Open your terminal or command prompt.
2.  Navigate to your project folder.
3.  Run the script, providing the `part_id` of the data file you want to evaluate as a command-line argument. The `part_id` must be a number from 1 to 5.

    ```bash
    python main.py <PART_ID>
    ```

    **Example:** To evaluate the `part_1.json` file, run:
    ```bash
    python main.py 1
    ```

4.  The script will start a local web server and print a URL, usually `http://127.0.0.1:7860`. Open this URL in your web browser to access the UI.

## Using the Interface

The interface is designed for a straightforward evaluation workflow:



1.  **Information Panel:** The top text boxes (`Domain`, `Dimension`, `Question`) show the context for the current item. These are for display only.

2.  **Statements Panel:**
    -   The five `Level X Statement` boxes contain the model-generated output. **You can edit the text in these boxes directly.**
    -   The **Reset Button (🔄)** next to each statement will restore its text to the original value from the input file.

3.  **Action Buttons:** After reviewing (and editing, if necessary), click one of the three buttons to save and move to the next item.

    -   `Accept (Good)`: Click this if the five statements are accurate and well-formed as they are. No changes are needed.

    -   `Save Changes (Bad)`: Click this if the statements were semantically incorrect or poorly phrased and **you have edited them to be correct**.

    -   `Save Manual Fix (Format Error)`: This button **only appears** if the original model output could not be parsed (e.g., it was not valid JSON). In this case, you must manually type in all five statements before clicking this button to save your work.

4.  **Progress Bar:** The label at the bottom shows your overall progress for the current data part.

## Key Features

-   **Automatic Progress Saving:** The application saves your work after every item is evaluated. You can safely stop the script (Ctrl+C in the terminal) and relaunch it later. It will automatically resume from the last completed item.
-   **Output File:** Your evaluated data is saved in the `outputs/` directory in a file named `evaluated_data_part_<PART_ID>.jsonl`. Each line in this file is a complete JSON object representing one evaluated item.
