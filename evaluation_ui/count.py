import json
import logging
import sys
from pathlib import Path
from collections import defaultdict

def analyze_and_print_report(data: list):
    """
    Analyzes filtered data to count domains per dimension and prints a report.

    Args:
        data: A list of dictionary items, each expected to have
              'dimension_code' and 'domain' keys.
    """
    # Use defaultdict for easier counting.
    # The structure will be: {'dimension_code': {'domain': count}}
    dimension_domain_counts = defaultdict(lambda: defaultdict(int))

    for item in data:
        dimension = item.get("dimension_code")
        domain = item.get("domain")

        if dimension and domain:
            dimension_domain_counts[dimension][domain] += 1
        else:
            # Log a warning if essential keys are missing from an item
            item_id = item.get('index', 'N/A')
            logging.warning(
                f"Skipping item (index: {item_id}) due to missing 'dimension_code' or 'domain'."
            )

    logging.info("\n--- Domain Count per Dimension ---")
    if not dimension_domain_counts:
        logging.info("No data to report.")
        return

    # Sort dimensions alphabetically for consistent output
    for dimension, domain_counts in sorted(dimension_domain_counts.items()):
        logging.info(f"\nDimension: {dimension}")
        total_for_dimension = 0
        
        # Sort domains alphabetically for consistent output
        for domain, count in sorted(domain_counts.items()):
            logging.info(f"  - {domain}: {count}")
            total_for_dimension += count
        
        logging.info(f"  --------------------")
        logging.info(f"  Total for dimension: {total_for_dimension}")
    logging.info("\n----------------------------------")


def main():
    # Setup logging to print to the console
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s", stream=sys.stdout)

    # Parse command-line arguments
    import argparse
    parser = argparse.ArgumentParser(
        description="Filter JSON data and report on domain counts per dimension.")
    parser.add_argument(
        "input", type=Path,
        help="Path to the input JSON file containing an array.")
    args = parser.parse_args()

    # Load JSON data from the file
    try:
        text = args.input.read_text(encoding="utf-8")
        data = json.loads(text)
    except Exception as e:
        logging.error(f"Failed to read or parse JSON file: {e}")
        sys.exit(1)

    if not isinstance(data, list):
        logging.error(f"Expected a JSON array at the top level, but got {type(data).__name__}.")
        sys.exit(1)

    original_count = len(data)
    logging.info(f"Loaded {original_count} items from {args.input}.")

    # Filter out items with the domain "Family" or "Lifestyle"
    domains_to_remove = {"Family", "Lifestyle"}
    filtered_data = [
        item for item in data if item.get("domain") not in domains_to_remove
    ]

    filtered_count = len(filtered_data)
    logging.info(f"Removed {original_count - filtered_count} items with domain 'Family' or 'Lifestyle'.")
    logging.info(f"Remaining items for analysis: {filtered_count}")

    if not filtered_data:
        logging.info("No data remains after filtering.")
        sys.exit(0)

    # Analyze the filtered data and print the summary report
    analyze_and_print_report(filtered_data)

if __name__ == "__main__":
    main()
