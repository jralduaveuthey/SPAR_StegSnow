import json
import os
from typing import Dict, Any, List

# Configuration
FILE_NAME = "experiment_logs_20240726_160740.json"  # Name of the input JSON file
INPUT_FILE = os.path.join(os.path.dirname(__file__),FILE_NAME)
FILTER_CRITERIA = {
    "completion_reason": "Error: LLM-E failed to generate valid JSON response"
}  # Add or modify filter criteria here

def load_json(file_path: str) -> Dict[str, Any]:
    """Load JSON data from a file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def save_json(data: Dict[str, Any], file_path: str) -> None:
    """Save JSON data to a file."""
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=2, ensure_ascii=False)

def filter_results(data: Dict[str, Any], filter_criteria: Dict[str, str]) -> List[Dict[str, Any]]:
    """Filter results based on the given criteria."""
    filtered_results = []
    for result in data.get('results', []):
        if all(result.get(key) == value for key, value in filter_criteria.items() if value):
            filtered_results.append(result)
    return filtered_results

def filter_logs(data: Dict[str, Any], filter_criteria: Dict[str, str]) -> List[Dict[str, Any]]:
    """Filter logs based on the given criteria."""
    filtered_logs = []
    for log in data.get('logs', []):
        if any(key in log for key in filter_criteria):
            if all(log.get(key) == value for key, value in filter_criteria.items() if key in log):
                filtered_logs.append(log)
    return filtered_logs

def create_output_path(input_file: str) -> str:
    """Create output folder and generate output file path."""
    input_dir, input_filename = os.path.split(input_file)
    folder_name = os.path.splitext(input_filename)[0]
    new_folder_path = os.path.join(input_dir, folder_name)
    os.makedirs(new_folder_path, exist_ok=True)
    output_filename = f"copy2debug_{input_filename}"
    output_path = os.path.join(new_folder_path, output_filename)
    return output_path

def main() -> None:
    """Main function to process the JSON file."""
    try:
        # Load the JSON data
        data = load_json(INPUT_FILE)

        # Filter the results
        filtered_results = filter_results(data, FILTER_CRITERIA)

        # Filter the logs if any filter criteria apply to log fields
        filtered_logs = filter_logs(data, FILTER_CRITERIA)

        # Create a new dictionary with filtered results
        filtered_data = {
            'setup': data.get('setup', {}),
            'results': filtered_results,
            'statistics': data.get('statistics', {})
        }

        # Only include logs if they were filtered
        if filtered_logs:
            filtered_data['logs'] = filtered_logs

        # Create output path
        output_file = create_output_path(INPUT_FILE)

        # Save the filtered data to a new JSON file
        save_json(filtered_data, output_file)
        print(f"Filtered data saved to {output_file}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()