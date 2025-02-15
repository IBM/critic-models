import json
import argparse
from datasets import load_dataset
import re
import multiprocessing
import os

def worker(code, test_list, return_dict):
    """
    Worker function to execute the code and run test cases.

    Args:
        code (str): The code to execute.
        test_list (list): List of test cases to execute.
        return_dict (dict): Shared dictionary to store results.
    """
    try:
        execution_context = {}
        exec(code, execution_context)  # Execute the provided code
        for test_case in test_list:
            exec(test_case, execution_context)  # Run each test
        return_dict["success"] = True
        return_dict["error"] = None
    except Exception as e:
        return_dict["success"] = False
        return_dict["error"] = str(e)


def exec_with_timeout(code, test_list, timeout=5):
    """
    Executes the given code and tests with a timeout using multiprocessing.

    Args:
        code (str): The code to execute.
        test_list (list): List of test cases to execute.
        timeout (int): Timeout in seconds.

    Returns:
        dict: A result dictionary with `success` (bool) and `error` (str) keys.
    """
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    process = multiprocessing.Process(target=worker, args=(code, test_list, return_dict))
    process.start()
    process.join(timeout)

    if process.is_alive():
        process.terminate()
        process.join()
        return {"success": False, "error": "Timeout exceeded"}

    return return_dict.copy()


def calculate_mbpp_accuracy(dataset, predictions):
    """
    Calculate accuracy of model predictions for MBPP dataset.

    Args:
        dataset (list of dict): List of problems with 'task_id', 'test_list', and 'code' keys.
        predictions (dict): Dictionary where keys are task_ids and values are predicted code strings.

    Returns:
        float: Accuracy of the predictions (percentage of problems where all tests passed).
    """
    total_problems = len(dataset)
    passed_count = 0

    for problem in dataset:
        task_id = problem['task_id']
        test_list = problem['test_list']
        task_text = problem['text']
        # Get the model's predicted code for the current task
        predicted_code = predictions.get(str(task_text))[-1]["content"]
        only_code = extract_code_from_output(predicted_code)
        if not only_code:
            print(f"Task {task_id}: No prediction found.")
            continue

        result = exec_with_timeout(only_code, test_list, timeout=5)
        if result["success"]:
            print(f"Task {task_id}: All tests passed.")
            passed_count += 1
        else:
            print(f"Task {task_id}: Failed with error: {result['error']}")

    accuracy = (passed_count / total_problems) * 100
    return accuracy


def main(predictions_dir):
    # Load the MBPP dataset
    dataset = load_dataset("google-research-datasets/mbpp", split="test")

    # Load predictions from the given JSON file
    results_json = {}
    for fname in os.listdir(predictions_dir):
        if ".json" not in fname:
            continue
        predictions_path = os.path.join(predictions_dir, fname)
        with open(predictions_path, "r") as f:
            predictions = json.load(f)
        if predictions.get("predictions_key") is not None:
            predictions = predictions[predictions.get("predictions_key")]

        # Calculate accuracy
        accuracy = calculate_mbpp_accuracy(dataset, predictions)
        print(f"Accuracy: {accuracy:.2f}%")
        results_json[fname] = accuracy

    for k in results_json:
        print(f"{k}: {results_json[k]}")



def extract_code_from_output(model_output):
    """
    Extract the relevant code block from the model output.

    Args:
        model_output (str): The complete output from the model.

    Returns:
        str: The extracted Python code, or an empty string if no code block is found.
    """
    # Use a regular expression to find the first code block between triple backticks
    match = re.search(r"```python\n(.*?)\n```", model_output, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate MBPP prediction accuracy.")
    parser.add_argument("--predictions_dir", type=str, help="Path to the predictions JSON file.")
    args = parser.parse_args()

    main(args.predictions_dir)
