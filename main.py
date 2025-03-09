import json
from src.adapters import (
    ProviderAdapter,
    AnthropicAdapter,
    OpenAIAdapter,
    DeepseekAdapter,
    GeminiAdapter,
    HuggingFaceFireworksAdapter,
    FireworksAdapter,
    GroqAdapter,
)
from dotenv import load_dotenv
import src.utils as utils
from src.schemas import ARCTaskOutput, ARCPair, Attempt
from src.prompts.prompt_manager import convert_task_pairs_to_prompt
from typing import List, Any, Optional
import os
import argparse
import re

load_dotenv()


class ARCTester:
    def __init__(
        self,
        config: str,
        save_submission_dir: str,
        overwrite_submission: bool,
        print_submission: bool,
        num_attempts: int,
        retry_attempts: int,
        print_logs: bool,
    ):
        self.config = config
        self.model_config = utils.read_models_config(config)
        self.provider = self.init_provider(self.model_config.provider)
        self.save_submission_dir = save_submission_dir
        self.overwrite_submission = overwrite_submission
        self.print_submission = print_submission
        self.num_attempts = num_attempts
        self.retry_attempts = retry_attempts
        self.print_logs = print_logs

    def init_provider(self, provider_name: str) -> ProviderAdapter:
        if provider_name == "anthropic":
            return AnthropicAdapter(self.config)
        elif provider_name == "openai":
            return OpenAIAdapter(self.config)
        elif provider_name == "deepseek":
            return DeepseekAdapter(self.config)
        elif provider_name == "groq":
            return GroqAdapter(self.config)
        elif provider_name == "gemini":
            return GeminiAdapter(self.config)
        elif provider_name == "huggingfacefireworks":
            return HuggingFaceFireworksAdapter(self.config)
        elif provider_name == "fireworks":
            return FireworksAdapter(self.config)
        else:
            raise ValueError(f"Unsupported provider: {provider_name}")

    def print_log(self, message: str):
        if self.print_logs:
            print(message)

    def convert_single_integer_to_2d_list(self, data: str) -> Optional[List[List[int]]]:
        """
        If the input string represents a single integer, return it as a nested list.
        Otherwise, return None.
        """
        try:
            parsed_data = int(data)
            result = [[parsed_data]]
            return result
        except ValueError:
            pass
        return None

    def convert_1d_list_to_2d_list(self, data: str) -> Optional[List[List[int]]]:
        """
        If the input string represents a single-item list containing one or more integers,
        return it as a nested list. Otherwise, return None.
        """
        try:
            # Remove whitespace and parse the string as JSON
            parsed_data = json.loads(data.strip())
            if (
                isinstance(parsed_data, list)
                and 1 <= len(parsed_data) <= 30
                and all(isinstance(item, int) for item in parsed_data)
            ):
                result = [[item] for item in parsed_data]
                return result
        except json.JSONDecodeError:
            pass
        return None

    def extract_json_from_response(self, response: str) -> List[List[int]]:
        """
        Extract JSON from various possible formats in the response.
        """
        # 1. Try to extract JSON from code block (most precise method)
        json_code_block_match = utils.extract_json_from_code_block(response)
        if json_code_block_match:
            return json_code_block_match

        # 2. Try to extract JSON grid from end of response (specialized for grid formats)
        json_grid_match = utils.extract_json_grid_from_end(response)
        if json_grid_match:
            return json_grid_match

        # 3. Try to extract JSON array using regex (more general approach)
        json_str_match = utils.regex_extract_json(response)
        if json_str_match:
            return json_str_match

        # 4. Finally, use an LLM to extract the JSON (last resort)
        json_llm_match = self.provider.extract_json_from_response(response)
        if json_llm_match:
            return json_llm_match

        # If all extraction methods fail, raise an exception
        raise json.JSONDecodeError(
            "Failed to extract valid JSON from the response", response, 0
        )

    @staticmethod
    def extract_grid_from_response(response: str) -> Optional[List[List[int]]]:
        """
        Extract a grid from a response that contains a grid format like:
        ```grid shape 11x15
        444444444444444
        444444442222444
        ...
        ```

        Returns a 2D list of integers if successful, None otherwise.
        """
        # Match grid shape header and grid content
        grid_pattern = r"```grid shape (\d+)x(\d+)\n([\d\s]+?)```"
        match = re.search(grid_pattern, response, re.DOTALL)

        if not match:
            return None

        n_rows, n_cols = int(match.group(1)), int(match.group(2))
        grid_text = match.group(3).strip()

        # Split into lines and convert to 2D array
        lines = grid_text.strip().split("\n")

        # Validate number of rows
        if len(lines) != n_rows:
            print(f"Grid row count mismatch: expected {n_rows}, got {len(lines)}")
            return None

        # Convert to numpy array
        grid = []
        for line in lines:
            line = line.strip()
            if len(line) != n_cols:
                print(f"Grid column count mismatch: expected {n_cols}, got {len(line)}")
                return None

            try:
                row = [int(char) for char in line]
                grid.append(row)
            except ValueError:
                print(f"Failed to convert grid characters to integers")
                return None

        return grid

    def parse_and_validate_json(self, response: str) -> ARCTaskOutput:
        """
        Parse the response string into JSON and validate its structure.

        This is unfortunately a necessary evil.
        """
        # Try to extract grid format first
        grid_match = self.extract_grid_from_response(response)
        if grid_match:
            return grid_match

        single_integer_match = self.convert_single_integer_to_2d_list(response)
        if single_integer_match:
            print(f"Extracted single integer: {single_integer_match}")
            return single_integer_match

        # Try to convert 1d list to 2d list
        # This is a band-aid hack when the LLM returns a single-item list containing an integer
        one_d_match = self.convert_1d_list_to_2d_list(response)
        if one_d_match:
            print(f"Extracted 1d list: {one_d_match}")
            return one_d_match

        # First, try to parse the raw JSON response
        try:
            parsed_json = json.loads(response)
            print(f"Extracted raw JSON: {parsed_json}")
            return parsed_json
        except:
            # If raw parsing fails, try to extract JSON from various formats
            parsed_json = self.extract_json_from_response(response)

        # Validate the structure of the parsed JSON
        if not isinstance(parsed_json, list) or not all(
            isinstance(row, list) for row in parsed_json
        ):
            raise ValueError("Invalid JSON structure: expected a list of lists")

        return parsed_json

    def predict_task_output(
        self,
        training_pairs: List[ARCPair],
        test_input: ARCPair,
        task_id: str,
        test_id: str,
    ):
        """
        Given a task, predict the test output. This reponse may need parsing.

        Convert the training pairs and test pairs into a prompt
        Give the prompt to the LLM
        return the response
        """

        # Convert the training pairs and test pairs into a prompt
        prompt = convert_task_pairs_to_prompt(training_pairs, test_input)

        self.print_log(f"Making prediction for task {task_id}, test {test_id}")
        response: Attempt = self.provider.make_prediction(
            prompt, task_id=task_id, test_id=test_id
        )

        return response

    def get_task_prediction(
        self,
        training_pairs: List[ARCPair],
        test_input: ARCPair,
        task_id: str,
        test_id: str,
    ) -> Attempt:
        """
        Modified to return the full Attempt object instead of just the parsed answer
        """
        # Get the initial response as an Attempt object
        attempt: Attempt = self.predict_task_output(
            training_pairs, test_input, task_id, test_id
        )

        try:
            # Parse the answer field but keep the full attempt object
            # Always use the last choice in the array (which should be the assistant's response)
            if attempt.metadata.choices:
                last_choice = attempt.metadata.choices[-1]
                if last_choice.message.content:
                    parsed_answer = self.parse_and_validate_json(
                        last_choice.message.content
                    )
                else:
                    raise ValueError("Assistant response is empty")
            else:
                raise ValueError("No choices found in response")

            # Update the answer in the original attempt - now accepts List[List[int]]
            attempt.answer = parsed_answer
            return attempt
        except json.JSONDecodeError as e:
            self.print_log(f"JSON parsing failed: {e}")
            raise

    def generate_task_solution(self, data_dir, task_id):
        """
        data_dir: str, the directory of the data set to run
        task_id: str, the specific task to run. If None, run all tasks.
        num_attempts: int the number of times to attempt a prediction. The official competition has 2 attempts.
        retry_attempts: int the number of times to retry a prediction if it fails
        save_submission: bool, whether to save the submission to a file after each task
        """

        self.print_log(f"Running task {task_id}")
        utils.validate_data(data_dir, task_id)

        # Use the config name as the test_id
        test_id = self.config

        self.print_log(f"Using test_id: {test_id}")

        # Logic for overwrite. If save_submission_dir is provided, check if the submission already exists
        if (
            self.save_submission_dir
            and utils.submission_exists(self.save_submission_dir, task_id)
            and not self.overwrite_submission
        ):
            self.print_log(f"Submission for task {task_id} already exists, skipping")
            return

        task_attempts = []

        train_pairs = utils.get_train_pairs_from_task(data_dir, task_id)
        test_input = utils.get_test_input_from_task(data_dir, task_id)

        # Go through each test pair to get a prediction. 96% of challenges have 1 pair.
        for t, pair in enumerate(test_input):
            current_test_id = str(t)
            self.print_log(f"Starting task {task_id}, Pair #{t+1}")
            pair_attempts = {}

            # Run through each prediction attempt
            for attempt in range(1, self.num_attempts + 1):
                attempt_key = f"attempt_{attempt}"
                pair_attempts[attempt_key] = None

                for retry in range(self.retry_attempts):
                    try:
                        self.print_log(
                            f"    Predicting attempt #{attempt}, retry #{retry + 1}"
                        )
                        # Now storing the full attempt object with task_id and test_id
                        attempt_obj = self.get_task_prediction(
                            training_pairs=train_pairs,
                            test_input=pair,
                            task_id=task_id,
                            test_id=test_id,
                        )

                        if attempt_obj is not None:
                            self.print_log(f"    Prediction: {attempt_obj.answer}")
                            pair_attempts[attempt_key] = attempt_obj.model_dump(
                                mode="json"
                            )
                            break
                    except Exception as e:
                        self.print_log(f"Retrying: {e}")

                    if retry == self.retry_attempts - 1:
                        self.print_log(f"    All retries failed for attempt #{attempt}")

            # Only append non-None attempts
            if any(v is not None for v in pair_attempts.values()):
                task_attempts.append(pair_attempts)

        if task_attempts:
            if self.print_submission:
                self.print_log(f"Submission for task {task_id}:\n{task_attempts}")

            if self.save_submission_dir:
                utils.save_submission(self.save_submission_dir, task_id, task_attempts)
        else:
            self.print_log(
                f"No valid predictions for task {task_id}, skipping submission"
            )

        return task_attempts if task_attempts else None


def test_grid_extraction():
    """
    Test function for the grid extraction functionality.
    """
    import re

    test_cases = [
        # Valid grid
        """Some text before
        ```grid shape 3x4
        1234
        5678
        9012
        ```
        Some text after""",
        # Invalid grid (row count mismatch)
        """4. **Construct the Output:** The resulting grid will have '2's in the central cross and surrounding diamond, replacing the original '1's and '4's.

**Answer:**

```grid shape 11x15
444444444444444
444444442222444
414214442224444
414414442212444
414214442222444
412114444444444
444444422222444
444444422222444
444444422222444
444444422212444
444444444444444
```None""",
    ]

    for i, test_case in enumerate(test_cases):
        print(f"Test case {i+1}:")
        print(test_case)
        print("Result:", ARCTester.extract_grid_from_response(test_case))
        print()


if __name__ == "__main__":
    # Add this to run the test function
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--test-grid":
        test_grid_extraction()
        sys.exit(0)

    parser = argparse.ArgumentParser(description="Run ARC Tester")
    parser.add_argument(
        "--data_dir", type=str, help="Data set to run. Configure in config/config.json"
    )
    parser.add_argument("--task_id", type=str, help="Specific task ID to run")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Configuration name (e.g., 'o1_high', 'gemini_short_response')",
    )
    parser.add_argument(
        "--save_submission_dir",
        type=str,
        metavar="FOLDER_NAME",
        help="Folder name to save the submissions under Ex: 'submissions/o1_high'",
    )
    parser.add_argument(
        "--overwrite_submission",
        action="store_true",
        help="Overwrite the submission if it already exists",
    )
    parser.add_argument(
        "--print_submission",
        action="store_true",
        help="Print the submission to the console after each task",
    )
    parser.add_argument(
        "--task_set",
        type=str,
        default="public_eval",
        choices=["public_eval", "public_training"],
        help="Task set to run",
    )
    parser.add_argument(
        "--num_attempts",
        type=int,
        default=2,
        help="Number of attempts for each prediction",
    )
    parser.add_argument(
        "--retry_attempts",
        type=int,
        default=2,
        help="Number of retry attempts for failed predictions",
    )
    parser.add_argument(
        "--print_logs",
        action="store_true",
        help="Disable printing logs to console (default: False)",
    )
    args = parser.parse_args()

    arc_solver = ARCTester(
        config=args.config,
        save_submission_dir=args.save_submission_dir,
        overwrite_submission=args.overwrite_submission,
        print_submission=args.print_submission,
        num_attempts=args.num_attempts,
        retry_attempts=args.retry_attempts,
        print_logs=args.print_logs,
    )

    arc_solver.generate_task_solution(data_dir=args.data_dir, task_id=args.task_id)
