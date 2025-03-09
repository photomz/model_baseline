from .provider import ProviderAdapter
import os
from dotenv import load_dotenv
import json
from openai import OpenAI
from openai.types.chat import ChatCompletionChunk, ChatCompletion
from typing import Iterator
from datetime import datetime
from src.schemas import (
    ARCTaskOutput,
    AttemptMetadata,
    Choice,
    Message,
    Usage,
    Cost,
    CompletionTokensDetails,
    Attempt,
)
from typing import Optional

load_dotenv()


def aggregate_stream(stream):
    # Return as response object as non-streaming OpenAI request, but with printing token deltas.
    full_content = ""
    response_id = None
    created = None
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    for chunk in stream:
        chunk = chunk  # type: ChatCompletionChunk
        if response_id is None:
            response_id = chunk.id
            created = chunk.created
            model = chunk.model

        choices = chunk.choices
        if choices:
            delta = choices[0].delta.content
            print(delta, flush=True, end="")
            if delta:
                full_content += delta

        # Handling token usage if available
        usage = usage or chunk.usage

    # Construct the final response to match non-streaming format
    return ChatCompletion(
        id=response_id,
        model=model,
        object="chat.completion",
        created=created,
        choices=[
            {
                "index": 0,
                "message": {"role": "assistant", "content": full_content},
                "finish_reason": "stop",
            }
        ],
        usage=usage,  # Token usage might not always be available
    )


class OpenAIAdapter(ProviderAdapter):
    def init_client(self):
        """
        Initialize the OpenAI client
        """
        if not os.environ.get("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY not found in environment variables")

        client = OpenAI()
        return client

    def make_prediction(
        self, prompt: str, task_id: Optional[str] = None, test_id: Optional[str] = None
    ) -> Attempt:
        """
        Make a prediction with the OpenAI model and return an Attempt object

        Args:
            prompt: The prompt to send to the model
            task_id: Optional task ID to include in metadata
            test_id: Optional test ID to include in metadata
        """
        start_time = datetime.utcnow()

        messages = [{"role": "user", "content": prompt}]
        response = self.chat_completion(messages)

        end_time = datetime.utcnow()

        # Use pricing from model config
        input_cost_per_token = (
            self.model_config.pricing.input / 1_000_000
        )  # Convert from per 1M tokens
        output_cost_per_token = (
            self.model_config.pricing.output / 1_000_000
        )  # Convert from per 1M tokens

        prompt_cost = response.usage.prompt_tokens * input_cost_per_token
        completion_cost = response.usage.completion_tokens * output_cost_per_token

        # Convert input messages to choices
        input_choices = [Choice(index=0, message=Message(role="user", content=prompt))]

        # Convert OpenAI response to our schema
        response_choices = [
            Choice(
                index=1,
                message=Message(
                    role=response.choices[0].message.role,
                    content=response.choices[0].message.content,
                ),
            )
        ]

        # Combine input and response choices
        all_choices = input_choices + response_choices

        # Create metadata
        metadata = AttemptMetadata(
            model=self.model_config.model_name,
            provider=self.model_config.provider,
            start_timestamp=start_time,
            end_timestamp=end_time,
            choices=all_choices,
            kwargs=self.model_config.kwargs,
            usage=Usage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
                completion_tokens_details=CompletionTokensDetails(
                    reasoning_tokens=0,
                    accepted_prediction_tokens=response.usage.completion_tokens,
                    rejected_prediction_tokens=0,
                ),
            ),
            cost=Cost(
                prompt_cost=prompt_cost,
                completion_cost=completion_cost,
                total_cost=prompt_cost + completion_cost,
            ),
            task_id=task_id,
            test_id=test_id,
        )

        attempt = Attempt(
            metadata=metadata, answer=response.choices[0].message.content.strip()
        )

        return attempt

    def chat_completion(self, messages: list) -> ChatCompletion:
        response = self.client.chat.completions.create(
            model=self.model_config.model_name,
            messages=messages,
            **self.model_config.kwargs,
        )

        if self.model_config.kwargs.get("stream", False):
            response = aggregate_stream(response)

        # Check if response has incomplete <think> tags
        content = response.choices[0].message.content
        if "<think>" in content and "</think>" not in content:
            # Create a new prompt with the incomplete response
            print("Incomplete <think> tags. Recall.")
            messages += [{"role": "assistant", "content": content}]
            return self.chat_completion(messages)

        return response

    def extract_json_from_response(self, input_response: str) -> list[list[int]] | None:
        prompt = f"""
You are a helpful assistant. Extract only the JSON array of arrays from the following response. 
Do not include any explanation, formatting, or additional text.
Return ONLY the valid JSON array of arrays with integers.

Response:
{input_response}

Example Input:
```grid shape 3x3
143
934
336
```

Example Output:
[[1, 4, 3], [9, 3, 4], [3, 3, 6]]

IMPORTANT: Return ONLY the array, with no additional text, quotes, or formatting.
"""
        completion = self.chat_completion(
            messages=[{"role": "user", "content": prompt}],
        )

        assistant_content = completion.choices[0].message.content.strip()

        # Try to extract JSON from various formats
        # Remove markdown code blocks if present
        if "```" in assistant_content:
            # Extract content between code blocks
            code_blocks = assistant_content.split("```")
            for block in code_blocks:
                if block.strip() and not block.strip().startswith("json"):
                    assistant_content = block.strip()
                    break

        # Remove any leading/trailing text that's not part of the JSON
        assistant_content = assistant_content.strip()

        # Try to find array start/end if there's surrounding text
        if assistant_content and not assistant_content.startswith("["):
            start_idx = assistant_content.find("[[")
            if start_idx >= 0:
                end_idx = assistant_content.rfind("]]") + 2
                if end_idx > start_idx:
                    assistant_content = assistant_content[start_idx:end_idx]

        try:
            # Try direct parsing first
            json_result = json.loads(assistant_content)
            if isinstance(json_result, list) and all(
                isinstance(item, list) for item in json_result
            ):
                return json_result

            # If we got a dict with a response key, use that
            if isinstance(json_result, dict) and "response" in json_result:
                return json_result.get("response")

            return None
        except json.JSONDecodeError:
            # If direct parsing fails, try to find and extract just the array part
            try:
                # Look for array pattern and extract it
                import re

                array_pattern = r"\[\s*\[\s*\d+(?:\s*,\s*\d+)*\s*\](?:\s*,\s*\[\s*\d+(?:\s*,\s*\d+)*\s*\])*\s*\]"
                match = re.search(array_pattern, assistant_content)
                if match:
                    return json.loads(match.group(0))
            except:
                pass

            return None
