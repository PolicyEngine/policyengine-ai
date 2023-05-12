from typing import Iterable

import os
import openai

openai.api_key = os.environ["OPENAI_API_KEY"]


def ask_gpt(prompt: str, model: str = "gpt-4") -> str:
    """Return the response to a prompt from the OpenAI API.

    Args:
        prompt (str): The prompt to send to the API.
        model (str, optional): The model to use. Defaults to "gpt-4".

    Returns:
        str: The response from the API.
    """
    return openai.ChatCompletion.create(
        model=model,
        messages=[
            dict(
                role="user",
                content=prompt,
            )
        ],
    )["choices"][0]["message"]["content"]


def ask_gpt_stream(prompt: str, model: str = "gpt-4") -> Iterable:
    """Return the response to a prompt from the OpenAI API, yielding the results as they come in.

    Args:
        prompt (str): The prompt to send to the API.
        model (str, optional): The model to use. Defaults to "gpt-4".

    Returns:
        Iterable: The response from the API.
    """
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            dict(
                role="user",
                content=prompt,
            )
        ],
        stream=True,
    )

    combined_response = ""

    for result in response:
        combined_response += (
            result["choices"][0].get("delta", {}).get("content", "")
        )
        yield combined_response