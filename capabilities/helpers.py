import openai
import os
from typing import Iterable, Union
from sentence_transformers import SentenceTransformer, util
import numpy as np

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


class KnowledgeBase:
    def __init__(self):
        self.data = []
        self.embeddings = []
        self.embeddings_array = None
        self.model = SentenceTransformer("all-mpnet-base-v2")

    def partition_data(self, value: str) -> Iterable[str]:
        """Sometimes we need to partition the data into smaller chunks to keep results useful. We'll ask GPT-3.5-turbo
        to do this for us.

        Args:
            value (str): The data to partition.

        Returns:
            Iterable[str]: The partitioned data.
        """
        partitioned_data = ask_gpt(
            prompt=f"""
            User input:
            {value}
            Instructions:
            Separate the above user-supplied data into smaller chunks (50-100 words), separated with ---. Also make sure that each chunk has a title at the start describing what it is.
            """,
            model="gpt-3.5-turbo",
        )
        return partitioned_data.split("---")

    def add(self, value: str):
        values_to_add = []
        for partitioned_value in self.partition_data(value):
            values_to_add.append(partitioned_value)
        embeddings_to_add = self.model.encode(values_to_add)

        self.data.extend(values_to_add)
        self.embeddings.extend(embeddings_to_add)
        if self.embeddings_array is None:
            self.embeddings_array = np.array(self.embeddings)
        else:
            self.embeddings_array = np.vstack(
                (self.embeddings_array, np.array(embeddings_to_add))
            )

    def search(self, query: str, top_n: int = 1) -> Iterable[str]:
        query_embedding = self.model.encode(query)
        similarity = util.dot_score(query_embedding, self.embeddings_array)[0]
        top_n_idx = np.argsort(similarity)[-top_n:]
        for idx in top_n_idx:
            yield self.data[idx]
