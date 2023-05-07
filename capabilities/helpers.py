import openai
import os
from typing import Iterable, Union
from sentence_transformers import SentenceTransformer, util
import numpy as np
import pandas as pd

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

    def save(self, path: str):
        np.save(path, self.embeddings_array)
        pd.DataFrame({"data": self.data}).to_csv(
            path + ".csv.gz", index=False, compression="gzip"
        )

    def load(self, path: str):
        self.embeddings_array = np.load(path)
        self.data = pd.read_csv(path + ".csv.gz")["data"].tolist()

    def partition_data(self, value: str) -> Iterable[str]:
        """Sometimes we need to partition the data into smaller chunks to keep results useful. We'll ask GPT-3.5-turbo
        to do this for us.

        Args:
            value (str): The data to partition.

        Returns:
            Iterable[str]: The partitioned data.
        """
        PROMPT = """
The user has a relevant copy-pasted text from a source. You should preprocess it into small chunks (30-50 words) of information

* Start each section with a YAML metadata entry with the title and URL of the source if applicable.
* Standardise all all formatting so it's easily readable, with newlines and indentation.
* Ensure ALL information is preserved. Do not alter which content lines up with which 1. (a) (i) etc.
* Always include two newlines between sections.

Example:

```yaml
title: "Income Tax Act 2007 s. 1"
url: "https://www.legislation.gov.uk/ukpga/2007/3/part/2/section/1"
```
1. Income tax is charged for each tax year.
    (a) Income tax is charged on the total income of the tax year.
    ...
```yaml
title: "Income Tax Act 2007 s. 2-3" # Group sections together if they're in a similar context.
url: "https://www.legislation.gov.uk/ukpga/2007/3/part/2/section/2"
```
2. ...
```

Content below. Return the standardised version.
"""
        partitioned_data = ask_gpt(
            prompt=PROMPT + "\n\n" + value,
            model="gpt-3.5-turbo",
        )
        return partitioned_data.split("\n\n")

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
