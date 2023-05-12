from typing import Iterable
from capabilities.helpers.llm import ask_gpt
import re

NEXT_PATTERN = [r"\n\n§", r"\n\n\([a-h]\)", r"\n\n\(\d+\)", r"\n\n\([A-Z]\)"]

def section_header_split(text:str, pattern = r"\n\n§", splits = []) -> Iterable[str]:
    idx = NEXT_PATTERN.index(pattern)
    if len(text.split(' ')) < 5:
        return splits

    if len(text.split(' ')) < 500 or idx == len(NEXT_PATTERN) - 1:
        splits.append(text)

    else:
        for s in re.split(pattern, text):
            splits = section_header_split(s, pattern = NEXT_PATTERN[idx + 1], splits = splits)

    return splits


def llm_split(text:str) -> Iterable[str]:
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
        prompt=PROMPT + "\n\n" + text,
        model="gpt-3.5-turbo",
    )
    return partitioned_data.split("\n\n")
