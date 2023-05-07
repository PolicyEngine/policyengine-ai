from .helpers import ask_gpt_stream

PROMPT = """

The user has a relevant copy-pasted text from a source. You should preprocess it into small chunks (30-50 words) of information

* Start each section with a YAML metadata entry with the title and URL of the source if applicable.
* Standardise all all formatting so it's easily readable, with newlines and indentation.
* Ensure ALL information is preserved. Do not alter which content lines up with which 1. (a) (i) etc.

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


def parse_legislation(text: str) -> str:
    """Take a copy-pasted extract from legislation and return a standardised Markdown version.

    Args:
        text (str): Legislation text.

    Returns:
        str: Policy text.
    """
    yield from ask_gpt_stream(
        prompt=PROMPT + text,
        model="gpt-3.5-turbo",
    )
