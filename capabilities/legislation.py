from .helpers import ask_gpt

PROMPT = """

The user has a relevant copy-pasted text from legislation. You should preprocess it into Markdown, in a standardised format using the rules below.

* Start with the title and subsection etc. of the legislation, inferring from the URL if necessary.
* Standardise all sublevels with Markdown indentations following e.g. 1., (a), (i).
* WHENEVER you use a level like the above, it must be indented and on a newline.

Example:

# Income Tax Act 2007 s. 1(1)(a)
1. This has two sublevels:
    (a) This is the first sublevel.
    (b) This is the second sublevel.
        (i) This is the first sublevel of the second sublevel.

Content below. Return the standardised version.
"""


def parse_legislation(text: str) -> str:
    """Take a copy-pasted extract from legislation and return a standardised Markdown version.

    Args:
        text (str): Legislation text.

    Returns:
        str: Policy text.
    """
    yield from ask_gpt(
        prompt=PROMPT + text,
        model="gpt-3.5-turbo",
        stream=True,
    )
