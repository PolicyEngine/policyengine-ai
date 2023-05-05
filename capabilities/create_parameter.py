import openai
import os


openai.api_key = os.environ["OPENAI_API_KEY"]

MODEL = "gpt-4"

PROMPT = f"""

PolicyEngine uses standardised YAML files to model policy parameters, in the form below:
```yaml
description: A full-sentence description of the parameter.
metadata:
    unit: either currency-GBP, currency-USD, hour, or something new like 'person' if it's a population size.
    period: the time period over which the parameter applies, one of 'year', 'month' or 'day'.
    label: a short phrase that describes the parameter, e.g. 'personal tax allowance'. You should be able to drop this into a sentence as-is.
    reference: # a list of any references used in writing the parameter if the user gave you any, that apply to *all* values.
    - title: The title of the web page or legislative reference.
      href: The URL of the web page or legislative reference.
values: # any number of values, each with a start date, in order.
    2018-04-01:
        value: 11_850 # the value of the parameter on this date.
        reference: # Any references which inform about *only* this value.
        - title: ...
          href: ...
    2019-04-01: ...
```
If you don't have enough information to be confident, return what you have but add a YAML comment explaining where you think you messed up. Return valid YAML only. Below is the information.
"""


def create_parameter(information: str) -> str:
    """Write a PolicyEngine parameter YAML file based on the information provided.

    Args:
        information (str): The information to use to create the parameter.

    Returns:
        str: The parameter YAML file.
    """

    prompt = PROMPT + information

    # Use the chat endpoint to generate the parameter.
    response = openai.ChatCompletion.create(
        model=MODEL,
        messages=[
            dict(
                role="user",
                content=prompt,
            )
        ],
    )["choices"][0]["message"]["content"]

    return response
