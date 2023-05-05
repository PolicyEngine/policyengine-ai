from .helpers import ask_gpt

MODEL = "gpt-4"

PROMPT = f"""

PolicyEngine (derived from OpenFisca) uses Python to model the logic of a policy, in the form below (an example):

income_tax.py
```python
from policyengine_uk.model_api import * # for uk-specific functions, we also model in policyengine_us.


class income_tax(Variable):
    label = "income tax"
    definition_period = YEAR # or MONTH
    entity = Household # or Person or TaxUnit (US) or BenUnit (UK)
    value_type = float # or int or bool or str
    reference = "https://www.gov.uk/income-tax-rates" # a reference for the parameter, if the user gave you one.
    defined_for = None # e.g. StateCode.NY if in the US and state-specific, otherwise None.
    documentation = "Federal income tax liabilities."

    def formula(person, period, parameters): # person might be household if entity is Household, etc.
        employment_income = person("employment_income", period) # You can retrieve other variables like this. You can also use `add(household, period, [list, of, variables])` to add up variables from a higher entity.
        parameter_subtree = parameters(period).gov.income_tax.income_tax_allowance # You can retrieve parameters like this, they're in a folder tree
        tax_rate = parameter_subtree.rate # You can retrieve values from the parameter tree like this if they exist.
        # variable values are all vectors, so use vectorisable operations.
        # min_(x, y) is a vectorisable function that returns the minimum of x and y for each element of the vector. We also have max_, and you can use other NumPy default functions too.
        return tax_rate * max_(employment_income - parameter_subtree.allowance, 0)
```

Add comments explaining the logic. The user will provide information after these prompts- write up all the variable files needed to accurately model the policy.

PolicyEngine uses standardised YAML files to model policy parameters, in the form below:
personal_allowance.yaml
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
        value: 11_850.00 # the value of the parameter on this date. Make sure to use underscores to separate thousands.
        reference: # Any references which inform about *only* this value.
        - title: ...
          href: ...
    2019-04-01: 12_500 # if there's no reference, you can use this shorthand.
    2020-04-01: ...
```
If you don't have enough information to be confident, return what you have but add a YAML comment explaining where you think you messed up. Return valid YAML only. 
If the user passes you data describing more than one parameter at a time, write parameter files for each one, with the filenames written above.


Below is the user's information. Write up all parameter and variable files needed to accurately model the policy. Before each Python or YAML snippet, write the filename (and folder location).
e.g. parameters/gov/income_tax/personal_allowance.yaml or variables/income_tax/income_tax.py
Do not give commentary between files. Do not deviate from the instructions given in the YAML and Python information above. Don't give references if the user didn't give you any.
"""


def model_policy(information: str) -> str:
    """Write a PolicyEngine parameter YAML file based on the information provided.

    Args:
        information (str): The information to use to create the parameter.

    Returns:
        str: The parameter YAML file.
    """

    prompt = PROMPT + information

    # Use the chat endpoint to generate the parameter.
    yield from ask_gpt(prompt, model=MODEL, stream=True)
