from capabilities.helpers.llm import ask_gpt_stream

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

Add comments explaining the logic. The user will provide information below- write up all the variable files needed to accurately model the policy.
"""


def create_variables(information: str) -> str:
    """Write a PolicyEngine parameter YAML file based on the information provided.

    Args:
        information (str): The information to use to create the parameter.

    Returns:
        str: The parameter YAML file.
    """

    prompt = PROMPT + information

    # Use the chat endpoint to generate the parameter.
    yield from ask_gpt_stream(prompt, model=MODEL)
