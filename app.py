import streamlit as st
import openai
import os
import yaml

from capabilities import (
    create_parameters,
    create_variables,
    model_policy,
    parse_legislation,
)


# Point OpenAI to the API key
openai.api_key = os.environ["OPENAI_API_KEY"]


st.title("PolicyEngine AI")
st.write(
    "This is a demo of the PolicyEngine AI. Each of the tabs below uses AI to allow you to perform a different task."
)

(
    create_parameters_tab,
    create_variables_tab,
    model_policy_tab,
    parse_legislation_tab,
) = st.tabs(
    [
        "Create parameters",
        "Create variables",
        "Model policy",
        "Parse legislation",
    ]
)

with create_parameters_tab:
    st.write(
        "This tab allows you to create a parameter for a policy. The AI will generate a parameter based on the information you provide."
    )

    information = st.text_area(
        "Information", "The UK personal tax allowance. was 12.5k in 2020."
    )
    submit = st.button("Create parameters")
    if submit:
        placeholder = st.empty()
        for result in create_parameters(information):
            placeholder.write(result)

with create_variables_tab:
    st.write(
        "This tab allows you to create a variable for a policy. The AI will generate a variable based on the information you provide."
    )

    information = st.text_area(
        "Information",
        "The UK personal tax allowance. was 12.5k in 2020. The UK personal tax allowance. was 12.5k in 2020.",
    )
    submit = st.button("Create variables")
    if submit:
        placeholder = st.empty()
        for result in create_variables(information):
            placeholder.write(result)

with model_policy_tab:
    st.write(
        "This tab allows you to model a policy. The AI will generate a policy based on the information you provide."
    )

    information = st.text_area(
        "Information",
        "A personal tax credit that phases in with income at 30%, up to a maximum of 1k, and then out at 10%, down to a minimum of 0.",
    )
    submit = st.button("Model policy")
    if submit:
        placeholder = st.empty()
        for result in model_policy(information):
            placeholder.write(result)

with parse_legislation_tab:
    st.write(
        "This tab allows you to parse legislation. The AI will generate a markdown version of the legislation based on the information you provide."
    )

    information = st.text_area(
        "Information",
        "Some info here",
    )
    submit = st.button("Parse legislation")
    if submit:
        placeholder = st.empty()
        for result in parse_legislation(information):
            placeholder.text(result)
