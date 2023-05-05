import streamlit as st
import openai
import os
import yaml

from capabilities import create_parameter

# Point OpenAI to the API key
openai.api_key = os.environ["OPENAI_API_KEY"]


st.title("PolicyEngine AI")
st.write(
    "This is a demo of the PolicyEngine AI. Each of the tabs below uses AI to allow you to perform a different task."
)

(create_parameter_tab,) = st.tabs(["Create parameter"])

if create_parameter_tab:
    st.write(
        "This tab allows you to create a parameter for a policy. The AI will generate a parameter based on the information you provide."
    )

    information = st.text_input(
        "Information", "The UK personal tax allowance. was 12.5k in 2020."
    )
    parameter_file = create_parameter(information)
    st.write(parameter_file)
