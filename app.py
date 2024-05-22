import json
import re

import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup

st.set_page_config(page_title="llm_api_cost_calc", page_icon="💰", layout="wide")

@st.cache_data(ttl=60 * 10)
def get_llm_api_cost():
    pattern = r"var p=(.*?),x"
    headers = {
        "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
        "accept-language": "en-US,en;q=0.9",
        "cache-control": "no-cache",
        "pragma": "no-cache",
        "user-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    }

    response = requests.get(
        "https://docsbot.ai/tools/gpt-openai-api-pricing-calculator", headers=headers
    )
    soup = BeautifulSoup(response.text)
    links = [i.get("src", "") for i in soup.find_all("script")]
    link = f"https://docsbot.ai{[i for i in links if 'gpt-openai-api-pricing-calculator' in i][0]}"
    response = requests.get(link, headers=headers)
    # Correct the JSON format
    corrected_text = re.sub(
        r"([{,])\s*(\w+)\s*:", r'\1 "\2":', re.search(pattern, response.text).group(1)
    )  # Enclose keys in double quotes
    corrected_text = corrected_text.replace(
        "'", '"'
    )  # Ensure all quotes are double quotes
    corrected_text = re.sub(
        r",\s*}", "}", corrected_text
    )  # Remove trailing commas before closing braces
    corrected_text = re.sub(
        r",\s*\]", "]", corrected_text
    )  # Remove trailing commas before closing brackets

    # Fix decimal number formats (prepend zero to decimal numbers without leading zero)
    corrected_text = re.sub(r":\s*\.([0-9]+)", r": 0.\1", corrected_text)

    # Parse the corrected JSON text
    data = json.loads(corrected_text)

    with open("cost.json", "w") as f:
        json.dump(data, f, indent=4)


# Load the JSON data
get_llm_api_cost()
with open("cost.json") as f:
    data = json.load(f)["Chat/Completion Models"]

# Streamlit app
st.header("LLM API Pricing Calculator")

# Convert results to DataFrame and sort by total cost
df = pd.DataFrame(data)
df.provider.replace("OpenAI / Azure", "OpenAI", inplace=True)

# Extract unique providers for the sidebar filters
providers = df.provider.unique()

# Default values
default_providers = ["OpenAI"]

# User inputs in the sidebar
with st.sidebar:
    input_tokens = st.number_input("Input Tokens", value=1000)
    output_tokens = st.number_input("Output Tokens", value=100)
    api_calls = st.number_input("API Calls", value=100)

    selected_providers = st.multiselect(
        "Select Providers", options=providers, default=default_providers
    )


# Calculate the total cost
df["Total"] = (
    (input_tokens / 1000) * df["input_token_cost_per_thousand"]
    + (output_tokens / 1000) * df["output_token_cost_per_thousand"]
) * api_calls

df = df[df.provider.isin(selected_providers)]

df = df.sort_values(by="Total")

# Format total cost as currency for display
df["Total"] = df["Total"].apply(lambda x: f"${x:.2f}")

# Display table
st.dataframe(df, use_container_width=True, hide_index=True)
