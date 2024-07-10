import json
import re
from typing import Dict

import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from bs4 import BeautifulSoup

st.set_page_config(page_title="LLM API Cost Calculator", page_icon="💰", layout="wide")

# Constants
DATA_URL = "https://docsbot.ai/tools/gpt-openai-api-pricing-calculator"
CACHE_TTL = 60 * 60  # 60 minutes
DEFAULT_PROVIDERS = ["Anthropic", "OpenAI"]
DEFAULT_MODEL = "GPT-3.5 Turbo"
JSON_FILE_PATH = "cost.json"


def extract_and_correct_json(text: str) -> Dict:
    """Extract and correct JSON data from the script content."""
    pattern = r"var p=(.*?),x"
    json_str = re.search(pattern, text).group(1)

    # Correct JSON format
    json_str = re.sub(r"([{,])\s*(\w+)\s*:", r'\1 "\2":', json_str)
    json_str = json_str.replace("'", '"')
    json_str = re.sub(r",\s*}", "}", json_str)
    json_str = re.sub(r",\s*\]", "]", json_str)
    json_str = re.sub(r":\s*\.([0-9]+)", r": 0.\1", json_str)
    json_str = json_str.split(",A=")[0]

    return json.loads(json_str)


@st.cache_data(ttl=CACHE_TTL)
def fetch_llm_api_cost() -> Dict:
    """Fetch and parse LLM API cost data from the website."""
    headers = {
        "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
        "accept-language": "en-US,en;q=0.9",
        "cache-control": "no-cache",
        "pragma": "no-cache",
        "user-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    }

    try:
        response = requests.get(DATA_URL, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        script_links = [i.get("src", "") for i in soup.find_all("script")]
        target_script = f"https://docsbot.ai{[i for i in script_links if 'gpt-openai-api-pricing-calculator' in i][0]}"

        script_content = requests.get(target_script, headers=headers)
        script_content.raise_for_status()
        json_data = extract_and_correct_json(script_content.text)

        with open(JSON_FILE_PATH, "w") as f:
            json.dump(json_data, f, indent=4)

        return json_data["Chat/Completion Models"]
    except requests.RequestException as e:
        st.error(f"Error fetching data: {str(e)}")
        return {}


def load_data() -> pd.DataFrame:
    """Load and preprocess the LLM API cost data."""
    data = fetch_llm_api_cost()
    df = pd.DataFrame(data)
    df["provider"] = df["provider"].replace("OpenAI / Azure", "OpenAI")
    return df


def calculate_costs(
    df: pd.DataFrame,
    input_tokens: int,
    output_tokens: int,
    api_calls: int,
    default_model: str,
    show_token_costs: bool,
) -> pd.DataFrame:
    """Calculate total and relative costs for each model."""
    df["Total"] = (
        (input_tokens / 1000) * df["input_token_cost_per_thousand"]
        + (output_tokens / 1000) * df["output_token_cost_per_thousand"]
    ) * api_calls

    default_cost = df[df.model_name == default_model]["Total"].values[0]
    df["Relative Cost"] = df["Total"] / default_cost
    df["Relative Cost"] = df["Relative Cost"].apply(
        lambda x: f"{x:.2f} * {default_model}"
    )
    
    df = df.sort_values(by="Total")
    df["Total"] = df["Total"].apply(lambda x: f"${x:.2f}")

    if show_token_costs:
        df["Input Token Cost (per 1k)"] = df["input_token_cost_per_thousand"].apply(
            lambda x: f"${x:.4f}"
        )
        df["Output Token Cost (per 1k)"] = df["output_token_cost_per_thousand"].apply(
            lambda x: f"${x:.4f}"
        )
        columns = [
            "model_name",
            "provider",
            "Input Token Cost (per 1k)",
            "Output Token Cost (per 1k)",
            "Total",
            "Relative Cost",
        ]
    else:
        columns = ["model_name", "provider", "Total", "Relative Cost"]

    return df[columns]


def create_total_cost_chart(df: pd.DataFrame) -> px.bar:
    """Create a horizontal bar chart for total cost by model."""
    df_chart = df.copy()
    df_chart["Total"] = df_chart["Total"].str.replace("$", "").astype(float)
    fig = px.bar(
        df_chart,
        y="model_name",  # Swap x and y
        x="Total",  # Swap x and y
        color="provider",
        title="Total Cost by Model",
        orientation="h",  # Set orientation to horizontal
    )
    fig.update_layout(
        yaxis_title="Model",  # Swap x and y axis titles
        xaxis_title="Total Cost ($)",
        height=600,  # Increase height to accommodate all models
        yaxis={"categoryorder": "total descending"},  # Sort bars by total cost
    )
    return fig


def main():
    st.header("LLM API Pricing Calculator")

    df = load_data()
    providers = df.provider.unique()
    models = df.model_name.unique()

    with st.sidebar:
        st.subheader("Input Parameters")
        input_tokens = st.number_input("Input Tokens", value=10000, min_value=1)
        output_tokens = st.number_input("Output Tokens", value=1000, min_value=1)
        api_calls = st.number_input("API Calls", value=100, min_value=1)

        selected_providers = st.multiselect(
            "Select Providers", options=providers, default=DEFAULT_PROVIDERS
        )
        default_model = st.selectbox(
            "Select default model for relative cost comparison",
            options=models,
            index=models.tolist().index(DEFAULT_MODEL),
        )

        show_token_costs = st.toggle("Show input/output tokens cost", value=False)

    df_filtered = df[df.provider.isin(selected_providers)]
    df_costs = calculate_costs(
        df_filtered,
        input_tokens,
        output_tokens,
        api_calls,
        default_model,
        show_token_costs,
    )

    st.dataframe(df_costs, use_container_width=True, hide_index=True)

    fig_total = create_total_cost_chart(df_costs)
    st.plotly_chart(fig_total, use_container_width=True)


if __name__ == "__main__":
    main()
