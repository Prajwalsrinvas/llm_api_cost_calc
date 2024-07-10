# LLM API Cost Calculator

Streamlit app to calculate and compare the costs of various LLM APIs based on input parameters.

![image](https://github.com/Prajwalsrinvas/llm_api_cost_calc/assets/24704548/7968c00d-99e7-47f6-9acd-e7ea7b50d9fe)


## Features

- Fetches up-to-date pricing data from [docsbot.ai](https://docsbot.ai/tools/gpt-openai-api-pricing-calculator)
- Allows user input for tokens and API calls
- Filters results by provider
- Calculates total and relative costs
- Visualizes costs with an interactive bar chart

## Dependencies

- pandas
- plotly
- requests
- beautifulsoup4
- streamlit

## Key Functions

### `fetch_llm_api_cost()`

Fetches and parses LLM API cost data from the website. Uses caching to reduce API calls.

### `load_data()`

Loads and preprocesses the LLM API cost data into a pandas DataFrame.

### `calculate_costs()`

Calculates total and relative costs for each model based on user inputs.

### `create_total_cost_chart()`

Creates a horizontal bar chart visualizing total costs by model.

## Main Application Flow

1. Load and preprocess data
2. Display user input sidebar for parameters and filtering
3. Calculate costs based on user inputs
4. Display results in a table and chart

## Usage

Run the application with:

```
streamlit run app.py
```
