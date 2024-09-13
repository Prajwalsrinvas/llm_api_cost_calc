# LLM API Cost Calculator

Streamlit app to calculate and compare the costs of various LLM APIs based on input parameters.

![screencapture-llm-api-cost-streamlit-app-2024-09-13-12_29_09](https://github.com/user-attachments/assets/0b702d86-b053-49c0-bf81-814a4db2d096)



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
