# LLM Deployment

## Setup

1. Install Ollama from https://ollama.com/

2. Open terminal and pull the needed models (for this example phi3, tinyllama, stablelm2)
    ```bash
    ollama pull phi3 tinyllama stablelm2
    ```

3. Create and activate a virtual environment:
    ```bash
    python -m venv llm-env
    source llm-env/bin/activate  # On Windows, use `llm-env\Scripts\activate`
    ```

4. Install dependencies:
    ```bash
    pip install pandas ollama
    ```

## Usage

1. Test the models:
    ```bash
    python ollama-ctd.py
    ```

## Project Structure

- `ollama-ctd.py`: Loading the LLMs, execute the hardcoded test-prompts, export the result to .csv file.
- `README.md`: Project documentation.
- `llm-test-results-evaluated.csv`: Initially I created and modified the original .csv output file with my personal verdict about the LLMs reponses on the test-prompts
- `csv-evaluation.py`: Create a .csv from the evaluated test results
- `llm-test-results-summary.csv`: Pass rate of tests of different models