# Testing LLMs

## LLM Deployment

1. Install Ollama from https://ollama.com/

2. Open terminal and pull the needed models (for this example phi3, tinyllama, stablelm2)
    ```bash
    ollama pull phi3 tinyllama stablelm2
    ```

## Usage

1. Create and activate a virtual environment:
    ```bash
    python -m venv llm-env
    source llm-env/bin/activate  # On Windows, use `llm-env\Scripts\activate`
    ```

2. Install dependencies:
    ```bash
    pip install pandas ollama nltk textstat textblob
    ```

3. Install NLTK data:
    ```bash
    python
    >>> import nltk
    >>> nltk.download()
    ```
    Select All to download.

4. Test the models:
    ```bash
    python ollama-ctd.py
    ```

## Project Structure

.
├── ctd-llm-basic-eval/            # Evaluated .csv files. Contains the prompts, the responses, and the PASS/FAIL verdict.
├── ctd-llm-nltk-eval/             # Evaluated .csv files. Contains the prompts, the responses, and the PASS/FAIL verdict and NLTK based statistical metrics
├── llm_nltk_eval.py               # Response evaluation library based on text statistics
├── ollama-ctd.py                  # Executing the the test
├── LICENSE                        # License information
└── README.md                      # This README file