# Testing LLMs

## LLM Deployment

1. Install Ollama from https://ollama.com/

2. Open terminal and pull the needed models (for this example phi3, tinyllama, stablelm2)
    ```bash
    ollama pull phi3 tinyllama stablelm2
    ```
## Preparation
1. Create and activate a virtual environment:
    ```bash
    python -m venv llm-env
    source llm-env/bin/activate  # On Windows, use `llm-env\Scripts\activate`
    ```

2. Add your OPENAI_API_KEY to the llm-env/bin/activate.bat file

3. Install dependencies:
    ```bash
    pip install pandas ollama nltk textstat textblob deepeval evidently
    ```

## Usage of the initial test
1. Test the models:
    ```bash
    python ollama-ctd.py
    ```

## Usage of NLTK

1. Install NLTK data:
    ```bash
    python
    >>> import nltk
    >>> nltk.download()
    ```
    Select All to download.

2. Test the models:
    ```bash
    python llm_nltk_eval.py
    ```

## Usage of DeepEval

1. The LLMs to be tested is hardcoded in the code. Modify the LLM in llm_deepeval_eval.py, if you'd like to test a different one.

2. Test the models:
    ```bash
    python llm_deepeval_eval.py
    ```

## Usage of Evidenly AI

1. Test the models:
    ```bash
    python llm_evidenlty_eval.py
    ```

## Project Structure
```bash
.
├── ctd-llm-basic-eval/            # Evaluated .csv files. Contains the prompts, the responses, and the PASS/FAIL verdict.
├── ctd-llm-nltk-eval/             # Evaluated .csv files. Contains the prompts, the responses, and the PASS/FAIL verdict and NLTK based statistical metrics
├── ctd-llm-deepeval-eval/         # Evaluated .csv files. Contains the prompts, the responses, and the PASS/FAIL verdict based on DeepEval
├── ctd-llm-evidently-eval/        # Evaluated .json and .html files. Contains the prompts, the responses. For the PASS/FAIL rate, execute test_eval.py
├── llm_deepeval_eval.py           # Executing the test based on DeepEval
├── llm_evidently_eval.py          # Executing the test based on Evidently AI
├── llm_nltk_eval.py               # Executing the test based on NLTK
├── ollama_llm_responses.py        # Lightweight method for just executing a prompt from a given LLM
├── ollama-ctd.py                  # Executing the initial test
├── LICENSE                        # License information
└── README.md                      # This README file
```
