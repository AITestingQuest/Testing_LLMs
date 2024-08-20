import ollama

def execute_prompt(llm, prompt):
    try:
        response = ollama.chat(model=llm, messages=[{'role': 'user', 'content': prompt},],)
        return response['message']['content']
    except Exception as e:
        return str(e)
