from ollama_llm_responses import execute_all_prompts

llms = ["phi3","tinyllama", "stablelm2"]

# Prepare to collect results
results = []

# Loop through each model and prompt
for llm in llms:
    df = execute_all_prompts(llm)
    print(f"Prompts has been executed with {llm}")
    json_filename = f"result_basic_{llm}.json"
    df.to_json(json_filename)
    print(f"Prompts and responses has been exported to {json_filename}")
