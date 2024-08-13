import ollama
import pandas as pd
from Testing_LLMs.llm_nltk_eval import evaluate_llm_response

llms = ["phi3","tinyllama", "stablelm2"]

# Define your test prompts
test_prompts = [
    'Classify this sentence as positive, negative or neutral: "I am very disappointed with the service I received at the restaurant"'#,
    #'Select the location and the organization from that statement: "The United Nations is an international organization based in New York City."',
    #'In the context of climate change, what is the greenhouse effect?',
    #'Describe a day in the life of a time traveler who visits ancient Rome.',
    #'Are the phrases "She enjoys hiking in the mountains" and "She loves trekking in the hills" similar in meaning?',
    #'Correct the grammar in the sentence: "She don’t like the new movie."',
    #'What is the result of 45 multiplied by 9?',
    #'What is the result of this arithmetic expression: 2 + 2 * 2 + 2 * 5 - 4',
    #'If all cats are animals and some animals are not dogs, can we conclude that some cats are not dogs?',
    #'What is the capital of Paris?',
]

def execute_prompt(llm, prompt):
    try:
        print(llm)
        print(prompt)
        response = ollama.chat(model=llm, messages=[{'role': 'user', 'content': prompt},],)
        print(response['message']['content'])
        return response['message']['content']
    except Exception as e:
        return str(e)

# Prepare to collect results
results = []

# Loop through each model and prompt
for llm in llms:
    for prompt in test_prompts:
        response = execute_prompt(llm, prompt)
        metrics = evaluate_llm_response(response)
        print(metrics)
        results.append({
            'Model': llm,
            'Prompt': prompt,
            'Response': response,
            'Metrics': metrics
        })

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Save results to a CSV file
results_df.to_csv('llm_test_results.csv', index=False)

# Display the DataFrame
print(results_df)
