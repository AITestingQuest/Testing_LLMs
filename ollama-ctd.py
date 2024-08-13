import ollama
import pandas as pd
from llm_nltk_eval import evaluate_llm_response

llms = ["phi3","tinyllama", "stablelm2"]

# Define your test prompts
test_prompts = [
    'Classify this sentence as positive, negative or neutral: "I am very disappointed with the service I received at the restaurant"',
    'Select the location and the organization from that statement: "The United Nations is an international organization based in New York City."',
    'In the context of climate change, what is the greenhouse effect?',
    'Describe a day in the life of a time traveler who visits ancient Rome.',
    'Are the phrases "She enjoys hiking in the mountains" and "She loves trekking in the hills" similar in meaning?',
    'Correct the grammar in the sentence: "She don’t like the new movie."',
    'What is the result of 45 multiplied by 9?',
    'What is the result of this arithmetic expression: 2 + 2 * 2 + 2 * 5 - 4',
    'If all cats are animals and some animals are not dogs, can we conclude that some cats are not dogs?',
    'What is the capital of Paris?',
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
        response = response.strip("\r\n")
        results.append({
            'Model': llm,
            'Prompt': prompt,
            'Response': response,
            'Word Count': metrics["word_count"],
            "Sentence Count": metrics["sentence_count"],
            "Avg Sentence Length": metrics["avg_sentence_length"],
            "Lexical Diversity": metrics["lexical_diversity"],
            "Stopword Ratio": metrics["stopword_ratio"],
            "Flesch Kincaid Grade": metrics["flesch_kincaid_grade"],
            "Flesch Reading Ease": metrics["flesch_reading_ease"],
            "Sentiment Polarity": metrics["sentiment_polarity"],
            "Sentiment Subjectivity": metrics["sentiment_subjectivity"],
            "Content Word Ratio": metrics["content_word_ratio"],
            "Transition Word Ratio": metrics["transition_word_ratio"],
            "Repetition Rate": metrics["repetition_rate"]
        })

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Save results to a CSV file
results_df.to_csv('llm_nltk_test_results.csv', index=False, sep="\t")

# Display the DataFrame
print(results_df)
