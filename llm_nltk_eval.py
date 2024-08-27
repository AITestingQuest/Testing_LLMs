from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk import FreqDist
import textstat
from textblob import TextBlob
import ollama
import pandas as pd

def evaluate_llm_response(response, keywords=None):
    # Tokenize words and sentences
    words = word_tokenize(response)
    sentences = sent_tokenize(response)
    
    word_count = len(words)
    sentence_count = len(sentences)
    
    # Lexical Diversity
    lexical_diversity = len(set(words)) / word_count if word_count else 0
    
    # Average Sentence Length (Syntactic Complexity)
    avg_sentence_length = word_count / sentence_count if sentence_count else 0
    
    # Stopword Frequency (Vocabulary Usage)
    stop_words = set(stopwords.words('english'))
    stopword_count = sum(1 for word in words if word.lower() in stop_words)
    stopword_ratio = stopword_count / word_count if word_count else 0
    
    # Part of Speech Tagging
    pos_tags = pos_tag(words)
    pos_counts = FreqDist(tag for (word, tag) in pos_tags)
    
    # Readability Metrics
    flesch_kincaid_grade = textstat.flesch_kincaid_grade(response)
    flesch_reading_ease = textstat.flesch_reading_ease(response)
    
    # Sentiment Analysis (Fluency and Coherence)
    sentiment_analysis = TextBlob(response)
    sentiment_polarity = sentiment_analysis.sentiment.polarity
    sentiment_subjectivity = sentiment_analysis.sentiment.subjectivity
    
    # Content Word Ratio (Information Density)
    content_word_count = sum(1 for word, tag in pos_tags if tag.startswith(('N', 'V', 'J', 'R')))
    content_word_ratio = content_word_count / word_count if word_count else 0
    
    # Transition Word Frequency (Logical Flow)
    transition_words = set(["however", "therefore", "moreover", "thus", "furthermore", "consequently", "nevertheless", "on the other hand", "in contrast", "in conclusion"])
    transition_word_count = sum(1 for word in words if word.lower() in transition_words)
    transition_word_ratio = transition_word_count / word_count if word_count else 0
    
    # Repetition Rate (Error Analysis)
    word_freq = FreqDist(words)
    most_common_word, most_common_word_count = word_freq.most_common(1)[0] if word_freq else (None, 0)
    repetition_rate = most_common_word_count / word_count if word_count else 0
    
    
    # Compile all metrics into a dictionary
    metrics = {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "avg_sentence_length": avg_sentence_length,
        "lexical_diversity": lexical_diversity,
        "stopword_ratio": stopword_ratio,
        "pos_counts": pos_counts.most_common(),
        "flesch_kincaid_grade": flesch_kincaid_grade,
        "flesch_reading_ease": flesch_reading_ease,
        "sentiment_polarity": sentiment_polarity,
        "sentiment_subjectivity": sentiment_subjectivity,
        "content_word_ratio": content_word_ratio,
        "transition_word_ratio": transition_word_ratio,
        "repetition_rate": repetition_rate,
    }
    
    return metrics

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

