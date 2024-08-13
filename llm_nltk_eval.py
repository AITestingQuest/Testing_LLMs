import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk import FreqDist
import textstat
from textblob import TextBlob

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

# Example usage
#response = "The quick brown fox jumps over the lazy dog. It then runs away quickly."
#metrics = evaluate_llm_response(response)
#print(metrics)
