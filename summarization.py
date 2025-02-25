from transformers import pipeline
import re
from detoxify import Detoxify

# Load the summarization model from Hugging Face
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def filter_toxicity(text):
    """Checks and removes toxic content from text."""
    toxicity_scores = Detoxify('original').predict(text)
    if toxicity_scores['toxicity'] > 0.5:
        return "Content removed due to detected toxicity."
    return text

def summarize_text(text, summary_length="short"):
    """Generates a text summary while ensuring safety and neutrality."""
    
    # Apply safety filters
    text = filter_toxicity(text)
    if text == "Content removed due to detected toxicity.":
        return text

    # Define summary length parameters
    length_options = {
        "short": (10, 30),   # Min 10 words, Max 30 words
        "medium": (30, 60),  # Min 30 words, Max 60 words
        "long": (60, 120)    # Min 60 words, Max 120 words
    }

    # Get the appropriate min/max length
    min_length, max_length = length_options.get(summary_length, (10, 30))

    # Generate summary
    summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)

    return summary[0]['summary_text']

# Test the summarizer
if __name__ == "__main__":
    sample_text = "Generative AI models can generate human-like text based on given input. They are used in various applications like content generation, chatbots, and text summarization."
    
    print("Short Summary:\n", summarize_text(sample_text, "short"))
