"""
Extractive Text Summarization using TF-IDF

This script performs extractive summarization by:
1. Splitting input text into smaller sentence-like chunks
2. Converting each chunk into a TF-IDF vector
3. Scoring chunks based on word importance
4. Selecting the top-K highest scoring chunks as the summary

This approach does NOT generate new text — it selects the most
important sentences from the original input.
"""

# Imports a smart text splitter from LangChain
# It splits text while respecting sentence and paragraph boundaries
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Imports TF-IDF vectorizer to convert text into numerical importance scores
from sklearn.feature_extraction.text import TfidfVectorizer

# NumPy is used for efficient numerical operations
import numpy as np


def extractive_summary(text, top_k=3):
    """
    Generates an extractive summary of the input text.

    Parameters:
    text (str): Input document to summarize
    top_k (int): Number of top-ranked sentences to include in summary

    Returns:
    str: Extracted summary composed of original sentences
    """

    # Create a text splitter that breaks text into chunks
    # chunk_size=200 → max characters per chunk
    # chunk_overlap=0 → no overlapping text between chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=0
    )

    # Split the input text into sentence-like chunks
    sentences = splitter.split_text(text)

    # Initialize TF-IDF vectorizer
    # stop_words="english" removes common words like "the", "is", "and"
    vectorizer = TfidfVectorizer(stop_words="english")

    # Convert sentences into a TF-IDF matrix
    # Rows = sentences, Columns = unique words
    tfidf = vectorizer.fit_transform(sentences)

    # Sum TF-IDF scores across each sentence
    # Higher score → more important sentence
    scores = np.sum(tfidf.toarray(), axis=1)

    # Get indices of top-K highest scoring sentences
    # argsort() sorts indices by score
    # [-top_k:] selects top K
    # [::-1] reverses to descending order
    top_indices = scores.argsort()[-top_k:][::-1]

    # Combine the selected sentences into a single summary string
    return " ".join([sentences[i] for i in top_indices])


# Sample input text
text = """
President Donald Trump says the US is exploring a potential deal on Greenland after talks with Nato as he backed off threats to tariff European allies that had opposed his plans for America to acquire the island.

On social media, Trump offered few details about a discussion that both he and Nato described as "very productive".

After rattling the transatlantic alliance with weeks of rhetoric, the US president said the meeting had led to the "framework" of a potential agreement.

But there was no suggestion of a deal that might meet Trump's demand for "ownership" of Greenland, an ambition he restated at the World Economic Forum in Switzerland, while also ruling out military force.

On Truth Social on Wednesday, the US president said: "We have formed the framework of a future deal with respect to Greenland and, in fact, the entire Arctic Region.

"This solution, if consummated, will be a great one for the United States of America, and all Nato Nations."

Diplomatic sources told the BBC's US partner CBS that there was no agreement for American control or ownership of the autonomous Danish dependent territory.
"""

# Generate and print the extractive summary
print(extractive_summary(text))
