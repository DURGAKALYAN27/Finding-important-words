# Finding Important Words Using TF-IDF with TextBlob

## Introduction
This project demonstrates how to identify the most important words in a set of documents using the **TF-IDF (Term Frequency-Inverse Document Frequency) algorithm** with **TextBlob (v0.19.0)**.

TF-IDF is a statistical measure used to evaluate how important a word is in a document relative to a collection (or corpus) of documents. It is commonly used in **text mining and information retrieval** to rank words by their relevance.

## Installation
Ensure that you have Python installed, then install **TextBlob** and download the necessary NLTK corpora:

```bash
pip install -U textblob
python -m textblob.download_corpora
```

## Implementation

### 1. Import Required Libraries
```python
import math
from textblob import TextBlob
```

### 2. Define TF-IDF Functions
```python
def tf(word, blob):
    return blob.words.count(word) / len(blob.words)

def n_containing(word, bloblist):
    return sum(1 for blob in bloblist if word in blob.words)

def idf(word, bloblist):
    return math.log(len(bloblist) / (1 + n_containing(word, bloblist)))

def tfidf(word, blob, bloblist):
    return tf(word, blob) * idf(word, bloblist)
```

### 3. Prepare Sample Documents
```python
document1 = TextBlob("""Python is a 2000 made-for-TV horror movie directed by Richard
Clabaugh. The film features several cult favorite actors...""")

document2 = TextBlob("""Python, from the Greek word (πύθων/πύθωνας), is a genus of
nonvenomous pythons found in Africa and Asia...""")

document3 = TextBlob("""The Colt Python is a .357 Magnum caliber revolver formerly
manufactured by Colt's Manufacturing Company...""")
```

### 4. Compute and Display TF-IDF Scores
```python
bloblist = [document1, document2, document3]
for i, blob in enumerate(bloblist):
    print(f"Top words in document {i + 1}")
    scores = {word: tfidf(word, blob, bloblist) for word in blob.words}
    sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    for word, score in sorted_words[:3]:
        print(f"\tWord: {word}, TF-IDF: {round(score, 5)}")
```

## Output
This script prints the top three most important words from each document along with their **TF-IDF scores**. These scores highlight words that are significant within their respective documents but not overly common across all documents.

## Notes
- This implementation is compatible with **TextBlob v0.19.0** and Python 3.
- Common stopwords (e.g., "the", "and", "of") are **not** removed, so consider pre-processing your text to filter them out.
- You can expand this project by integrating **NLTK** or **Scikit-learn** for more advanced NLP tasks.



