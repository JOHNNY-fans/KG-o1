import re
import string
from typing import List


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s.replace("_", " ")))))


def preprocess(text: str, stop_words: set) -> List[str]:
    text = text.lower()
    tokens = re.findall(r'\b\w+\b', text)
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return filtered_tokens
