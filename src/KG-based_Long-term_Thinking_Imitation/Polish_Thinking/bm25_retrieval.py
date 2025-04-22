from typing import List, Tuple, Dict
from rank_bm25 import BM25Okapi
from text_processing import preprocess

def bm25_retrieval(query: str, tokenized_documents: List[List[str]], stop_words: set, top_k: int = 10) -> List[Tuple[int, float]]:
    bm25 = BM25Okapi(tokenized_documents)
    tokenized_query = preprocess(query, stop_words)
    scores = bm25.get_scores(tokenized_query)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    top_scores = [scores[i] for i in top_indices]
    return list(zip(top_indices, top_scores))

def generate_triplets(keys: List[str], results: List[Tuple[int, float]], dictionary: Dict[Tuple[str, str], List[str]], source: str, top_n: int = 5) -> List[List]:
    triplets = []
    for idx, score in results:
        key = keys[idx]
        parts = key.split(" ", 1)
        if len(parts) != 2:
            continue
        entity, predicate = parts
        key_tuple = tuple(key.split(" ", 1))
        associated_entities = dictionary.get(key_tuple, [])[:top_n]
        
        if source == 'headrel2tail':
            triplet = [entity, predicate, associated_entities]
        elif source == 'tailrel2head':
            triplet = [associated_entities, predicate, entity]
        else:
            triplet = []
        
        if triplet:
            triplets.append(triplet)
    
    return triplets
