import re
import string
from difflib import SequenceMatcher
from collections import defaultdict

def normalize_answer(s):
    return ' '.join(re.sub(r'[^\w\s]', '', s.lower().replace("_", " ")).split())

def resolve_triples(triples, entity_map):
    id_to_entities = defaultdict(list)
    for name, eid in entity_map.items():
        id_to_entities[eid].append(name)

    def resolve(e):
        if isinstance(e, list):
            return " and ".join(" and ".join(id_to_entities.get(i, [i])) for i in e)
        return " and ".join(id_to_entities.get(e, [e]))

    return {
        str([s, r, o]): f"{resolve(s)} {r.split('/')[-1]} {resolve(o)}"
        for s, r, o in triples
    }

def find_most_similar_key(input_str, candidate_dict):
    return max(candidate_dict.items(), key=lambda x: SequenceMatcher(None, input_str, x[0]).ratio())
