import json
from collections import Counter

def load_triples(file_path):
    triples = []
    with open(file_path, "r") as file:
        for line in file:
            data_tuple = tuple(item.strip() for item in line.strip().split(","))
            if data_tuple[0] != data_tuple[2]:
                triples.append(data_tuple)
    return triples

def build_tailrel2head(triples):
    tailrel2head = {}
    for (head, r, tail) in triples:
        tailrel2head.setdefault((tail, r), []).append(head)
    return tailrel2head

def load_mid2name(file_path):
    mid2name = {}
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            mid, name = line.strip().split("\t")
            mid2name[mid] = name
    return mid2name

def convert_mid_to_name(nested_list, mid2name):
    return [[mid2name.get(mid, mid) for mid in item] for item in nested_list]

def write_to_jsonl(filepath, data):
    with open(filepath, 'a') as f:
        f.write(json.dumps(data) + "\n")

def find_leaf_nodes(triples):
    all_entities = []
    for triple in triples:
        subject, _, obj = triple
        all_entities.append(subject if not isinstance(subject, list) else tuple(subject))
        all_entities.append(obj if not isinstance(obj, list) else tuple(obj))
    entity_counter = Counter(all_entities)
    return [e if not isinstance(e, tuple) else list(e) for e, count in entity_counter.items() if count == 1]
