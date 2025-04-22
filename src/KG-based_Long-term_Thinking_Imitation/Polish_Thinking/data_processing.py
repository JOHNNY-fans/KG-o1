import json
from typing import List, Tuple, Dict


def load_triples(file_path: str, mid2name: Dict[str, str]) -> List[Tuple[str, str, str]]:
    triples = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            data_tuple = tuple(item.strip() for item in line.split(","))
            new_tuple = (
                mid2name.get(data_tuple[0], data_tuple[0]),
                data_tuple[1],
                mid2name.get(data_tuple[2], data_tuple[2]),
            )
            if data_tuple[0] != data_tuple[2]:
                triples.append(new_tuple)
    return triples


def build_dictionaries(triples: List[Tuple[str, str, str]]) -> Tuple[Dict[Tuple[str, str], List[str]], Dict[Tuple[str, str], List[str]]]:
    headrel2tail = {}
    tailrel2head = {}
    for (head, r, tail) in triples:
        if (head, r) not in headrel2tail:
            headrel2tail[(head, r)] = []
        headrel2tail[(head, r)].append(tail)
        if (tail, r) not in tailrel2head:
            tailrel2head[(tail, r)] = []
        tailrel2head[(tail, r)].append(head)
    return headrel2tail, tailrel2head
