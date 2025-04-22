import json

def load_triples(filepath):
    triples = []
    with open(filepath, "r") as file:
        for line in file:
            data_tuple = tuple(item.strip() for item in line.strip().split(","))
            if data_tuple[0] != data_tuple[2]:
                triples.append(data_tuple)

    print(f"The length of FB15K that remove self-loop triplets:{len(triples)}")
    return triples


def build_index_structures(triples):
    node2rel_in, node2rel_out = {}, {}
    rel2triple, headrel2tail, tailrel2head = {}, {}, {}

    for head, r, tail in triples:
        node2rel_out.setdefault(head, []).append(r)
        node2rel_in.setdefault(tail, []).append(r)
        rel2triple.setdefault(r, []).append([head, tail])
        headrel2tail.setdefault((head, r), []).append(tail)
        tailrel2head.setdefault((tail, r), []).append(head)

    return node2rel_in, node2rel_out, rel2triple


def load_initial_triples(path):
    with open(path, 'r') as f:
        data = json.load(f)

    filtered = []
    for item in data:
        valid_triples = [i for i in item if i[0] != i[2]]
        if valid_triples:
            filtered.append(valid_triples)

    print(f"The length of initial triples that remove self-loop triplets:{len(filtered)}")
    return filtered
