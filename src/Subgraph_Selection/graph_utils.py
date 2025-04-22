import random

def extract_node(triples):
    entities = set()
    for s, _, o in triples:
        entities.update([s, o])
    return list(entities)


def get_rel(triples):
    return list({r for _, r, _ in triples})


def get_expand_rel(entities, init_triple, node2rel_in, node2rel_out):
    result = []
    existing_rels = set(get_rel(init_triple))
    
    for n in entities:
        candidate_rels = node2rel_in.get(n, []) + node2rel_out.get(n, [])
        filtered_rels = set(r for r in candidate_rels if r not in existing_rels)
        if len(filtered_rels) <= 50:
            result.extend(filtered_rels)

    return result


def get_subgraph(entities, r, initial_triple, rel2triple):
    subgraph = [tuple(i) for i in initial_triple]
    extend_subgraph = []

    for n in entities:
        for h, t in rel2triple.get(r, []):
            if h == n or t == n:
                triple = (h, r, t)
                if triple not in extend_subgraph:
                    extend_subgraph.append(triple)

    random.seed(42)
    subgraph += random.sample(extend_subgraph, min(3, len(extend_subgraph)))
    return subgraph
