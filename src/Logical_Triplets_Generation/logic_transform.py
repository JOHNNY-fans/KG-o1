from collections import defaultdict
from data_utils import find_leaf_nodes

def transform_triples(triples):
    grouped = defaultdict(list)
    for h, r, t in triples:
        if h.startswith("#") and not t.startswith("#"):
            grouped[(h, r, 0)].append(t)
        elif t.startswith("#") and not h.startswith("#"):
            grouped[(r, t, 1)].append(h)
        else:
            grouped[(h, r, t)] = None

    transformed = []
    for key, values in grouped.items():
        if isinstance(values, list) and values:
            if key[2] == 0:
                transformed.append((key[0], key[1], values[:3] if len(values) > 1 else values[0]))
            elif key[2] == 1:
                transformed.append((values[:3] if len(values) > 1 else values[0], key[0], key[1]))
        else:
            transformed.append(key)
    return transformed

def logicExpressionTransfer(s, g, q_entity):
    entity_counts = {}
    for h, _, t in s:
        entity_counts[h] = entity_counts.get(h, 0) + 1
        entity_counts[t] = entity_counts.get(t, 0) + 1

    leaf_cluster, hidden_cluster = [], []
    for cluster in g:
        if all(entity_counts.get(e, 0) <= 1 for e in cluster):
            leaf_cluster.append(cluster)
        else:
            hidden_cluster.append(cluster)

    idx, entity2idx = 0, {}
    for cluster in hidden_cluster:
        for e in cluster:
            entity2idx[e] = f'#{idx}'
        idx += 1

    for cluster in leaf_cluster:
        if q_entity in cluster:
            for e in cluster:
                entity2idx[e] = f'#{idx}'
            idx += 1
        else:
            for e in cluster:
                entity2idx[e] = e

    triplets = [(entity2idx[h], r, entity2idx[t]) for h, r, t in s]
    return transform_triples(list(set(triplets))), entity2idx[q_entity], entity2idx

def checkLogicSubgraph(logic_sg_list):
    filtered = []
    if any("#" not in h or "#" not in t for h, _, t in logic_sg_list):
        for h, r, t in logic_sg_list:
            if any(t == eh and h == et for eh, _, et in filtered):
                return False
            if h == t or r.split('/')[1] in ['common', 'type', 'user']:
                return False
            filtered.append((h, r, t))
        return True
    return False

def check_logic_graph(q_entity, logic_subgraph):
    for leaf in find_leaf_nodes(logic_subgraph):
        if isinstance(leaf, str) and '#' in leaf and leaf != q_entity:
            return False
    return True
