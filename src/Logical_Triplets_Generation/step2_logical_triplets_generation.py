import json
from tqdm import tqdm
from collections import defaultdict, Counter

with open("./guide_info/guide_info.jsonl", 'r') as infile:
    hop_data = [json.loads(l) for l in infile]

Guide_info = [(item['C'], item['g'], item['cluster2tree'], item['item']) for item in hop_data]

triples = []
with open("../Subgraph_Selection/initial_source/FB15K.txt", "r") as file:
    for line in file:
        items = tuple(i.strip() for i in line.strip().split(","))
        if items[0] != items[2]:
            triples.append(items)

tailrel2head = {}
for head, rel, tail in triples:
    tailrel2head.setdefault((tail, rel), []).append(head)

mid2name = {}
with open("../Subgraph_Selection/initial_source/data_FB15k_mid2name.txt", "r", encoding="utf-8") as file:
    total_lines = sum(1 for _ in file)
with open("../Subgraph_Selection/initial_source/data_FB15k_mid2name.txt", "r", encoding="utf-8") as file:
    for line in tqdm(file, total=total_lines, desc="Loading data"):
        mid, name = line.strip().split("\t")
        mid2name[mid] = name

def convert_mid_to_name(nested_list, mid2name=mid2name):
    return [[mid2name.get(mid, mid) for mid in item] for item in nested_list]

def convert_mid2name(g, mid2name=mid2name):
    return [[mid2name[e] for e in item] for item in g]

def combine_nested_lists(nested_lists):
    combined_list = []
    for lst in nested_lists:
        combined_list.extend(lst)
    return list(set(combined_list))

def transform_triples(triples):
    grouped = defaultdict(list)
    for head, rel, tail in triples:
        if head.startswith("#") and not tail.startswith("#"):
            grouped[(head, rel, 0)].append(tail)
        elif tail.startswith("#") and not head.startswith("#"):
            grouped[(rel, tail, 1)].append(head)
        else:
            grouped[(head, rel, tail)] = None
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
    entity_counts = Counter()
    for head, rel, tail in s:
        entity_counts[head] += 1
        entity_counts[tail] += 1

    leaf_cluster, hidden_cluster = [], []
    for item in g:
        if all(entity_counts[e] <= 1 for e in item):
            leaf_cluster.append(item)
        else:
            hidden_cluster.append(item)

    entity2idx = {}
    idx = 0
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

    result = []
    for head, rel, tail in s:
        triple = (entity2idx[head], rel, entity2idx[tail])
        if triple not in result:
            result.append(triple)

    return transform_triples(result), entity2idx[q_entity], entity2idx

def logicExpressionTransfer_cluster(s, g, c):
    entity_counts = Counter()
    for head, rel, tail in s:
        entity_counts[head] += 1
        entity_counts[tail] += 1

    leaf_cluster, hidden_cluster = [], []
    for item in g:
        if all(entity_counts[e] <= 1 for e in item):
            leaf_cluster.append(item)
        else:
            hidden_cluster.append(item)

    entity2idx = {}
    idx = 0
    for cluster in hidden_cluster:
        for e in cluster:
            entity2idx[e] = f'#{idx}'
        idx += 1

    for cluster in leaf_cluster:
        if cluster == c:
            for e in cluster:
                entity2idx[e] = f'#{idx}'
            idx += 1
        else:
            for e in cluster:
                entity2idx[e] = e

    queryable_cluster = entity2idx[c[0]]
    result = []
    for head, rel, tail in s:
        triple = (entity2idx[head], rel, entity2idx[tail])
        if triple not in result:
            result.append(triple)

    return transform_triples(result), queryable_cluster, entity2idx

def checkLogicSubgraph(triples):
    filtered = []
    for head, rel, tail in triples:
        if any(tail == h and head == t for h, _, t in filtered):
            return False
        if any(head == h and tail == t for h, _, t in filtered):
            return False
        if head == tail or rel.split('/')[1] in ['common', 'type', 'user']:
            return False
        filtered.append((head, rel, tail))
    return True

def find_leaf_nodes(triples):
    entities = []
    for s, _, o in triples:
        entities.append(tuple(s) if isinstance(s, list) else s)
        entities.append(tuple(o) if isinstance(o, list) else o)
    counter = Counter(entities)
    return [list(k) if isinstance(k, tuple) else k for k, v in counter.items() if v == 1]

def check_logic_graph(q_entity, triples):
    leaf_nodes = find_leaf_nodes(triples)
    if not leaf_nodes:
        return False
    for node in leaf_nodes:
        if isinstance(node, str) and "#" in node and node != q_entity:
            return False
    return True

def check_entity2idx(entity2idx):
    return any("#" not in v for v in entity2idx.values())

P, all_P = {}, []
for idx, (C, g, cluster2tree, subgraph) in tqdm(enumerate(Guide_info), total=len(Guide_info), desc="Processing subgraphs"):
    g = convert_mid2name(g)
    C = convert_mid2name(C)
    P[idx] = []
    for c in C:
        if c in g:
            subgraph = convert_mid_to_name(subgraph)
            logic_subgraph, q_entity, entity2idx = logicExpressionTransfer_cluster(subgraph, g, c)
        else:
            for e in c:
                if e in cluster2tree:
                    subgraph = convert_mid_to_name(combine_nested_lists(cluster2tree[e]))
                    logic_subgraph, q_entity, entity2idx = logicExpressionTransfer(subgraph, g, e)
                    break
        if checkLogicSubgraph(logic_subgraph) and check_logic_graph(q_entity, logic_subgraph) and check_entity2idx(entity2idx):
            P[idx].append((q_entity, logic_subgraph, c, entity2idx))
            all_P.append((q_entity, logic_subgraph, c, entity2idx))

def process_jsonl_with_node_count(all_P, output_path):
    label_stats = defaultdict(lambda: {"count": 0, "ids": [], "node_counts": []})
    results = []

    for idx, item in enumerate(all_P):
        logic_subgraph, entity2idx, q_entity = item[1], item[-1], item[0]
        next_idx = max([int(v[1:]) for v in entity2idx.values() if v.startswith('#')], default=0) + 1
        mapping = dict(entity2idx)
        value_to_symbol = {}
        in_deg, out_deg = defaultdict(int), defaultdict(int)
        updated = []

        def get_symbol(node):
            nonlocal next_idx
            if isinstance(node, list):
                key = json.dumps(node, sort_keys=True)
                if key not in value_to_symbol:
                    value_to_symbol[key] = f"#{next_idx}"
                    next_idx += 1
                return value_to_symbol[key]
            if node not in mapping:
                mapping[node] = f"#{next_idx}"
                next_idx += 1
            return mapping[node]

        for s, r, t in logic_subgraph:
            s, t = get_symbol(s), get_symbol(t)
            out_deg[s] += 1
            in_deg[t] += 1
            updated.append([s, r, t])

        degree_stats = defaultdict(int)
        for node in set(in_deg.keys()).union(out_deg.keys()):
            key = f"{out_deg[node]}_{in_deg[node]}"
            degree_stats[key] += 1
        node_count = len(set(in_deg.keys()).union(out_deg.keys()))
        label_key = f"{node_count}_" + json.dumps(dict(degree_stats), sort_keys=True)

        results.append({"id": idx, "logic_subgraph": updated, "label": dict(degree_stats), "node_count": node_count})
        label_stats[label_key]["count"] += 1
        label_stats[label_key]["ids"].append(idx)
        label_stats[label_key]["node_counts"].append(node_count)

    with open(output_path, 'w') as f:
        json.dump(label_stats, f, ensure_ascii=False, indent=2)

    return label_stats

label_stats = process_jsonl_with_node_count(all_P, "./guide_info/hop_summary.json")

def average_sampling_with_entity_and_relation(label_stats, all_P):
    S, R, G = set(), set(), []
    type_counts = {k: 0 for k in label_stats}
    max_count = max(d["count"] for d in label_stats.values())
    total_steps = max_count * len(label_stats)

    with tqdm(total=total_steps, desc="Processing Subgraphs") as pbar:
        for i in range(max_count):
            for label_key, label_data in label_stats.items():
                if i < len(label_data["ids"]):
                    subgraph = all_P[label_data["ids"][i]]
                    entities = set(subgraph[-1].keys())
                    relations = {t[1] for t in subgraph[1]}
                    if not entities.issubset(S) or not relations.issubset(R):
                        G.append(subgraph)
                        S.update(entities)
                        R.update(relations)
                        type_counts[label_key] += 1
                pbar.update(1)
    return G, S, type_counts

G, S, type_counts = average_sampling_with_entity_and_relation(label_stats, all_P)

with open("./guide_info/selected_guide_info.jsonl", 'w') as f:
    json.dump(G, f, indent=2)
