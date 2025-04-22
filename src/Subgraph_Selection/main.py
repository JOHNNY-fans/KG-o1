import json
import random
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from data_loader import load_triples, build_index_structures, load_initial_triples
from graph_utils import extract_node, get_expand_rel, get_subgraph

random.seed(42)

def process_triple(t, node2rel_in, node2rel_out, rel2triple):
    S = set()
    nodes = extract_node(t)
    relations = get_expand_rel(nodes, t, node2rel_in, node2rel_out)

    if len(relations) > 10:
        relations = random.sample(relations, 10)

    for r in relations:
        s = get_subgraph(nodes, r, t, rel2triple)
        if len(s) <= 20:
            S.add(tuple(tuple(x) for x in s))
    return S if S else None


def main():
    all_triples = load_triples('./initial_source/FB15K.txt')
    node2rel_in, node2rel_out, rel2triple = build_index_structures(all_triples)
    initial_triples_list = load_initial_triples('./initial_source/initial_triplets.json')

    H = 2
    T = initial_triples_list

    while H < 4:
        all_S = set()
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = {
                executor.submit(process_triple, t, node2rel_in, node2rel_out, rel2triple): t
                for t in T
            }
            for future in tqdm(as_completed(futures), desc=f"Processing round {H}", total=len(T)):
                result = future.result()
                if result:
                    all_S.update(result)

        H += 1
        T = [list(s) for s in all_S]
        with open(f'./generated_subgraphs/{H}_rel.json', 'w', encoding="utf-8") as f:
            json.dump(T, f, indent=4, ensure_ascii=True)


if __name__ == "__main__":
    main()
