from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

triples = []

with open("./scripts/initial_source/FB15K.txt", "r") as file:
    for line in file:
        line = line.strip()
        data_tuple = tuple(item.strip() for item in line.split(","))
        if data_tuple[0] != data_tuple[2]:
            triples.append(data_tuple)


node2rel_in = {}
node2rel_out = {}
rel2triple = {}
headrel2tail = {}
tailrel2head = {}


for (head,r,tail) in triples:
    if head not in node2rel_out.keys():
        node2rel_out[head] = []
        node2rel_out[head].append(r)
    else:
        node2rel_out[head].append(r)

    if tail not in node2rel_in.keys():
        node2rel_in[tail] = []
        node2rel_in[tail].append(r)
    else:
        node2rel_in[tail].append(r)

    if r not in rel2triple.keys():
        rel2triple[r] = []
        rel2triple[r].append([head,tail])
    else:
        rel2triple[r].append([head,tail])

    if (head,r) not in headrel2tail.keys():
        headrel2tail[(head,r)] = []
        headrel2tail[(head,r)].append(tail)
    else:
        headrel2tail[(head,r)].append(tail)

    if (tail,r) not in tailrel2head.keys():
        tailrel2head[(tail,r)] = []
        tailrel2head[(tail,r)].append(head)
    else:
        tailrel2head[(tail,r)].append(head)


import json
save_initial_triplets_path = './initial_source/initial_triplets.json'
with open(save_initial_triplets_path,'r') as f:
    initial_triples_list = json.load(f)

print(len(initial_triples_list))

new_initial_triples = []
for item in initial_triples_list:
    tmp = []
    for i in item:
        if i[0] != i[2]:
            tmp.append(i)
    if tmp != []:
        new_initial_triples.append(tmp)

initial_triples_list = new_initial_triples

def extractNode(triples):
    entities = set() 
    for triple in triples:
        subject, predicate, object_ = triple
        entities.add(subject) 
        entities.add(object_)
    return list(entities)

def getRel(i):
    R = []
    for _,r,_ in i:
        if r not in R:
            R.append(r)
    return R


def getExpandRel(N, initial_triple, node2rel_in=node2rel_in,node2rel_out=node2rel_out):
    result = []
    R_i = getRel(initial_triple)
    for n in N:
        R = []
        if n in node2rel_in.keys():
            R += node2rel_in[n]
        if n in node2rel_out.keys():
            R += node2rel_out[n]
        R = set([i for i in R if i not in R_i])
        length = len(R)
        if length >50:
            continue
        else:
            result+=R
    return result


import random
def getSubgraph(N, r, initial_triple, rel2triple=rel2triple):
    """
    Extract a subgraph from a given list of triples that match specific relations and entities.

    Parameters:
    - N: The specified entity. Triples containing this entity in either the head entity or tail entity will be selected.
    - r: The specified relation. The relation in the triples must be equal to this relation.
    - triples: A list of triples, where each element is (head entity, relation, tail entity).

    Returns:
    - A list of triples that meet the specified conditions.

    """
    subgraph = [tuple(item) for item in initial_triple]
    extend_subgraph = []

    for n in N:
        rt = rel2triple[r]
        for item in rt:
            if item[0] == n or item[1] == n:
                if (item[0], r, item[1]) not in extend_subgraph:
                    extend_subgraph.append((item[0], r, item[1]))
    random.seed(42)

    if len(extend_subgraph)>3:
        subgraph+=random.sample(extend_subgraph, 3)
    else:
        subgraph+=extend_subgraph

    return subgraph


H = 2
T = initial_triples_list
I = set()

def add_to_set(I, E, s, cluster2tree):
    E_tuple = tuple(tuple(sublist) for sublist in E)  # Convert list of lists to tuple of tuples
    s_tuple = tuple(s)  # Convert list of triples to tuple
    cluster2tree_tuple = tuple((k, tuple(tuple(triple) for triple in v)) for k, v in cluster2tree.items())
    I.add((E_tuple, s_tuple, cluster2tree_tuple))


def process_triple(t):
    S = set()
    N = extractNode(t)
    R = getExpandRel(N, t)
    random.seed(42)
    if len(R) > 10:
        R = random.sample(R,10) 


    if len(R)!=0:
        for r in R:
            s = getSubgraph(N, r, t)
            if len(s)>20:
                continue
            else:
                S.add(tuple(tuple(item) if isinstance(item, list) else item for item in s))  # Ensure elements in s are hashable
        return S
    else:return None

while H < 4:
    I = set()
    all_S = set()
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(process_triple, t): t for t in T}
        for future in tqdm(as_completed(futures), desc="total initial triplets", total=len(T)):
            S = future.result()
            if S is not None: 
                all_S.update(S)

    H += 1
    save_path = f'./generated_subgraphs/{H}.json'
    T = [list(s) for s in all_S]
    with open(save_path, 'w', encoding="utf-8") as f:
        json.dump(T, f, indent=4, ensure_ascii=True)

