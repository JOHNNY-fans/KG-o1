from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib
from tqdm import tqdm
import json
from data_utils import load_triples, load_mid2name, build_tailrel2head, write_to_jsonl
from logic_transform import checkLogicSubgraph
from getcluster import getEntityCluster

def load_subgraph(path):
    with open(path, 'r') as f:
        return json.load(f)

def process_item(item, tailrel2head, save_path):
    if checkLogicSubgraph(item):
        C, g, cluster2tree = getEntityCluster(item, tailrel2head)
        if C:
            write_to_jsonl(save_path, {"C": C, "g": g, "cluster2tree": cluster2tree, "item": item})
            return C, g, cluster2tree, item
    return None

def main():
    triples = load_triples("../Subgraph_Selection/initial_source/FB15K.txt")
    tailrel2head = build_tailrel2head(triples)
    subgraphs = load_subgraph('../Subgraph_Selection/generated_subgraphs/2_rel.json') # Change your subgraph path here
    save_path = "./guide_info/guide_info.jsonl"

    Guide_info = []
    with tqdm_joblib(tqdm(desc="Processing subgraph", total=len(subgraphs))):
        results = Parallel(n_jobs=20)(delayed(process_item)(item, tailrel2head, save_path) for item in subgraphs)
        Guide_info = [r for r in results if r]

    return Guide_info

if __name__ == "__main__":
    main()
