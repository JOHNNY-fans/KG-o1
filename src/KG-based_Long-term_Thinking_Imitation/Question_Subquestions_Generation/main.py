import json
import concurrent.futures
from tqdm import tqdm
from question_generator import generate_multihop_question, decompose_question, verify_subquestions_match
from utils import normalize_answer
import os

with open("../../Logical_Triplets_Generation/guide_info/logic_transfer_result/select_guide_info.jsonl", "r") as f:
    P = json.load(f)


MHQA_list = []
error_log = "error_list.jsonl"
num_workers = 5
output_path = "MHQ_qsub.jsonl"

def process_item(idx, q_entity, logic_subgraph, answer, entity2idx):
    try:
        if any(normalize_answer(a) in ["male", "female"] for a in answer):
            return None

        q_dict = generate_multihop_question(idx, q_entity, logic_subgraph, answer, entity2idx)

        if any(normalize_answer(a) in normalize_answer(q_dict['MHQuestion']) for a in answer):
            with open(error_log, 'a') as f:
                f.write(json.dumps({"id": idx, "error": "answer in question"}) + "\n")
            return None

        subq = decompose_question(q_dict)
        if verify_subquestions_match(subq, logic_subgraph, entity2idx):
            with open(output_path, 'a', encoding='utf-8') as outputfile:
                outputfile.write(json.dumps({"id": idx, "question": q_dict['MHQuestion'], "subq":subq}) + "\n")
            return {"id": idx, "question": q_dict['MHQuestion']}
    except Exception as e:
        with open(error_log, 'a') as f:
            f.write(json.dumps({"id": idx, "error": str(e)}) + "\n")
    return None

with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
    futures = [executor.submit(process_item, idx, q_entity, logic_subgraph, answer, entity2idx)
               for idx, (q_entity, logic_subgraph, answer, entity2idx) in enumerate(P)]
    for fut in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
        res = fut.result()
        if res:
            MHQA_list.append(res)
