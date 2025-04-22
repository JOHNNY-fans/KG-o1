import json
from collections import defaultdict
from typing import Dict, List, Tuple, Any
from retrying import retry
from tqdm import tqdm
from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib
from difflib import SequenceMatcher

# Import local modules
from data_processing import load_triples, build_dictionaries
from text_processing import normalize_answer, preprocess
from bm25_retrieval import bm25_retrieval, generate_triplets
from reasoning import extract_topics, convert_reason_triple_raw
from utils import find_json_output, generate_gpt4o
from prompt import prompt_first, prompt_mid, prompt_last, prompt_last_retry,prompt_generata_answer

# Constants
STOP_WORDS = {
    'a', 'an', 'the', 'and', 'or', 'but', 'if', 'while', 'with', 'without', 'of',
    'at', 'by', 'for', 'to', 'in', 'on', 'from', 'up', 'down', 'out', 'over',
    'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where',
    'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other',
    'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
    'very', 'can', 'will', 'just', 'don', 'should', 'now'
}

# Load data files
def load_data_files():
    """Load and initialize all required data files."""
    # Load mid2name mapping
    mid2name = {}
    with open("../../Subgraph_Selection/initial_source/data_FB15k_mid2name.txt", "r", encoding="utf-8") as file:
        for line in tqdm(file, desc="Loading mid2name data"):
            mid, name = line.strip().split("\t")
            mid2name[mid] = name

    # Load triples
    triples = load_triples("../../Subgraph_Selection/initial_source/FB15K.txt", mid2name)
    headrel2tail, tailrel2head = build_dictionaries(triples)

    # Load guide info
    guide_info = []
    with open("../../Logical_Triplets_Generation/guide_info/logic_transfer_result/select_guide_info_2rel.jsonl", 'r',encoding='utf-8') as infile:
        guide_info = json.load(infile)

    guide_info_dict = {str(idx): item for idx, item in enumerate(guide_info)}
    guide_info_id2entity_dict = {str(idx): item[-1] for idx, item in enumerate(guide_info)}

    return mid2name, headrel2tail, tailrel2head, guide_info_dict, guide_info_id2entity_dict

# Triple processing functions
def resolve_triples(triples: List[Tuple], entity_mapping: Dict) -> Dict:
    """Resolve triples to human-readable format using entity mapping."""
    id_to_entities = defaultdict(list)
    for entity, eid in entity_mapping.items():
        id_to_entities[eid].append(entity)

    def resolve_entity(entity):
        if isinstance(entity, list):
            return " and ".join([" and ".join(id_to_entities.get(e, [e])) for e in entity])
        return " and ".join(id_to_entities.get(entity, [entity]))

    resolved_triples_dict = {}
    for triple in triples:
        if not isinstance(triple, list) or len(triple) != 3:
            raise ValueError(f"Invalid triple format: {triple}")
        resolved_triples_dict[str(triple)] = triple  # Store original triple as value

    return resolved_triples_dict

def find_most_similar_key(input_str: str, data_dict: Dict) -> Tuple[str, float]:
    """Find the most similar key in a dictionary to the input string."""
    max_similarity = 0
    most_similar_key = None
    for key in data_dict:
        similarity = SequenceMatcher(None, input_str, key).ratio()
        if similarity > max_similarity:
            max_similarity = similarity
            most_similar_key = key
    return most_similar_key, max_similarity

# Knowledge retrieval functions
def get_bm25_result(sub_problem: str, reasoning_triplet: List, 
                    headrel2tail: Dict, tailrel2head: Dict) -> Tuple[List, List]:
    """Retrieve relevant triples using BM25 algorithm."""
    topics = extract_topics(reasoning_triplet, mid2name, sub_problem)
    topics = list(set(topics))
    query = " ".join(topics)
    
    head_keys = [" ".join(key) for key in headrel2tail.keys()]
    tail_keys = [" ".join(key) for key in tailrel2head.keys()]
    
    head_tokenized = [preprocess(doc, STOP_WORDS) for doc in head_keys]
    tail_tokenized = [preprocess(doc, STOP_WORDS) for doc in tail_keys]
    
    head_bm25_results = bm25_retrieval(query, head_tokenized, STOP_WORDS, top_k=10)
    tail_bm25_results = bm25_retrieval(query, tail_tokenized, STOP_WORDS, top_k=10)
    
    head_triplets = generate_triplets(head_keys, head_bm25_results, headrel2tail, 'headrel2tail', top_n=3)
    tail_triplets = generate_triplets(tail_keys, tail_bm25_results, tailrel2head, 'tailrel2head', top_n=3)

    return head_triplets, tail_triplets

@retry(stop_max_attempt_number=5, wait_exponential_multiplier=1000, wait_exponential_max=10000)
def convert_triple_para(triplets: List) -> str:
    """Convert triples to natural language description."""
    prompt = f'''
    Convert the following triples into a coherent natural language description.
    All information of the triplet must be used.
    Input: {triplets}
    Output in JSON format:
    {{
        "description":"a natural language description"
    }}
    '''
    return find_json_output(generate_gpt4o(prompt))['description']

# Question processing functions
def find_sub_key(q_sub_dict: Dict) -> str:
    """Find the key containing 'sub_problem' in a dictionary."""
    for key in q_sub_dict:
        if 'sub_problem' in key:
            return key

def extract_sub_problems(data: List[Dict]) -> List[str]:
    """Extract sub-problems from a list of dictionaries."""
    transformed_data = []
    for item in data:
        transformed_item = {}
        for key in item:
            if key.startswith('sub_problem_'):
                transformed_item[key] = item[key]
        transformed_data.append(transformed_item)
    return [list(item.values())[0] for item in transformed_data]

@retry(stop_max_attempt_number=5, wait_exponential_multiplier=1000, wait_exponential_max=10000)
def get_split_details(q: str, q_sub: List[str], think: str) -> List[Dict]:
    """Split reasoning process into parts corresponding to sub-problems."""
    prompt = f'''
    You are given:
    1. A **multi-step question**: {q}
    2. A list of **sub-questions** derived from the multi-step question: {q_sub}
    3. A detailed **reasoning process**: {think}

    **Your task** is to:
    1. Divide the reasoning process into **N** parts, where **N** corresponds to the number of sub-questions.
    2. Each part of the reasoning process should correspond directly to one sub-question.
    3. Produce a **list** of dictionaries, where each dictionary has exactly one key-value pair.
    - The **key** should be the specific content of each sub-question from the **sub-questions** list.
    - The **value** should be the corresponding part of the reasoning process that is relevant to that sub-question.

    **Output format**: 
    Return the final list of dictionaries, preserving the order, and do not include any extra explanations.

    Example of the desired structure:

    ```json
    [
        {{"<<Content of Sub-question #1>>": "Relevant reasoning part for sub-question #1"}},
        {{"<<Content of Sub-question #2>>": "Relevant reasoning part for sub-question #2"}},
        ...
    ]
    ```
    '''
    return find_json_output(generate_gpt4o(prompt))

@retry(stop_max_attempt_number=2, wait_exponential_multiplier=1000, wait_exponential_max=10000)
def generate_thinking(prompt,history=None):
    output = find_json_output(generate_gpt4o(prompt,history))
    return output

# Verification functions
@retry(stop_max_attempt_number=5, wait_exponential_multiplier=1000, wait_exponential_max=10000)
def verify(q_sub: str, gt: str, detail: str) -> bool:
    """Verify if ground truth is included in candidate answers."""
    prompt = f'''
    Check if this answer is correct for the question:
    Question: {q_sub}
    Answer: {gt}
    Candidate: {detail}
    Output format:
    {{
        "evaluation": "Yes or No",
        "justification": "reason"
    }}
    '''
    output = find_json_output(generate_gpt4o(prompt))
    return "Y" in output['evaluation'] or "y" in output['evaluation']

# Main processing function
def process_item(item_all: Dict, mid2name: Dict, headrel2tail: Dict, 
                tailrel2head: Dict, guide_info_dict: Dict, 
                guide_info_id2entity_dict: Dict) -> Dict:
    """Process a single question item through the full pipeline."""
    try:
        # Initialize data from input
        example = item_all
        q = example['q']
        logic_subgraph = example['logic_subgraph']
        entity2idx = guide_info_id2entity_dict[example['id']]
        q_sub_dict = example['q_sub_dict']
        think = example['details']
        ground_truth = guide_info_dict[example['id']][2]

        # Step 1: Match sub-questions with triples
        print("# Step 1: Match sub-questions with triples")
        resolved_triples_dict = resolve_triples(logic_subgraph, entity2idx)
        q_sub_clean_list = []

        count = 0
        for item in q_sub_dict:
            count+=1
            input_str = str(item['reasoning_triplet'])
            most_similar_key, _ = find_most_similar_key(input_str, resolved_triples_dict)
            sub_key = next(k for k in item.keys() if "sub_problem" in k)
            
            q_sub_clean_list.append({
                f"sub_problem_{count}": item[sub_key],
                "reasoning_triplet": resolved_triples_dict[most_similar_key],
                "reason": item['reason']
            })

        # Step 2: Retrieve knowledge for each sub-question
        print("# Step 2: Retrieve knowledge for each sub-question")
        qsub2para = {}
        # sub_problem_str = ' '.join(item[find_sub_key(item)] for item in q_sub_clean_list)
        
        for item in q_sub_clean_list:
            sub_pro = item[find_sub_key(item)]
            reasoning_triplet = item['reasoning_triplet']
            
            head_triplets, tail_triplets = get_bm25_result(
                sub_pro, reasoning_triplet, headrel2tail, tailrel2head
            )
            
            para = convert_triple_para(
                head_triplets + tail_triplets + convert_reason_triple_raw(reasoning_triplet, entity2idx)
            )
            
            qsub2para[sub_pro] = {
                "knowledge": para,
                "ground_truth": convert_reason_triple_raw(reasoning_triplet, entity2idx)
            }

        print("qsub2para key:",list(qsub2para.keys()))

        # Step 3: Split and polish reasoning
        q_sub = extract_sub_problems(q_sub_clean_list)
        split_details = get_split_details(q, q_sub, think)
        
        # Prepare knowledge for polishing
        source_knowledge = {
            normalize_answer(q): {
                "knowledge": qsub2para[q]['knowledge'],
                "sub_gt": qsub2para[q]['ground_truth']
            } for q in qsub2para.keys()
        }

        print(source_knowledge.keys())
        
        guide_info = []
        for item in split_details:
            key = normalize_answer(list(item.keys())[0])
            guide_info.append({
                "question": list(item.keys())[0],
                "knowledge": source_knowledge[key],
                "details": list(item.values())[0]
            })

        # Generate polished thinking
        all_result = []
        for idx, item in enumerate(guide_info):
            sub_question = item['question']
            process = item['details']
            knowledge = item['knowledge']['knowledge']
            sub_a = item['knowledge']['sub_gt']
            max_iterations = 2
            history = []
            counter = 0

            if idx == 0:
                while True:
                    prompt = prompt_first.format(
                        q=q,
                        sub_q=q_sub,
                        sub_question=sub_question,
                        process=process,
                        knowledge=knowledge,
                        sub_a=sub_a
                    )
                    output = generate_thinking(prompt)
                    label = verify(sub_question, sub_a, output["candidate_answers"])
                    print(label)

                    counter += 1
                    if label or counter >= max_iterations:
                        break

                all_result.append(output)
                history.append({"user": prompt, "assistant": output})

            elif idx == (len(guide_info) - 1):
                while True:
                    prompt = prompt_last.format(
                        q=q,
                        sub_q=q_sub,
                        sub_question=sub_question,
                        process=process,
                        knowledge=knowledge,
                        ground_truth=ground_truth
                    )
                    output = generate_thinking(prompt,history)
                    label = verify(sub_question, ground_truth, output["final_answer"])
                    print(label)
                    counter += 1
                    if label or counter >= max_iterations:
                        break

                if label:
                    all_label = True
                    final_answer = output["final_answer"]
                    all_result.append(output)
                else:
                    prompt = prompt_last_retry.format(
                        q=q,
                        sub_q=q_sub,
                        sub_question=sub_question,
                        process=process,
                        knowledge=knowledge,
                        ground_truth=ground_truth
                    )
                    output = generate_thinking(prompt,history)
                    label = verify(sub_question, ground_truth, output["final_answer"])

                    all_result.append(output)
                    if label:
                        all_label = True
                    else:
                        all_label = False

            else:
                while True:
                    prompt = prompt_mid.format(
                        q=q,
                        sub_q=q_sub,
                        sub_question=sub_question,
                        process=process,
                        knowledge=knowledge,
                        sub_a=sub_a
                    )
                    output = generate_thinking(prompt,history)
                    label = verify(sub_question, sub_a, output["candidate_answers"])
                    counter += 1
                    if label or counter >= max_iterations:
                        break

                all_result.append(output)
                history.append({"user": prompt, "assistant": output})


            
        polish_details = "\n\n".join(item["brainstorming_process"] for item in all_result)
        save_dict = example
        save_dict["entity2idx"] = entity2idx
        save_dict["guide_info"] = guide_info
        save_dict["polish_details"] = polish_details
        ans_prompt = prompt_generata_answer.format(q=q, details=polish_details, q_sub=q_sub_clean_list, ground_truth=ground_truth)
        output = find_json_output(generate_gpt4o(ans_prompt))['Reasoning_Process']
        save_dict["answer"] = output
        save_dict["polish_details_usable"] = all_label

        
        with open(output_file, 'a', encoding='utf-8') as outfile:
            outfile.write(json.dumps(save_dict) + "\n")
        
        return save_dict

    except Exception as e:
        with open(error_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps({"id": item_all['id'], "error": str(e)}) + '\n')
        return {"error": str(e), "id": item_all['id']}

if __name__ == "__main__":
    # Initialize
    mid2name, headrel2tail, tailrel2head, guide_info_dict, guide_info_id2entity_dict = load_data_files()
    
    # Load input data
    with open("./MHQ_sub_wih_details.jsonl", "r", encoding='utf-8') as f:
        MHQ_sub = [json.loads(l) for l in f]

    '''
    ##Example Sample of MHQ_sub_wih_details.jsonl##
    {
        "id": "idx",
        "q": "Multi-hop question",
        "q_sub_dict": [
            {
                "sub_problem_1": "",
                "reason": "",
                "reasoning_triplet": ""
            },
            ...
        ],
        "Questioned_Entity": "#0",
        "logic_subgraph": list of logical subgraph,
        "details": "Distilled from o1"
    }

    '''
    
    # Filter items with details
    remain_rel = [item for item in MHQ_sub if "details" in item]
    
    # Process items in parallel
    error_file = "error_list.jsonl"
    output_file = "polish_details_MHQA.jsonl"
    
    with tqdm_joblib(tqdm(desc="Processing items", total=len(remain_rel))):
        results = Parallel(n_jobs=16)(
            delayed(process_item)(
                item_all, mid2name, headrel2tail, 
                tailrel2head, guide_info_dict, guide_info_id2entity_dict
            ) for item_all in remain_rel
        )
    

            

