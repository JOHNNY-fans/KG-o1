import json
import numpy as np
from retrying import retry
from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib
import os
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from concurrent.futures import ThreadPoolExecutor, as_completed
import string
from collections import Counter
import argparse
from openai import OpenAI
from rich import print
import random
import time

def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluation script with LLM and dataset parameters")
    parser.add_argument('--llm_name', type=str, required=True, help="Name of LLM to use, e.g., kg-o1-llama3.1-8b")
    parser.add_argument('--vllm_base_url', type=str, required=True, help="URL of LLM while using vllm")
    parser.add_argument('--dataset_type', type=str, required=True, help="Type of dataset to evaluate, e.g., hotpotqa")
    return parser.parse_args()

def find_json_output(prompt):
    return json.loads(prompt.split("```json")[-1].split("```")[0])

def normalize_answer(s):
    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_punc(lower(s)))

def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0, 0, 0)

    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall

def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))

def update_answer(metrics, prediction, gold):
    em = exact_match_score(prediction, gold)
    f1, prec, recall = f1_score(prediction, gold)
    metrics['em'] += float(em)
    metrics['f1'] += f1
    metrics['prec'] += prec
    metrics['recall'] += recall
    return em, prec, recall

def eval_cwq(prediction_file, gold_file):
    with open(prediction_file) as f:
        pred = [json.loads(l) for l in f]
    with open(gold_file) as f:
        gold = [json.loads(l) for l in f]

    predicitons = {"answer": {}}

    for item in pred:
        key = list(item.keys())[0]
        value = item[key]
        predicitons['answer'][key] = value

    metrics = {'em': 0, 'f1': 0, 'prec': 0, 'recall': 0}

    for dp in gold:
        cur_id = dp['question_id']
        answer = dp['answers_objects'][0]["spans"][0]

        if cur_id not in predicitons['answer'].keys():
            print(cur_id)
        else:
            em, prec, recall = update_answer(
                metrics, predicitons['answer'][cur_id].replace("_", " "), answer.replace("_", " "))

    N = len(pred)
    for k in metrics.keys():
        metrics[k] /= N
        metrics[k] = metrics[k]*100
    return metrics


def process_item(item, prompt_zero_shot):
    idx = item['question_id']
    question = item['question_text']
    documents = [
        {'title': item['title'], 'text': item['paragraph_text']}
        for item in item['contexts']
    ]
    formatted_documents = [
        f"Document [{i+1}](Title: {doc['title']}) {doc['text']}"
        for i, doc in enumerate(documents)
    ]

    content = prompt_zero_shot.format(question=question, context="\n".join(formatted_documents))
    output = generate_gpt4o(content)

    return {str(idx): output,"prompt":content}

def generateModelAns(prompt, tem, llm_name, vllm_base_url):
    client = OpenAI(
    base_url = vllm_base_url
    )
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        model=llm_name,
        max_tokens = 12000,
        temperature=tem
    )
    response = chat_completion.choices[0].message.content
    
    return response


def main():
    args = parse_arguments()
    llm_name = args.llm_name
    vllm_base_url = args.vllm_base_url
    dataset_types = args.dataset_type.split(',')

    for dataset_type in dataset_types:
        dataset_path = f'../data/{dataset_type}/test_subsampled.jsonl'
        output_file_path = f"../code/model_result/{dataset_type}_{llm_name}_0shot.jsonl"
        result_file_path = f"../code/result/result.jsonl"

        with open('../code/combine_ans.txt') as f:
            prompt_zero_shot = f.read().rstrip("\n")

        dev_subsampled_data = open(dataset_path).readlines()
        test_subsampled_data = [json.loads(item) for item in dev_subsampled_data]

        try:
            with open(output_file_path,"r") as f:
                exist_data = [json.loads(l) for l in f] 

            exist_ids = []
            
            for item in exist_data:
                key = [k for k in list(item.keys()) if "raw" not in k][0]
                exist_ids.append(key)
        except:
            exist_ids = []

        test_subsampled_data = []
        for item in dev_subsampled_data:
            if json.loads(item)['question_id'] not in exist_ids:
                test_subsampled_data.append(json.loads(item))

        for item in tqdm(test_subsampled_data):
            idx = item['question_id']
            question = item['question_text']
            if dataset_type != "kg_mhqa" and dataset_type != "mintqa":
                documents = [
                    {'title': item['title'], 'text': item['paragraph_text']}
                    for item in item['contexts']
                ]
                formatted_documents = [
                    f"Document [{i+1}](Title: {doc['title']}) {doc['text']}"
                    for i, doc in enumerate(documents)
                ]
            else:
                documents = [
                    {'text': item['paragraph_text']}
                    for item in item['contexts']
                ]
                formatted_documents = [
                    f"Document [{i+1}]:{doc['text']}"
                    for i, doc in enumerate(documents)
                ]

            content = prompt_zero_shot.format(question=question, context="\n".join(formatted_documents))

            count = 0
            tem = 0.001

            while True:
                output = generateModelAns(content,tem,llm_name)

                if "</output>" in output:
                    break

                count+=1

                if tem<0.91:
                    tem += 0.3

            output_last = output.split("The final answer is: ")[-1].split("\n</")[0]

            with open(output_file_path, 'a', encoding='utf-8') as output_file:
                output_file.write(json.dumps({str(idx): output_last,"raw_output":output}, ensure_ascii=False) + '\n')

        print(f"All tasks completed for dataset: {dataset_type}.")

        result = eval_cwq(output_file_path, dataset_path)
        save_dcit = {"model_type": llm_name, "dataset_type": dataset_type, "result": result}
        with open(result_file_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(save_dcit) + "\n")

if __name__ == "__main__":
    main()











