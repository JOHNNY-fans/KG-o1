from typing import List, Dict
from retrying import retry
from utils import find_json_output, generate_gpt4o


@retry(stop_max_attempt_number=5, wait_exponential_multiplier=1000, wait_exponential_max=10000)
def extract_topics(reasoning_triplet: List[str], mid2name: Dict[str, str],subquestion: str) -> List[str]:
    head, predicate, obj = reasoning_triplet
    topics = []
    
    if isinstance(head,list):
        topics += head
    else:
        if "#" not in head:
            topics.append(head)
    
    topics.append(predicate)
    
    if isinstance(obj,list):
        topics += obj
    else:
        if "#" not in obj:
            topics.append(obj)

    question_topic = extract_topics_from_subquestion(subquestion)
    
    return topics+question_topic


@retry(stop_max_attempt_number=5, wait_exponential_multiplier=1000, wait_exponential_max=10000)
def extract_topics_from_subquestion(subquestion):
    prompt = f'''
    You will be given a subquestion. Please extract the key topics from it. The topics should be clear nouns or noun phrases that represent the main concepts or entities in the subquestion.

    Subquestion: {subquestion}

    Output in JSON format:
    ```json
    {{
        "topics":["xxx","xxx"]
    }}
    ```
    '''
    output = find_json_output(generate_gpt4o(prompt))['topics']
    return output


def convert_reason_triple_raw(reasoning_triplet, entity2idx):
    inverted_dict = {}
    for key, value in entity2idx.items():
        if value not in inverted_dict:
            inverted_dict[value] = []
        inverted_dict[value].append(key)
    converted_triplet = []
    for element in reasoning_triplet:
        if isinstance(element, str) and "#" in element:
            if element in inverted_dict:
                converted_triplet.append(inverted_dict[element])
            else:
                converted_triplet.append(element)
        else:
            converted_triplet.append(element)
    return converted_triplet
