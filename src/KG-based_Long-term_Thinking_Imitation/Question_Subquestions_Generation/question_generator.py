import json
from gpt_api_utils import call_gpt, extract_json_from_markdown
from utils import normalize_answer, resolve_triples, find_most_similar_key
from retrying import retry

example = '''
    ##Example1##
    Questioned_Entity: #2  
    Triplet_Information:
    [('#2', '/military/military_command/military_conflict', '#0'),
    ('#1', '/military/military_command/military_conflict', '#0'),
    (['War_of_the_Fourth_Coalition', 'Ottoman_wars_in_Europe'],
    '/time/event/locations','#1'),
    ('#0', '/time/event/locations', 'Middle_East')]

    Answer: Erich_Ludendorff


    ##Example1 Output##
    {  
        "MHQuestion": "He and the nation that once served as the backdrop for both the War_of_the_Fourth_Coalition and the Ottoman_wars_in_Europe commanded the same conflict in the Middle_East. Who is he?",  
        "Questioned_Entity": "#2"  
    }

    ##Example2##
    Questioned_Entity: #1  
    Triplet_Information:
    [('#0', '/music/artist/origin', '#1'), ('#0', '/music/artist/track_contributions', ['Cello', 'Synthesizer']), ('#0', '/music/artist/track_contributions', '#2'), (['Experimental_rock', 'Jazz_fusion'], '/music/genre/artists', '#0'), ('#2', '/music/genre/artists', 'Leon_Russell')]

    Answer: ['Rochester']


    ##Example2 Output##
    {  
        "MHQuestion": "He, an Experimental_rock and Jazz_fusion musician who has contributed tracks featuring the Cello and Synthesizer and worked with another artist known to fans of Leon_Russell, is originally from it. Which place is it?",  
        "Questioned_Entity": "#1"  
    }
'''

def generate_multihop_question(idx, q_entity, logic_subgraph, answer, entity2idx):
    gen_prompt = f'''
    Please generate a multi-hop reasoning question based on the triplet information ,the specified question subject and , with the following requirements:

    1. Identifiers in the question must appear as pronouns (e.g., “she,” “he”) or as “pronoun + relationship” (e.g., “his wife”) rather than using node names. Moreover, the pronoun form of the questioned entity should be determined by the answer.
    2. The resulting multi-hop problem must contain all the information from the triplet, especially the non-Identifiers nodes in the triplet.
    3. Placeholders like "#0" or other abstract tags can't be included in the question.
    4. Ensure that the generated question is logically clear and semantically fluent.
    5. Ensure that the answer does not appear in the question.

    {example}  

    ##Triplet Information##
    Questioned_Entity: {q_entity}
    Triplet_Information:
    {logic_subgraph}

    Answer:{answer}

        
    Please output in the following JSON format, Ensure that the JSON output is properly formatted with double quotes around keys and string values, without trailing commas, and follows standard JSON syntax to be compatible with json.loads().:  
    ```json
    {{     
        "MHQuestion": xxx,      
        "Questioned_Entity": xxx  
    }}
    ```
    ''' 
    output = call_gpt(gen_prompt)
    q_dict = json.loads(output.split("```json")[-1].split("```")[0])
    q_dict.update({
        'id': idx,
        'logic_subgraph': logic_subgraph,
        'ground_truth': answer,
        'q_entity': q_entity,
        'OriginQuestion': q_dict['MHQuestion']
    })
    return q_dict

@retry(stop_max_attempt_number=2)
def decompose_question(q_dict):
    logic_sg = q_dict['logic_subgraph']
    q = q_dict['MHQuestion']
    prompt = f'''
    Please analyze the following multi-step reasoning question and break it down into multiple sub-problems. 

    Each sub-problem must correspond to one reasoning triplet from the provided reasoning path. Ensure the number of sub-problems matches the number of reasoning triplets, and there is a one-to-one correspondence between them.

    For each sub-problem:
    1. Provide a clear and concise description of the sub-problem.
    2. Sub-problem should not contain placeholders like "#0" or other abstract labels.
    3. Explain the reasoning and thought process for breaking down the original question into this specific sub-problem.
    4. Assign the corresponding reasoning triplet from the provided reasoning path.

    ### Multi-step Reasoning Problem ###
    {q}

    ### Reasoning Path ###
    {logic_sg}

    ### Output Requirements ###
    Ensure that the JSON output is properly formatted with double quotes around keys and string values, without trailing commas, and follows standard JSON syntax to be compatible with json.loads(). Ensure that the "reasoning triplet" field in each sub-problem is selected directly from the reasoning path provided:

    ```json
    [
        {{
            "sub_problem_1": "xxx",
            "reason": "xxx",
            "reasoning_triplet":"xxx"
        }},
        {{
            "sub_problem_2": "xxx",
            "reason": "xxx",
            "reasoning_triplet":"xxx"
        }}
    ]
    ```
    '''
    output = call_gpt(prompt)
    return json.loads(output.split("```json")[-1].split("```")[0])

def verify_subquestions_match(subquestions, logic_subgraph, entity2idx):
    resolved = resolve_triples(logic_subgraph, entity2idx)
    keys = []
    if len(subquestions) != len(logic_subgraph):
        return False
    for sq in subquestions:
        input_str = str(sq['reasoning_triplet'])
        most_similar_key, _ = find_most_similar_key(input_str, resolved)
        keys.append(most_similar_key)
    return set(keys) == set(resolved.keys())
