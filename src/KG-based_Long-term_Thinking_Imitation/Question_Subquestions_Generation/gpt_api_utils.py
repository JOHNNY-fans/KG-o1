from openai import OpenAI
import json
import re
from retrying import retry


client = OpenAI(api_key = 'YOUR_API_KEY')  

def extract_json_from_markdown(markdown: str) -> list:
    if not markdown:
        return []
    pattern = r"\{[^{}]*\}"
    matches = re.findall(pattern, markdown)
    result = []
    for match in matches:
        try:
            d = eval(match)
            if isinstance(d, dict):
                result.append(d)
        except:
            continue
    return result

def extract_final_answer(text: str):
    pattern = r'"final_answer": (\[.*?\])'
    match = re.search(pattern, text)
    if not match:
        return None
    try:
        import ast
        final_answer = ast.literal_eval(match.group(1))
        return final_answer[0] if isinstance(final_answer, list) else final_answer
    except:
        return None

@retry(stop_max_attempt_number=5, wait_exponential_multiplier=1000)
def call_gpt(prompt: str, history=None) -> str:
    messages = []
    if history:
        for h in history:
            messages.append({"role": "user", "content": h["user"]})
            messages.append({"role": "assistant", "content": h["assistant"]})
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )
    return response.choices[0].message.content
