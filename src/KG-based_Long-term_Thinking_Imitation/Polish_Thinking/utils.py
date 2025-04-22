import json
from openai import OpenAI
from retrying import retry

def find_json_output(prompt):
    return json.loads(prompt.split("```json")[-1].split("```")[0])


client = OpenAI(api_key = 'YOUR_API_KEY')

@retry(stop_max_attempt_number=5, wait_exponential_multiplier=1000)
def generate_gpt4o(prompt: str, history=None) -> str:
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

