prompt_first = '''

Multi-hop Question: {q}  

Corresponding Sub-questions: {sub_q}  

The brainstorming_process for solving {sub_question} is as follows:  
{process}  

Knowledge Source:  
{knowledge}  

Real Answer of the sub-question:
{sub_a}

Based on the original brainstorming_process's writing style (i.e., title + discussion text) and the logical connections between the titles, use the knowledge provided to generate a brainstorming process and candidate_answers for addressing the sub-question. The requirements are as follows:  

1. Follow the format of the given brainstorming_process, including title + discussion text. The number of title + discussion text segments should exceed those in the original text.  

2. Make the brainstorming_process more comprehensive by discussing background information, such as the characteristics and attributes of relevant options, as well as different categories of institutions or organizations.  

3. Utilize all the provided knowledge in the generated text.  

4. Ensure the generated brainstorming_process is detailed.  

5. The number of title + discussion text segments in the generated brainstorming_process should be significantly more than those in the given brainstorming_process.  

6. At the part of candidate_answers , you need to provide several candidate answers in natural language, and these answers must include the given real answer of the sub-question.  

Output the final brainstorming process in JSON format. The structure should be as follows:  
Ensure the generated JSON is valid and can be correctly parsed using json.loads.  

```json
{{
  "brainstorming_process": "An unstructured natural language text",
  "candidate_answers":"several candidate answers in natural language"
}}
```
'''

# %%
prompt_mid = '''

Multi-hop Question: {q}  

Corresponding Sub-questions: {sub_q}  

The brainstorming_process for solving {sub_question} is as follows:  
{process}  

Knowledge Source:  
{knowledge}  

Real Answer of the sub-question:
{sub_a}

Taking into account your candidate_answers and brainstorming_process of the previous sub-questions, complete the following tasks:
Based on the original brainstorming_process's writing style (i.e., title + discussion text) and the logical connections between the titles, use the knowledge provided to generate a brainstorming process and candidate_answers for addressing the sub-question. The requirements are as follows:  

1. Follow the format of the given brainstorming_process, including title + discussion text. The number of title + discussion text segments should exceed those in the original text.  

2. Make the brainstorming_process more comprehensive by discussing background information, such as the characteristics and attributes of relevant options, as well as different categories of institutions or organizations.  

3. Utilize all the provided knowledge in the generated text.  

4. Ensure the generated brainstorming_process is detailed.  

5. The number of title + discussion text segments in the generated brainstorming_process should be significantly more than those in the given brainstorming_process.  

6. At the part of candidate_answers , you need to provide several candidate answers in natural language, and these answers must include the given real answer of the sub-question.  

Output the final brainstorming process in JSON format. The structure should be as follows:  
Ensure the generated JSON is valid and can be correctly parsed using json.loads.  

```json
{{
  "brainstorming_process": 'An unstructured natural language text',
  "candidate_answers":"several candidate answers in natural language"
}}
```
'''

prompt_last = '''

Multi-hop Question: {q}  

Corresponding Sub-questions: {sub_q}  

The brainstorming_process for solving {sub_question} is as follows:  
{process}  

Knowledge Source:  
{knowledge}  

Answer of the Multi-hop Question：{ground_truth}

Taking into account your candidate_answers and brainstorming_process of the previous sub-questions, complete the following tasks:
Based on the original brainstorming_process's writing style (i.e., title + discussion text) and the logical connections between the titles, use the knowledge provided to generate a brainstorming process for addressing the sub-question, generate the final answer to the Multi-hop Question. The requirements are as follows:  

1. Follow the format of the given brainstorming_process, including title + discussion text. The number of title + discussion text segments should exceed those in the original text.  

2. Make the brainstorming_process more comprehensive by discussing background information, such as the characteristics and attributes of relevant options, as well as different categories of institutions or organizations.  

3. Utilize all the provided knowledge in the generated text.  

4. Ensure the generated brainstorming_process is detailed.  

5. The number of title + discussion text segments in the generated brainstorming_process should be significantly more than those in the given brainstorming_process.  

6. At the final_answer part, your answer should be the given final answer( in natural language ) to the multi-hop problem.

Output the final brainstorming process in JSON format. The structure should be as follows:  
Ensure the generated JSON is valid and can be correctly parsed using json.loads.  

```json
{{
  "brainstorming_process": "An unstructured natural language text",
  "final_answer":"the final answer"
}}
```
'''



# %%
prompt_last_retry = '''

Multi-hop Question: {q}  

Corresponding Sub-questions: {sub_q}  

The brainstorming_process for solving {sub_question} is as follows:  
{process}  

Knowledge Source:  
{knowledge}  

Answer of the Multi-hop Question：{ground_truth}

Taking into account your candidate_answers and brainstorming_process of the previous sub-questions, complete the following tasks:
Based on the original brainstorming_process's writing style (i.e., title + discussion text) and the logical connections between the titles, use the knowledge provided to generate a brainstorming process for addressing the sub-question, generate the final answer to the Multi-hop Question. The requirements are as follows:  

1. Follow the format of the given brainstorming_process, including title + discussion text. The number of title + discussion text segments should exceed those in the original text.  

2. Make the brainstorming_process more comprehensive by discussing background information, such as the characteristics and attributes of relevant options, as well as different categories of institutions or organizations.  

3. Utilize all the provided knowledge in the generated text.  

4. Ensure the generated brainstorming_process is detailed.  

5. The number of title + discussion text segments in the generated brainstorming_process should be significantly more than those in the given brainstorming_process.  

6. At the final_answer part, your answer should be the given final answer( in natural language ) to the multi-hop problem.

Note: There may be problems with the object of the multi-hop question, so you can analyze why the given answer is inconsistent with the answer obtained by brainstorming, and finally output the given answer.

Output the final brainstorming process in JSON format. The structure should be as follows:  
Ensure the generated JSON is valid and can be correctly parsed using json.loads.  

```json
{{
  "brainstorming_process": "An unstructured natural language text",
  "final_answer":"the given answer"
}}
```
'''


prompt_generate = '''
## Multi-hop Question: 
{q}  

## Corresponding Sub-questions: 
{q_sub}  

## The knowledge needed to solve the Sub-questions:
{knowledge}  

## Answers of Sub-questions:
{answers}

## Answer of the Multi-hop Question：{ground_truth}

## Example:
Convenience in Communication
Social media provides a quick and efficient way for people to communicate. It allows users to stay connected across cities or even countries through texts, calls, and video chats. However, while convenience reduces physical barriers, does it compromise the quality of communication? Can it truly replace the depth of face-to-face interactions?

Expansion of Social Circles
Social media creates opportunities to connect with a wider network of people, including those who share similar interests. This helps individuals meet new friends or professional contacts. Yet, are these connections meaningful, or are they mostly superficial and lacking emotional depth?

Reduction in Face-to-Face Interaction
With the rise of social media, people increasingly prefer online communication over in-person meetings, potentially weakening real-world relationships. Can digital interactions fully meet emotional needs, or does the lack of physical presence contribute to feelings of isolation?

Triggering Social Anxiety
The curated and often idealized content on social media platforms can foster unhealthy comparisons, leading to feelings of inadequacy or anxiety. Can users manage this pressure through self-regulation, or do platforms and society need to offer stronger safeguards against these effects?
...

## Refer to the writing style of given example, use the provided knowledge to generate a brainstorming process to solve the multi-hop problem, and generate the final answer to the multi-hop problem. The requirements are as follows:

1. You should first generate sub-brainstorming process corresponds to each sub_question in format of multiple title + discussion text fragments. 

2. Make each sub-brainstorming process more comprehensive by discussing background information (such as the characteristics and attributes of relevant options, and different categories of institutions or organizations) using knowledge corresponding to sub-question.

3. The number of title + discussion text segments in Each sub-brainstorming process should be more than 2.

4. title + discussion text fragments of Each sub-brainstorming process should include all the knowledge corresponding to sub-question.(The title should be short sentence about knowledge)

5. The overall brainstorming process should logically connects the sub-brainstorming process and get the answer to the final multi-hop question.

6. In the final_answer section, your answer should be the given answer to the multi-hop question (expressed in natural language).

Note:
1:Do not use a clear title to solve the problem. For example, sub_question 1. The final output text contains multiple subtitles and discussion text. 
2:There may be problems with the object of the multi-hop question, so you can analyze why the given answer is inconsistent with the answer obtained by your brainstorm process, and finally output the given answer.

## Output the final brainstorming process in JSON format.  
Ensure the generated JSON is valid and can be correctly parsed using json.loads.  
The structure should be as follows: 
```json
{{
  "overall_brainstorming_process": "An natural language text",
  "final_answer":"the final answer"
}}
```
'''

prompt_generate_polish = '''
Take the provided input text and refine it to create a cohesive and well-flowing version. Remove the main headings and reorganize the content into smaller, meaningful sections with subheadings. The final output should consist of subheadings followed by their respective refined text, maintaining a logical and polished structure. Ensure clarity, readability, and a seamless connection between ideas.a

Input:{details}

Output in JSON format.  
Ensure the generated JSON is valid and can be correctly parsed using json.loads.
The structure should be as follows:   

```json
{{
  "polish_texts": "A natural language text",
}}
```
'''



prompt_generata_answer = '''
Task Description:
Given an input, including a multi-hop question, the process of solving the problem, and the correct answer, your task is to generate a process for answering the multi-hop question based on the information provided. The process should include the solution process of the sub-questions and the final response, and the final response should include the final answer (in natural language format).

Input:

Multi-hop Question: {q}
Solution Process: {details}
Sub_questions: {q_sub}
Correct Answer: {ground_truth}
Output Format:

Output in JSON format.  
Ensure the generated JSON is valid and can be correctly parsed using json.loads.
The structure should be as follows:   

```json
{{
  "Reasoning_Process": "Briefly describe the reasoning logic and key steps, and provide the final answer"
}}
```
'''