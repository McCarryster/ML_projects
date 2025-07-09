from huggingface_hub import InferenceClient

repo_id = "mccarryster/com-gen-llama3.1-8B"

llm_client = InferenceClient(model=repo_id, timeout=120)

def call_llm(inference_client: InferenceClient, prompt: str):
    response = inference_client.text_generation(
        prompt,
        max_new_tokens=200,
    )
    return response[0]["generated_text"]

prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a social media assistant. Generate a comment. Consider the Reaction of users and the tone of the Post.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Post: Мемолога на зарплату 55 тысяч рублей в месяц ищет компания в Туле. У сотрудника будет всего одна задача:  отправлять мемы в рабочие чаты, чтобы помогать коллегам справиться с выгоранием
Reactions: like:2874, love:343, poop:193, pray:109, clown:15969, enjoy:305, laugh:3938, vomit:413, dislike:54
Position: 4
Desired reactions: clown:64<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""

response = call_llm(llm_client, prompt)
print(response)
