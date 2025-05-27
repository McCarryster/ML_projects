from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "mccarryster/com-gen-llama3.1-8B"
cache_dir = "/cache/dir/"

model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir, device_map="cpu")
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)


prompt = """
You are a social media assistant. Generate a comment. Consider the Reaction of users and the tone of the Post.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Post: Мемолога на зарплату 55 тысяч рублей в месяц ищет компания в Туле. У сотрудника будет всего одна задача:  отправлять мемы в рабочие чаты, чтобы помогать коллегам справиться с выгоранием
Reactions: like:2874, love:343, poop:193, pray:109, clown:15969, enjoy:305, laugh:3938, vomit:413, dislike:54
Position: 1054
Desired reactions: clown:647<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""

def generate_text(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens=200,         # Maximum length of generated comment to keep responses concise and relevant
            temperature=0.7,            # Controls randomness: lower values make output more focused, 0.7 balances creativity and coherence
            repetition_penalty=1.2,     # Discourages the model from repeating the same words or phrases, improving comment diversity
            no_repeat_ngram_size=2,     # Prevents repeating any 2-word sequences, reducing redundant phrasing in comments
            length_penalty=1.0,         # Neutral setting that neither favors longer nor shorter comments, maintaining natural length
            do_sample=True,             # Enables sampling to generate varied and less deterministic outputs rather than greedy decoding
            top_k=50,                   # Restricts token selection to the 50 most probable next tokens, limiting unlikely word choices
            top_p=1.0,                  # Uses nucleus sampling with full probability mass (1.0), allowing diverse but plausible token choices
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

result = generate_text(prompt)
print(f"Generated text:\n{result}")