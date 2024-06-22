from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
from torch.cuda.amp import autocast


# Instead of using model.chat(), we directly use model.generate()
# But you need to use tokenizer.apply_chat_template() to format your inputs as shown below
device = "cuda" # the device to load the model onto
model_path = "Qwen/Qwen1.5-4B"
# Now you do not need to add "trust_remote_code=True"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_path,padding_side='left')
allocated_memory = torch.cuda.memory_allocated(device)
print("model allocated memory:", allocated_memory)

def get_response(model,tokenizer,contents,num_return_sequences=1):
    print("gogo")
    prompt = "[instruct]:提炼出案件中的关键点，这些关键点应当具备以下特征：1. 易于与其他案件区分。2. 有助于对案件进行准确定性。具体要求：1. 提供涉案人员的具体特征，如年龄、职业、作案手段等。2. 描述案件的独特性，例如使用的技术手段、诈骗金额、受害人的类型等。3. 指出案件中的关键证据或线索，如聊天记录、银行转账记录等。4. 总结案件的影响或后果，例如对社会的影响、受害人损失等。"
    messages = [{"role": "user", "content":"[content]:"+ content+prompt} for content in contents]
    texts = [tokenizer.apply_chat_template([message], tokenize=False, add_generation_prompt=True) for message in messages]
    model_inputs = tokenizer(texts, return_tensors="pt",padding = True).to(device)
    inputs_len  = model_inputs["input_ids"].shape[1]
    # print(model_inputs["input_ids"].shape)
    pad_token_id = tokenizer.eos_token_id
    with autocast():
        generated_ids = model.generate(model_inputs.input_ids,
                                    max_new_tokens=128,
                                    do_sample=True,
                                    pad_token_id=pad_token_id,
                                    top_k=50,
                                    top_p=0.95,
                                    temperature=0.6,
                                    attention_mask=model_inputs.attention_mask,
                                    num_return_sequences=num_return_sequences,
                                    eos_token_id= pad_token_id,
                                    repetition_penalty=1.2,
                                    use_cache=True
                                    )
    del model_inputs
    # print(type(generated_ids))
    # print(generated_ids.shape)
    generated = [output_ids[inputs_len:] for output_ids in generated_ids]
    # print(generated[0].shape)
    del generated_ids
    responses = tokenizer.batch_decode(generated)
    del generated
    responses = [response.split('<|im_end|>')[0].strip() for response in responses]
    
    
    # for res in responses:
    #     print(res)
    #     print("-"*10)
    torch.cuda.empty_cache()
    allocated_memory = torch.cuda.memory_allocated(device)
    print("weight allocated memory:", allocated_memory)
    return responses



if __name__ == "__main__":
    with open ("dataset/train.json") as f:
        configs = json.load(f)
    num_return_sequences = 2
    step = 4
    for start in range(16109, len(configs), step):
        contents = []
        labels = []
        end = start + step
        print(f"Augmenting: {start}/{len(configs)}")
        for config in configs[start:end]:
            
            content = config["案情描述"]
            label = config["案件类别"]
            contents.append(content)
            for i in range(num_return_sequences):
                labels.append(label)
        res = get_response(model, tokenizer, contents,num_return_sequences=num_return_sequences)
        new_cases = [{"案情描述":r,"案件类别":l}for r,l in zip(res,labels)]
        with open('dataset/augment.json', 'r', encoding='utf-8') as file:
            existing_cases = json.load(file)
        # print(len(existing_cases))
        
        all_cases = existing_cases + new_cases
        with open('dataset/augment.json', 'w', encoding='utf-8') as file:
            json.dump(all_cases, file, ensure_ascii=False, indent=4)

    