from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel

mode_path = '../huanhuan-chat/model/LLM-Research/Meta-Llama-3.1-8B-Instruct'
lora_path = '../huanhuan-chat/llama3_lora' # lora权重路径

# 加载tokenizer
#一个AutoTokenizer对象，用于将文本转换为token序列,AutoTokenizer会根据模型的类型（如BERT、GPT、Llama等）自动选择合适的分词器实现。例如，对于Llama模型，它会加载LlamaTokenizer
tokenizer = AutoTokenizer.from_pretrained(mode_path)

# 加载模型,AutoModelForCausalLM是用于处理因果语言模型的基类，它继承自AutoModel，并添加了因果语言模型的特定功能。
model = AutoModelForCausalLM.from_pretrained(mode_path, device_map="auto",torch_dtype=torch.bfloat16)

# 加载lora权重
model = PeftModel.from_pretrained(model, model_id=lora_path)

prompt = "你是谁？"
messages = [
    # {"role": "system", "content": "现在你要扮演皇帝身边的女人--甄嬛"},
    {"role": "user", "content": prompt}
]

#将对话格式的输入文本转换为模型可以处理的格式。得到一个字符串，表示转换后的文本模板。
#这个函数会根据对话格式将输入文本格式化为模型可以理解的格式，例如添加角色标记、分隔符等
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

#将文本转换为模型输入的张量格式，return_tensors="pt"：返回PyTorch张量
#返回值：一个字典，包含以下键：
#input_ids：文本的token ID序列。
#attention_mask：注意力掩码，用于指示哪些token是有效的。
#to('cuda')：将张量移动到GPU上，以便在GPU上进行计算。
model_inputs = tokenizer([text], return_tensors="pt").to('cuda')

#model.generate()：生成文本。
#model_inputs.input_ids：输入文本的token ID序列。
#max_new_tokens=512：生成文本的最大长度。
#do_sample=True：是否使用采样生成文本。
#top_p=0.9：采样时的nucleus采样参数。
#temperature=0.5：采样时的温度参数。
generated_ids = model.generate(
    model_inputs.input_ids,
    max_new_tokens=512,
    do_sample=True,
    top_p=0.9, 
    temperature=0.5, 
    repetition_penalty=1.1,
    eos_token_id=tokenizer.encode('<|eot_id|>')[0],
)

# 从generated_ids中提取生成的文本。
# 将生成的token ID序列转换为文本。
# skip_special_tokens=True：是否跳过特殊token（如<|eot_id|>）。
# [0]：返回第一个生成的文本。
# 遍历每对输入和输出token ID序列，从输出token ID序列中截取从输入token ID序列长度开始的部分。
# 返回值：一个列表，包含每个输入和输出token ID序列的生成文本。
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

# 将生成的token ID序列转换为文本。
# skip_special_tokens=True：是否跳过特殊token（如<|eot_id|>）。
# [0]：返回第一个生成的文本。
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print(response)