'''
可以参考使用langchain的memory，来实现对话记忆，但是需要自己实现记忆的保存和加载，以及记忆的清理。
在这人使用自定义的记录

ConversationBufferMemory：简单的对话历史缓冲区，适合短对话。
ConversationSummaryMemory：对历史对话做摘要，适合长对话。
ConversationBufferWindowMemory：只保留最近N轮对话。
ConversationTokenBufferMemory：按token数限制历史长度。
'''

from LLM import HuanhuanLLM

lora_path = "./llm-chat/llama3_lora"
model_name = "./llm-chat/model/LLM-Research/Meta-Llama-3.1-8B-Instruct"
# 加载lora权重
llm = HuanhuanLLM(model_name=model_name,lora_path=lora_path)
#不带lora
#llm_no_lora = HuanhuanLLM(model_name=model_name)

history = [] 

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit", "bye"]:
        break

    # 添加用户输入到历史
    history.append({"role": "user", "content": user_input})

    # 只保留最近5轮（10条消息）
    if len(history) > 10:
        history = history[-10:]

    text = llm.tokenizer.apply_chat_template(
        [{"role": "system", "content": "假设你是甄嬛，现在你正在和皇帝聊天，请根据皇帝的问题，回答皇帝的问题。"}] + history,
        tokenize=False,
        add_generation_prompt=True
    )
    assistant_reply = llm(text)
    print(f"Assistant: {assistant_reply}")

    history.append({"role": "assistant", "content": assistant_reply})

    if len(history) > 10:
        history = history[-10:]