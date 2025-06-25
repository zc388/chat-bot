from langchain.llms.base import LLM
from typing import Any, List, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

model_name = "/data/zhangjian/Personal/chat/huanhuan-chat/model/LLM-Research/Meta-Llama-3.1-8B-Instruct"

class HuanhuanLLM(LLM):
    tokenizer: Any = None
    model: Any = None
    
    def __init__(self, model_name: str = model_name,lora_path=None):
        super().__init__()
        if not lora_path:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name).to('cuda:0')
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name).to('cuda:0')
            self.model = PeftModel.from_pretrained(self.model, model_id=lora_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        print("完成本地模型的加载")
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None,
              run_manager: Optional[CallbackManagerForLLMRun] = None
              ) -> str:
        
        model_inputs=self.tokenizer([prompt],return_tensors='pt').to('cuda')
        #input_ids：文本的token ID序列。
        input_ids=model_inputs.input_ids
        #生成文本,返回值：generated_ids：生成的token ID序列。
        generated_ids=self.model.generate(input_ids,max_new_tokens=256,pad_token_id=self.tokenizer.eos_token_id)
        # 遍历每对输入和输出token ID序列，从输出token ID序列中截取从输入token ID序列长度开始的部分。
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(input_ids, generated_ids)
        ]
        #返回值：一个列表，包含每个输入和输出token ID序列的生成文本。
        responses = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        # print(responses)

        return responses[0]

    def _llm_type(self) -> str:
        return "huanhuan-llm"