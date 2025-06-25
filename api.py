from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
from pydantic import BaseModel

# 导入你的LLM类
from LLM import HuanhuanLLM

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

lora_path = "./huanhuan-chat/output/llama3_1_instruct_lora/checkpoint-5000"
model_name = "./huanhuan-chat/model/LLM-Research/Meta-Llama-3.1-8B-Instruct"
# 加载lora权重
llm = HuanhuanLLM(model_name=model_name,lora_path=lora_path)

class ChatRequest(BaseModel):
    message: str
    history: list = []

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/chat")
async def chat(request: ChatRequest):
    # 构建系统提示
    system_prompt = "你是甄嬛，一个以甄嬛传中甄嬛的语气说话的AI助手。你应该用典雅、含蓄而略带机锋的方式回答问题。"
    
    # 构建完整对话历史
    messages = [{"role": "system", "content": system_prompt}]
    
    # 添加历史对话
    for msg in request.history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    
    text = llm.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    response = llm(text)

    return {
        "response": response,
        "history": request.history + [
            {"role": "user", "content": request.message},
            {"role": "assistant", "content": response}
        ]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)