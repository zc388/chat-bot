# 对话助手 (Chat Assistant)
低智力，答非所问版本。
<p align="center">
  <img src="static/assistant_avatar.png" alt="甄嬛对话助手" width="200">
</p>

## 项目简介

基于Llama-3.1-8B-Instruct模型微调的聊天机器人，通过LoRA技术使用《甄嬛传》中甄嬛的台词进行训练，使模型能够以甄嬛的语气和风格进行对话。该助手能够模仿甄嬛典雅、含蓄而略带机锋的说话方式，为用户提供独特的对话体验。

## 功能特点

- 以甄嬛的语气和风格回应用户问题
- 支持上下文对话记忆
- 提供Web界面进行交互
- 支持命令行交互模式

## 技术栈

- **基础模型**: Meta-Llama-3.1-8B-Instruct
- **微调方法**: LoRA (Low-Rank Adaptation)
- **后端框架**: FastAPI
- **前端技术**: HTML, CSS, JavaScript
- **模型加载**: Hugging Face Transformers, PEFT
- **对话框架**: LangChain

## 环境要求

- Python 3.8+
- CUDA 11.7+ (用于GPU加速)
- 16GB+ GPU内存 (推荐)
- 32GB+ 系统内存

## 安装步骤

1. 克隆仓库
```bash
git clone https://github.com/yourusername/zhen-huan-chat.git
cd zhen-huan-chat
```

2. 安装依赖
```bash
pip install -r requirements.txt
```

3. 下载模型
   - 基础模型: Meta-Llama-3.1-8B-Instruct (需要访问权限)
   - 或者使用已微调的模型: 将模型文件放置在 `./llama3_lora` 目录下

## 使用方法

### Web界面

1. 启动Web服务
```bash
python api.py
```

2. 在浏览器中访问 `http://localhost:8000`

### 命令行交互

```bash
python test_LLM.py
```

## 模型训练

本项目使用LoRA技术对Llama-3.1-8B-Instruct模型进行微调，训练数据来源于《甄嬛传》中甄嬛的台词。

LoRA配置:
- rank (r): 8
- alpha: 32
- dropout: 0.1
- 目标模块: ["o_proj", "k_proj", "q_proj", "gate_proj", "v_proj", "up_proj", "down_proj"]

## 项目结构

```
.
├── LLM.py                  # 自定义LLM模型类
├── api.py                  # FastAPI Web服务
├── test_LLM.py             # 命令行测试脚本
├── llama3_lora/            # 微调后的LoRA权重
├── static/                 # 静态资源文件
│   ├── style.css           # 样式表
│   ├── assistant_avatar.png # 助手头像
│   └── ...
└── templates/              # HTML模板
    └── index.html          # 主页面
```

## 示例对话

用户: "今日天气如何？"

甄嬛: "窗外春光正好，花影婆娑，想是个晴朗的好日子。妾身近日足不出户，倒是不曾细察天色，若郎君欲知详情，不妨亲自掀帘一看。"

## 许可证

[MIT License](LICENSE)

## 致谢

- Meta AI 提供的Llama-3模型
- 《甄嬛传》剧组提供的优质台词素材 
