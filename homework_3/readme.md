## 大语言模型部署体验
### 项目背景
在魔搭平台上部署并测试大语言模型
### 一、环境搭建
1.手动下载CPU版本的cuda
```bash
cd /opt/conda/envs
 #问题：没有那个文件或目录
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh-b-p /opt/conda
echo 'export PATH="/opt/conda/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
conda--version
```
2.创建和激活环境
```bash
conda create-n qwen_env python=3.10-y
source /opt/conda/etc/profile.d/conda.sh
conda activate qwen_env
```
### 二、安装依赖
1.配置基础环境
```bash
pip install \
 torch==2.3.0+cpu \
 torchvision==0.18.0+cpu \--index-url https://download.pytorch.org/whl/cpu
```
2.配置基础依赖
先检查pip能否正常联网
```bash
pip install-U pip setuptools wheel
```
再安装基础依赖（兼容 transformers 4.33.3 和 neuralchat）
```bash
pip install \
 "intel-extension-for-transformers==1.4.2" \
 "neural-compressor==2.5" \
 "transformers==4.33.3" \
 "modelscope==1.9.5" \
 "pydantic==1.10.13" \
"sentencepiece" \
 "tiktoken" \
 "einops" \
 "transformers_stream_generator" \
 "uvicorn" \
 "fastapi" \
 "yacs" \
 "setuptools_scm"
```
最后安装fschat（需要启用PEP517构建）
```bash
pip install tqdm huggingface-hub
```
3.可选：安装 tqdm、huggingface-hub 等增强体验
```bash
pip install tqdm huggingface-hub
```
### 三、大模型实践
1.切换到数据目录
```bash
cd /mnt/data
```
2.下载对应大模型到本地
```bash
git clone https://www.modelscope.cn/ZhipuAI/chatglm3-6b.git
git clone https://www.modelscope.cn/qwen/Qwen-7B-Chat.git
```
3.切换到工作目录
```bash
cd /mnt/workspace
```
4.编写推理脚本run_xxx_cpu.py
```bash
from transformers import TextStreamer, AutoTokenizer, AutoModelForCausalLM
model_name = "/mnt/data/Qwen-7B-Chat" # 本地路径
prompt = "请说出以下两句话区别在哪里？ 1、冬天：能穿多少穿多少 2、夏天：能穿多少穿多少"
    tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype="auto" # 自动选择 float32/float16（根据模型配置）
).eval()
inputs = tokenizer(prompt, return_tensors="pt").input_ids
streamer = TextStreamer(tokenizer)
outputs = model.generate(inputs, streamer=streamer, max_new_tokens=300)
```
5.运行脚本
```bash
python run_xxx_cpu.py
```
