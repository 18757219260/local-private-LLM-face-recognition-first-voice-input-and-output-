# 本地私人 LLM 人脸识别与语音交互系统

本地大语言模型（LLM）、人脸识别、语音输入与输出。

---

## 目录

- [系统功能](#系统功能)
- [使用说明](#使用说明)
- [使用说明](#使用说明)

---

## 系统功能

- **人脸识别**  
  基于`opencv`的人脸识别，构建自身人脸库。

- **语音输入与输出**  
  采用 `百度语音识别api` 和 `edge_tts` 实现语音识别与语音合成，普通的异步输出、流式语音输出。

- **智能问答系统**  
  本地知识库（`knowledge.json`）与向量化检索`mk_faiss.py`里生成、更新、加载知识向量库。
  <br>调用本地ollama模型与知识向量库结合，可根据问答历史回答上下文（检索时间较长），也可以直接直接输出。
  <br>支持本地模型流式输出(速度更快)。

- **交互场景**  
  人脸识别后，（如“我是你的私人专属甘薯助手，你有什么问题吗？”）启动问答流程，语音提问，给出模型回答，并通过语音反馈。

---



### 环境要求

- Python 3.8 及以上
- 摄像头与麦克风（硬件设备）
- 网络环境（用于调用本地 LLM 服务，其接口地址为 `http://localhost:11434`）

### 依赖安装
```bash



1. 克隆项目仓库：
   
   git clone https://github.com/18757219260/local-private-LLM-face-recognition-first-voice-input-and-output.git
   cd local-private-LLM-face-recognition-first-voice-input-and-output
   pip install requirements.txt
```
## 使用说明
# 启动主程序

执行以下命令运行主交互程序：
```bash
python main.py
```
或者
```bash
python main_stream.py
```

