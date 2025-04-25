# 本地私人 LLM 人脸识别与语音交互系统



---

## 目录

- [功能](#功能)
- [配置](#配置)

---

## 系统功能

- **人脸识别**  
  `opencv` 人脸识别

- **语音输入与输出**  
   `百度语音识别API` 和 `edge_tts` 语音识别、语音合成
  - 普通的异步语音输出
  - 流式语音输出
  - 中断


---


## 配置
```bash

pip install -r requirements.txt

sudo apt-get install mpg123
sudo apt-get install python3-pyaudio
sudo apt-get install libportaudio2

```

执行以下命令运行主交互程序：

```bash
# 异步语音输出
python main.py

# 流式输出版本
python main_stream.py 

# 支持中断的版本
python main_interupt.py
```
