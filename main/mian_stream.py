import sys
import os
sys.path.append(os.path.abspath("/home/wuye/vscode/chatbox"))
import asyncio
from qa_model.qa_model_easy import KnowledgeQA
from ASR.asr import ASRhelper
from TTS.tts_stream import TTStreaming
import time
import re

async def main():
    # 初始化 QA 和 TTS
    qa = KnowledgeQA()
    tts = TTStreaming(voice="zh-CN-XiaoyiNeural")
    asr = ASRhelper()
    
    try:
        while True:
            buffer = ""
            min_length = 5  # 最小文本长度
            trigger_length = 40  # 触发语音的字符数
            timeout = 0.5  # 0.5秒超时触发语音
            last_chunk_time = time.time()
            
        
            print("请开始说话...")
            question = asr.real_time_recognition()
            
            if not question :
                print("无输入跳过")
                await asyncio.sleep(1) 
                continue
          

            
            question = question["result"][0]

            async def text_producer():
                """从 ask_stream 获取文本并放入队列"""
                async for chunk in qa.ask_stream(question):
                    print(f"🧠 输出块: {chunk}")
                    yield chunk
            
            async def text_consumer():
                """处理文本并触发语音"""
                nonlocal buffer, last_chunk_time
                async for chunk in text_producer():
                    chunk = tts.preprocess_text(chunk)  # 预处理文本
                    buffer += chunk
                    last_chunk_time = time.time()

                    # 触发语音：达到 trigger_length 或超时
                    if len(buffer) >= trigger_length or (time.time() - last_chunk_time) >= timeout:
                        if len(buffer) >= min_length:
                            print(f"语音输出: {buffer}")
                            await tts.speak('11' + buffer)
                            # 估算播放时间（假设每字符0.1秒）
                            await asyncio.sleep(len(buffer) * 0.1 + 0.5)
                            buffer = ""  # 清空缓冲区

                # 处理剩余文本
                if buffer and len(buffer) >= min_length:
                    print(f"语音输出（剩余）: {buffer}")
                    await tts.speak('11' + buffer)
                    await asyncio.sleep(len(buffer) * 0.1 + 0.5)

            try:
                await text_consumer()
            except Exception as e:
                print(f"处理时出错: {e}")
            
            # 循环间延迟，确保用户有时间准备
            await asyncio.sleep(1)

    finally:
        await tts.shutdown()
        asr.stop_recording()
        print("程序退出，资源已清理")

if __name__ == "__main__":
    asyncio.run(main())