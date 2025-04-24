import sys
import os
sys.path.append(os.path.abspath("/home/wuye/vscode/chatbox"))
import asyncio
from qa_model.qa_model_easy import KnowledgeQA
from ASR.asr import ASRhelper
from TTS.tts import TTSHelper 

async def run_interaction():
    # 初始化模块
    asr = ASRhelper()
    qa = KnowledgeQA()
    tts = TTSHelper(voice="zh-CN-XiaoyiNeural", rate="+0%", volume="+0%")

    while True:
        try:
            # 步骤 1：语音转文本（阻塞）
            print("\n📢 等待语音输入...")
            question = asr.real_time_recognition()
            
            if not question:
                continue  # 没识别到就跳过

            # 步骤 2：问答模型处理（非阻塞）
            # print(f"🧠 问题：{questionp["result"]}")
            answer = qa.ask(question)
            # print(f"💬 答案：{answer}")

            # 步骤 3：文本转语音输出（异步）
            await tts.text_to_speech("11"+answer)
            await asyncio.sleep(len(answer) * 0.1 + 0.5)

        except KeyboardInterrupt:
            print("\n🛑 停止交互")
            break
        except Exception as e:
            print(f"❌ 出错：{e}")

    # 清理资源
    asr.stop_recording()


if __name__ == '__main__':
    asyncio.run(run_interaction())
