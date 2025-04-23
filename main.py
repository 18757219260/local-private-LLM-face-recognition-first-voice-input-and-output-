import asyncio
from qa_model_easy import KnowledgeQA
from asr import ASRhelper
from tts import RealTimeTTSHelper 

async def run_interaction():
    # 初始化模块
    asr = ASRhelper()
    qa = KnowledgeQA()
    tts = RealTimeTTSHelper(voice="zh-CN-XiaoyiNeural", rate="+0%", volume="+0%")

    while True:
        try:
            # 步骤 1：语音转文本（阻塞）
            print("\n📢 等待语音输入...")
            question = asr.real_time_recognition()
            if not question:
                continue  # 没识别到就跳过

            # 步骤 2：问答模型处理（非阻塞）
            # print(f"🧠 问题：{questionp["result"]}")
            answer = qa.ask(question['result'][0])
            print(f"💬 答案：{answer}")

            # 步骤 3：文本转语音输出（异步）
            await tts.text_to_speech("11"+answer)

        except KeyboardInterrupt:
            print("\n🛑 停止交互")
            break
        except Exception as e:
            print(f"❌ 出错：{e}")

    # 清理资源
    asr.stop_recording()


if __name__ == '__main__':
    asyncio.run(run_interaction())
