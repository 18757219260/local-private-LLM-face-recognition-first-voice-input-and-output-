import asyncio
import os
import time
from face.face_recognize import FaceRecognizer
from qa_model.qa_model_easy import KnowledgeQA
from ASR.asr import ASRhelper
from TTS.tts import TTSHelper

async def run_sweet_potato_system():
    """运行甘薯知识系统的交互过程"""
    # 初始化模块
    asr = ASRhelper()
    qa = KnowledgeQA()
    tts = TTSHelper(voice="zh-CN-XiaoyiNeural", rate="+0%", volume="+0%")

    print("\n🎉 甘薯知识助手已启动")
    
    # 播放欢迎信息
    await tts.text_to_speech("11甘薯知识助手已启动，请问有什么可以帮助你的？")
    # 确保音频完全播放完毕，额外等待
    await asyncio.sleep(5.0)  # 给欢迎语足够的播放时间
    
    while True:
        try:
            # 步骤 1：语音转文本
            print("\n📢 等待语音输入...")
            
            # 清空可能的音频缓冲
            if hasattr(asr, 'stream') and asr.stream:
                try:
                    # 等待一小段时间确保音频播放完全结束
                    time.sleep(0.5)
                    # 清空缓冲区中的数据
                    while hasattr(asr.stream, 'get_read_available') and asr.stream.get_read_available() > 0:
                        asr.stream.read(asr.CHUNK, exception_on_overflow=False)
                except Exception as e:
                    print(f"清理音频缓冲区时出错: {e}")
            
            # 执行语音识别
            question_data = asr.real_time_recognition()
            question = question_data['result'][0]
            if not question:
                continue  # 没识别到就跳过
            print(f"🧠 问题：{question}")

            # 步骤 2：问答模型处理
            answer = qa.ask(question)
            print(f"💬 答案：{answer}")

            # 步骤 3：文本转语音输出
            await tts.text_to_speech('11'+answer)
            
            # 增加等待时间，确保语音完全播放完毕
            # 基于文本长度动态调整等待时间
            wait_time = len(answer) * 0.04 + 3.0  
            print(f"等待语音播放完成，等待时间: {wait_time:.2f}秒")
            await asyncio.sleep(wait_time)

        except KeyboardInterrupt:
            print("\n🛑 停止交互")
            break
        except Exception as e:
            print(f"❌ 出错：{e}")
            await tts.text_to_speech("11抱歉，系统遇到了一些问题，请再试一次")
            # 同样为错误信息等待足够时间
            await asyncio.sleep(4.0)

    # 清理资源
    asr.stop_recording()
    print("👋 感谢使用甘薯知识助手，再见！")
    await tts.text_to_speech("11感谢使用甘薯知识助手，再见！")


async def main():
    """主函数：包含人脸认证和甘薯知识系统"""
    print("🚀 启动系统...")
    
    # 初始化TTS用于欢迎消息
    tts = TTSHelper(voice="zh-CN-XiaoyiNeural", rate="+0%", volume="+0%")
    
    # 初始化人脸认证系统
    face_system = FaceRecognizer()
    if not face_system.initialize():
        print("系统初始化失败，程序退出")
        await tts.text_to_speech("11系统初始化失败，请检查人脸模型")
        # 等待消息播放完毕
        await asyncio.sleep(3.0)
        return
    
    # 执行人脸认证
    print("开始人脸认证，请面向摄像头")
    await tts.text_to_speech("11开始人脸认证，请面向摄像头")
    # 等待消息播放完毕
    await asyncio.sleep(3.0)
    
    auth_success, user_name = face_system.recognize_face()
    
    # 认证通过后运行甘薯知识系统
    if auth_success:
        welcome_message = f"11欢迎你，{user_name}。已进入甘薯知识系统。"
        print(welcome_message)
        await tts.text_to_speech(welcome_message)
        # 等待欢迎消息播放完毕
        await asyncio.sleep(5.0)
        
        await run_sweet_potato_system()
    else:
        deny_message = "11你是谁呀？我不认识你。系统将退出。"
        print("认证失败，拒绝访问")
        await tts.text_to_speech(deny_message)
        # 等待拒绝消息播放完毕
        await asyncio.sleep(5.0)


if __name__ == "__main__":
    # 运行主程序
    asyncio.run(main())