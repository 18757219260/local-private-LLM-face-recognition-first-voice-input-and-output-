import sys
import os
sys.path.append(os.path.abspath("/home/wuye/vscode/chatbox"))
import asyncio
import os
import time
import logging
from face.face_recognize import FaceRecognizer
from qa_model.qa_model_easy import KnowledgeQA
from ASR.asr import ASRhelper
from interupt import InterruptibleTTS  


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("sweet_potato.log"), logging.StreamHandler()]
)

async def run_sweet_potato_system():
    """运行甘薯知识系统的交互过程，增加中断功能"""
    # 初始化模块
    asr = ASRhelper()
    qa = KnowledgeQA()
    tts = InterruptibleTTS(voice="zh-CN-XiaoyiNeural", rate="+0%", volume="+0%")

    print("\n🎉 甘薯知识助手已启动")
    
    # 播放欢迎信息
    await tts.text_to_speech("甘薯知识助手已启动，请问有什么可以帮助你的？")
    # 确保音频完全播放完毕，额外等待
    await asyncio.sleep(2.0)  # 缩短欢迎语的等待时间
    
    while True:
        try:
            # 步骤 1：语音转文本
            print("\n📢 等待语音输入...")
            
            # 确保在每次循环开始时都可用一个有效的输入流
            if not hasattr(tts, 'input_stream') or tts.input_stream is None:
                tts.setup_input_stream()
            
            # 执行语音识别
            question_data = asr.real_time_recognition()
            
            # 检查语音识别是否成功
            if question_data['err_no'] != 0:
                print(f"❌ 语音识别失败: {question_data['err_msg']}")
                await tts.text_to_speech("抱歉，我没听清楚，请再说一次。")
                await asyncio.sleep(1.0)  # 缩短错误信息的等待时间
                continue
                
            question = question_data['result'][0]
            if not question:
                continue  
                
            print(f"🧠 问题：{question}")

            # 步骤 2：问答模型处理
            answer = qa.ask(question)
            print(f"💬 答案：{answer}")

            # 步骤 3：文本转语音输出（可中断版本）
            was_interrupted, interrupt_question = await tts.speak_with_interrupt(answer)
            
            # 如果被中断，处理中断后的问题
            if was_interrupted and interrupt_question:
                print(f"⚠️ 回答被中断！用户新问题：{interrupt_question}")
                
                # 处理中断后的问题
                interrupt_answer = qa.ask(interrupt_question)
                print(f"💬 中断后的答案：{interrupt_answer}")
                
                # 输出中断后问题的答案
                await tts.speak_with_interrupt(interrupt_answer)

        except KeyboardInterrupt:
            print("\n🛑 停止交互")
            break
        except Exception as e:
            logging.error(f"系统错误: {e}", exc_info=True)
            print(f"❌ 出错：{e}")
            await tts.text_to_speech("抱歉，系统遇到了一些问题，请再试一次")
            # 同样为错误信息等待足够时间
            await asyncio.sleep(2.0)

    # 清理资源
    if hasattr(asr, 'stop_recording'):
        asr.stop_recording()
    if hasattr(tts, 'cleanup'):
        tts.cleanup()
        
    print("👋 感谢使用甘薯知识助手，再见！")
    await tts.text_to_speech("感谢使用甘薯知识助手，再见！")


async def main():
    """主函数：包含人脸认证和甘薯知识系统"""
    print("🚀 启动系统...")
    
    # 初始化TTS用于欢迎消息
    tts = InterruptibleTTS(voice="zh-CN-XiaoyiNeural", rate="+0%", volume="+0%")
    
    try:
        # 初始化人脸认证系统
        face_system = FaceRecognizer()
        if not face_system.initialize():
            print("系统初始化失败，程序退出")
            await tts.text_to_speech("系统初始化失败，请检查人脸模型")
            # 等待消息播放完毕
            await asyncio.sleep(2.0)
            return
        
        # 执行人脸认证
        print("开始人脸认证，请面向摄像头")
        await tts.text_to_speech("开始人脸认证，请面向摄像头")
        # 等待消息播放完毕
        await asyncio.sleep(2.0)
        
        auth_success, user_name = face_system.recognize_face()
        
        # 认证通过后运行甘薯知识系统
        if auth_success:
            welcome_message = f"欢迎你，{user_name}。已进入甘薯知识系统。"
            print(welcome_message)
            await tts.text_to_speech(welcome_message)
            # 等待欢迎消息播放完毕
            await asyncio.sleep(2.0)
            
            await run_sweet_potato_system()
        else:
            deny_message = "你是谁呀？我不认识你。系统将退出。"
            print("认证失败，拒绝访问")
            await tts.text_to_speech(deny_message)
            # 等待拒绝消息播放完毕
            await asyncio.sleep(2.0)
    finally:
        # 确保资源被正确清理
        if hasattr(tts, 'cleanup'):
            tts.cleanup()


if __name__ == "__main__":
    # 运行主程序
    asyncio.run(main())