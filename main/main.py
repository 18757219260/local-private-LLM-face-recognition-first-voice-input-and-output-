import sys
import os
sys.path.append(os.path.abspath("/home/wuye/vscode/chatbox"))
import asyncio
import os
import time
import logging
import signal
from face.face_recognize import FaceRecognizer
from qa_model.qa_model_easy import KnowledgeQA
from ASR.asr import ASRhelper
from TTS.tts import TTSHelper

# 配置日志 - 美化日志格式
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)8s │ %(message)s",
    handlers=[logging.FileHandler("chatbox.log"), logging.StreamHandler()]
)

# 全局退出事件
shutdown_event = asyncio.Event()

async def run_sweet_potato_system(user_name):
    """运行甘薯知识系统的交互过程"""
    print("\n🎉✨ 甘薯知识助手已启动 ✨🎉")
    
    # 先创建必要组件
    tts = None
    asr = None
    qa = None
    
    try:
        # 初始化TTS
        print("🔊 初始化语音系统...")
        tts = TTSHelper(voice="zh-CN-XiaoyiNeural", rate="+0%", volume="+0%")
        
        # 使用简单打印方式初始通知，防止TTS错误中断整个流程
        print("🎤 正在初始化语音识别系统...")
        await asyncio.sleep(0.5)  # 短暂停顿
        
        # 确保TTS正常
        try:
            await tts.text_to_speech("11正在初始化系统...", wait=True)
        except Exception as e:
            logging.error(f"TTS初始化测试失败: {e}")
            print("⚠️ 警告: 语音合成服务不可用，将以文本方式提供反馈")
        
        # 初始化ASR（语音识别）
        print("🎤 初始化语音识别...")
        asr = ASRhelper()
        
        # 初始化QA模型 - 这是最耗时的操作
        print("🧠 正在加载知识模型...")
        
        # 在单独的任务中初始化QA模型
        qa_init_task = asyncio.create_task(initialize_qa_model())
        
        # 等待QA模型初始化完成
        qa = await qa_init_task
        
        print("\n✅ 知识模型加载完成！系统准备就绪")
        print(f"👋 欢迎你，{user_name}。正在进入甘薯知识系统。")
        print("\n" + "═" * 150)
        print(f"{'🌟 甘薯知识问答系统已启动 🌟':^150}")
        print(f"{'👤 用户: ' + user_name:^150}")
        print(f"{'⌨️  按 Ctrl+C 退出':^150}")
        print("═" * 150 + "\n")
        
        try:
            await tts.text_to_speech("11甘薯知识助手已准备就绪，请问有什么可以帮助你的？", wait=True)
        except Exception as e:
            logging.error(f"播放欢迎信息失败: {e}")
            print("🤖 甘薯知识助手已准备就绪，请问有什么可以帮助你的？")
        
        # 主对话循环
        while not shutdown_event.is_set():
            try:
                # 清空可能的音频缓冲
                if hasattr(asr, 'stream') and asr.stream:
                    try:
                        # 清空缓冲区中的数据
                        while hasattr(asr.stream, 'get_read_available') and asr.stream.get_read_available() > 0:
                            asr.stream.read(asr.CHUNK, exception_on_overflow=False)
                    except Exception as e:
                        print(f"⚠️ 清理音频缓冲区时出错: {e}")

                # 步骤 1：语音转文本
                print("\n📢 等待语音输入...")
                # 执行语音识别
                question_data = asr.real_time_recognition()
                
                # 检查是否收到退出信号
                if shutdown_event.is_set():
                    break
                
                # 检查语音识别结果
                if 'err_no' in question_data and question_data['err_no'] != 0:
                    print(f"❌ 语音识别失败: {question_data.get('err_msg', '未知错误')}")
                    try:
                        await tts.text_to_speech("11抱歉，我没有听清您说的话，请再说一次。", wait=True)
                    except:
                        print("🔄 抱歉，我没有听清您说的话，请再说一次。")
                    continue
                    
                if 'result' not in question_data or not question_data['result']:
                    print("❌ 未检测到语音输入")
                    try:
                        await tts.text_to_speech("11我没有听到您的问题，请再说一次。", wait=True)
                    except:
                        print("🔄 我没有听到您的问题，请再说一次。")
                    continue
                    
                question = question_data['result'][0]
                print(f"🧠 问题：{question}")
                
                # 检查是否是退出命令
                if question.lower() in ["退出。", "没有了。", "没了。", "无。", "关闭。", "停止。", "拜拜。", "再见。","退出了。"]:
                    print("="*50)
                    print(f"🚪 收到退出命令: '{question}'，lower() 结果是: '{question.lower()}'")
                    print("="*50)
                    
                    try:
                        await tts.text_to_speech("11好的，感谢使用甘薯知识助手，再见！", wait=True)
                    except:
                        print("👋 好的，感谢使用甘薯知识助手，再见！")
                    # 设置退出事件，退出循环
                    shutdown_event.set()
                    break

                # 步骤 2：问答模型处理
                print("💭 正在思考问题...")
                start_time = time.time()
                answer = qa.ask(question)
                print(f"💬 答案 (用时: {time.time()-start_time:.2f}秒)：{answer}")

                # 步骤 3：文本转语音输出
                try:
                    await tts.text_to_speech('11'+answer, wait=True)
                    # 询问是否还有其他问题
                    await tts.text_to_speech("11您还有其他问题吗？", wait=True)
                except Exception as e:
                    logging.error(f"TTS失败: {e}")
                    print(f"\n{'═'*50}\n💬 回答: {answer}\n{'═'*50}")
                    print("❓ 您还有其他问题吗？")

            except KeyboardInterrupt:
                print("\n🛑 收到键盘中断")
                shutdown_event.set()
                break
            except asyncio.CancelledError:
                print("\n🛑 任务被取消")
                shutdown_event.set()
                break
            except Exception as e:
                logging.error(f"对话循环出错：{e}")
                print(f"❌ 出错：{e}")
                try:
                    await tts.text_to_speech("11抱歉，系统遇到了一些问题，请再试一次", wait=True)
                except:
                    print("⚠️ 抱歉，系统遇到了一些问题，请再试一次")
                # 等待短暂时间
                await asyncio.sleep(0.5)
    
    except Exception as e:
        logging.error(f"系统运行出错: {e}")
        print(f"❌ 系统运行出错: {e}")
    
    finally:
        # 清理资源
        print("🧹 正在清理资源...")
        if asr:
            try:
                asr.stop_recording()
                print("✅ ASR资源已释放")
            except Exception as e:
                logging.error(f"关闭ASR时出错: {e}")
        
        try:
            if tts and not shutdown_event.is_set():
                await tts.text_to_speech("11感谢使用甘薯知识助手，再见！", wait=True)
        except Exception as e:
            logging.error(f"播放告别语音失败: {e}")
            
        print(f"{'👋 感谢使用甘薯知识助手，再见！👋':^150}")
        return


async def initialize_qa_model():
    """单独的函数用于初始化QA模型"""
    try:
        qa = KnowledgeQA()
        return qa
    except Exception as e:
        logging.error(f"QA模型初始化失败: {e}")
        raise


# 信号处理函数，设置全局退出事件
def signal_handler():
    print(f"{'🛑 收到系统退出信号，正在安全退出... 🛑':^150}")
    shutdown_event.set()


async def main():
    """主函数：包含人脸认证和甘薯知识系统"""
    print(f"{'🚀 启动系统... 🚀':^150}")
    
    # 设置信号处理
    loop = asyncio.get_running_loop()
    for signame in ('SIGINT', 'SIGTERM'):
        loop.add_signal_handler(
            getattr(signal, signame),
            signal_handler
        )
    
    try:
        # 初始化TTS用于欢迎消息
        tts = TTSHelper(voice="zh-CN-XiaoyiNeural", rate="+0%", volume="+0%")
        
        # 初始化人脸认证系统
        face_system = FaceRecognizer()
        if not face_system.initialize():
            print("❌ 系统初始化失败，程序退出")
            try:
                await tts.text_to_speech("11系统初始化失败，请检查人脸模型", wait=True)
            except:
                print("❌ 系统初始化失败，请检查人脸模型")
            return
        
        # 执行人脸认证
        print("📷 开始人脸认证，请面向摄像头")
        try:
            await tts.text_to_speech("11开始人脸认证，请面向摄像头", wait=True)
        except Exception as e:
            logging.error(f"TTS失败: {e}")
            print("📷 开始人脸认证，请面向摄像头")
        
        # 执行人脸识别
        auth_success, user_name = face_system.recognize_face()
        
        # 认证通过后运行甘薯知识系统
        if auth_success:
            welcome_message = f"11欢迎你，{user_name}。正在进入甘薯知识系统。"
            try:
                await tts.text_to_speech(welcome_message, wait=True)
            except:
                pass
            
            await run_sweet_potato_system(user_name)

        else:
            deny_message = "11你是谁呀？我不认识你。系统将退出。"
            print("🚫 认证失败，拒绝访问")
            try:
                await tts.text_to_speech(deny_message, wait=True)
            except:
                print("🚫 认证失败，拒绝访问。系统将退出。")
    
    except KeyboardInterrupt:
        print("\n⌨️ 程序被用户中断")
    except Exception as e:
        logging.error(f"主程序错误: {e}")
        print(f"❌ 程序遇到错误: {e}")
    finally:
        # 确保所有资源都被清理
        print("🔄 程序正在退出...")


if __name__ == "__main__":
    # 运行主程序
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⌨️ 程序已退出")
    except Exception as e:
        logging.error(f"程序异常退出: {e}")
        print(f"❌ 程序异常退出: {e}")
    finally:
        print("👋 程序已完全退出")
   
        os._exit(0)