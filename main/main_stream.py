import sys
import os
sys.path.append(os.path.abspath("/home/wuye/vscode/chatbox"))
import asyncio
import logging
import signal
import argparse
import time
import itertools
import threading
from qa_model.qa_model_easy import KnowledgeQA
from ASR.asr import ASRhelper
from TTS.tts_stream import TTSStreamer  
from face.face_recognize import FaceRecognizer

# 配置日志 - 美化日志格式
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)8s │ %(message)s",
    handlers=[logging.FileHandler("chatbox.log"), logging.StreamHandler()]
)

# 加载动画类
class LoadingAnimation:
    def __init__(self, desc="加载中"):
        self.desc = desc
        self.done = False
        self.thread = None
        
    def animate(self):
        # 动画符号选项
        spinners = [
            "⣾⣽⣻⢿⡿⣟⣯⣷",
            "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏",
            "🌑🌒🌓🌔🌕🌖🌗🌘",
            "🕐🕑🕒🕓🕔🕕🕖🕗🕘🕙🕚🕛"
        ]
        spinner = spinners[2]  
        
        for char in itertools.cycle(spinner):
            if self.done:
                break
            sys.stdout.write(f"\r{char} {self.desc} ")
            sys.stdout.flush()
            time.sleep(0.1)
        # 清除加载动画行
        sys.stdout.write(f"\r{'✅ ' + self.desc + ' 完成!':60}\n")
        sys.stdout.flush()
        
    def start(self):
        self.thread = threading.Thread(target=self.animate)
        self.thread.start()
        
    def stop(self):
        self.done = True
        if self.thread:
            self.thread.join()

class SweetPotatoChatbox:
    def __init__(self, model="qwen2.5:7b", voice="zh-CN-XiaoyiNeural", debug=False):
        self.model = model
        self.voice = voice
        self.debug = debug
        self.shutdown_event = asyncio.Event()
        self.qa = None
        self.tts = None
        self.asr = None
        self.face_auth_success = False
        self.recognized_user = None
        self.first_interaction = True  # 标记是否是第一次交互
        
    async def authenticate_user(self):
        """使用人脸识别进行用户认证"""
        logging.info("🔐 开始人脸认证...")
        print("\n📷 开始人脸认证，请面向摄像头...")
        
        # 初始化TTS用于提示信息
        temp_tts = TTSStreamer(voice=self.voice)
        await temp_tts.speak_text("11开始人脸认证，请面向摄像头", wait=True)
        
        # 等待确保语音播放完毕后再进行识别
        await asyncio.sleep(1.0)
        
        # 初始化人脸识别系统
        face_system = FaceRecognizer()
        if not face_system.initialize():
            logging.error("❌ 人脸识别系统初始化失败")
            await temp_tts.speak_text("11人脸识别系统初始化失败,请检查人脸模型", wait=True)
            await temp_tts.shutdown()
            print("❌ 人脸识别系统初始化失败，程序退出")
            return False, None
        
        # 执行人脸认证
        auth_success, user_name = face_system.recognize_face()
        
        # 根据认证结果提供语音反馈
        if auth_success:
            welcome_message = f"11欢迎你{user_name}已进入甘薯知识系统。"
            logging.info(f"✅ 认证成功: {user_name}")
            print(f"\n✅ 认证成功！欢迎 {user_name}")
            await temp_tts.speak_text(welcome_message, wait=True)
        else:
            deny_message = "11你是谁我不认识你系统将退出。"
            logging.info("🚫 认证失败，拒绝访问")
            print("\n🚫 认证失败，无法识别用户，系统将退出")
            await temp_tts.speak_text(deny_message, wait=True)
        
        # 关闭临时TTS
        await temp_tts.shutdown()
        
        return auth_success, user_name
        
    async def initialize(self):
        """初始化所有组件"""
        try:
            logging.info("🚀 正在初始化甘薯问答系统...")
            print("\n🚀 正在初始化甘薯问答系统...")
            
            # 先初始化TTS
            tts_loader = LoadingAnimation("初始化语音合成系统")
            tts_loader.start()
            self.tts = TTSStreamer(voice=self.voice)
            tts_loader.stop()
            
            try:
                await self.tts.speak_text("11正在初始化系统...", wait=True)
            except Exception as e:
                logging.error(f"⚠️ TTS初始化测试失败: {e}")
                print("⚠️ 警告: 语音合成服务不可用，将以文本方式提供反馈")
            
            # 等待确保语音播放完毕
            await asyncio.sleep(0.5)
                
            # 初始化ASR
            logging.info("🎤 初始化语音识别...")
            asr_loader = LoadingAnimation("初始化语音识别系统")
            asr_loader.start()
            self.asr = ASRhelper()
            asr_loader.stop()
            
            # 初始化QA模型
            logging.info("🧠 正在加载知识模型，这可能需要一些时间...")
            try:
                await self.tts.speak_text("11正在加载知识模型，这可能需要一些时间...", wait=True)
            except Exception as e:
                logging.error(f"⚠️ TTS语音播放失败: {e}")
                print("🧠 正在加载知识模型，这可能需要一些时间...")
            
            # 等待确保语音播放完毕
            await asyncio.sleep(0.5)
            
            # 显示加载动画
            qa_loader = LoadingAnimation(f"加载知识模型 {self.model}")
            qa_loader.start()
                
            # 初始化QA模型
            self.qa = KnowledgeQA(llm_model=self.model)
            
            # 停止加载动画
            qa_loader.stop()
            
            logging.info("✨ 系统初始化完成")
            print("\n✨ 系统初始化完成，甘薯知识助手已准备就绪")
            return True
        except Exception as e:
            logging.error(f"❌ 初始化失败: {e}")
            print(f"\n❌ 初始化失败: {e}")
            return False
            
    def setup_signal_handlers(self):
        """设置信号处理器用于优雅退出"""
        loop = asyncio.get_running_loop()
        for signame in ('SIGINT', 'SIGTERM'):
            loop.add_signal_handler(
                getattr(signal, signame),
                self.signal_handler
            )
    
    def signal_handler(self):
        """处理系统信号"""
        logging.info("🛑 收到系统退出信号，正在安全退出...")
        print(f"\n{'🛑 收到系统退出信号，正在安全退出... 🛑':^80}")
        self.shutdown_event.set()
    
    async def clear_audio_buffer(self):
        """清理音频缓冲区"""
        try:
            if hasattr(self.asr, 'stream'):
                # 清除缓冲区中残留的音频数据
                time.sleep(0.2)  # 短暂等待以确保任何未处理的数据都到达缓冲区
                while self.asr.stream.get_read_available() > 0:
                    self.asr.stream.read(self.asr.CHUNK, exception_on_overflow=False)
                    
                logging.info("🧹 音频缓冲区已清理")
        except Exception as e:
            logging.warning(f"⚠️ 清理音频缓冲区时出错: {e}")
    
    async def process_user_input(self):
        """处理用户语音输入"""
        logging.info("\n🎤 等待语音播放完🎤")
        # print("\n🎤 等待语音输入...")
        
        # 确保TTS完全结束并额外等待，防止自我收听
        await self.tts.wait_until_done()
        
        # 提示文本，区分第一次交互和后续交互
        prompt_text = "11请问您有什么关于甘薯的问题？" if self.first_interaction else "11您还有什么问题吗？"
        self.first_interaction = False  # 重置第一次交互标志
        
        try:
            await self.tts.speak_text(prompt_text, wait=True)
        except Exception as e:
            logging.error(f"⚠️ 语音提示失败: {e}")
            print(prompt_text.replace("11", ""))
        
        # 等待一小段时间确保语音完全播放完毕，防止捕获自己的声音
        await asyncio.sleep(1.0)
        
        # 清空音频缓冲，防止捕获系统自己的语音输出
        await self.clear_audio_buffer()
        
        # 显示监听指示器
        listening_spinner = LoadingAnimation("正在聆听")
        listening_spinner.start()
        
        # 执行语音识别
        question_result = self.asr.real_time_recognition()
        
        # 停止监听指示器
        listening_spinner.stop()
        
        # 处理结果
        if not question_result or 'result' not in question_result or not question_result['result']:
            logging.info("❌ 未检测到有效语音输入")
            print("❌ 未检测到有效语音输入")
            try:
                await self.tts.speak_text("11我没有听到您的问题，请再说一次。", wait=True)
                # 等待确保提示语音播放完毕
                await asyncio.sleep(0.5)
                await self.clear_audio_buffer()
            except:
                print("🔄 我没有听到您的问题，请再说一次。")
            return None
            
        question = question_result["result"][0]
        logging.info(f"💬 问题: {question}")
        print(f"💬 问题: {question}")
        
        # 检查是否是退出命令
        if question.lower() in ["退出", "退出。", "没有了", "没有了。", "没了", "没了。", "无", "无。", "关闭", "关闭。", "停止", "停止。", "拜拜", "拜拜。", "再见", "再见。","退出了。"]:
            logging.info("="*50)
            logging.info(f"🚪 收到退出命令: '{question}'，lower() 结果是: '{question.lower()}'")
            logging.info("="*50)
            
            print("\n" + "═"*80)
            print(f"{'🚪 收到退出命令: ' + question:^80}")
            print("═"*80)
            
            try:
                await self.tts.speak_text("11好的，感谢使用甘薯知识助手，再见！", wait=True)
            except:
                print("👋 感谢使用甘薯知识助手，再见！")
                
            self.shutdown_event.set()
            return None
            
        return question
        
            
    
    async def run(self):
        """运行主循环"""
        # 首先进行人脸认证
        self.face_auth_success, self.recognized_user = await self.authenticate_user()
        
        # 如果人脸认证失败，退出程序
        if not self.face_auth_success:
            return
            
        # 人脸认证成功后，初始化系统组件
        if not await self.initialize():
            return
            
        self.setup_signal_handlers()
        
        # 启动提示
        print("\n" + "═" * 80)
        print(f"{'🌟 甘薯知识问答系统已启动 🌟':^80}")
        print(f"{'👤 用户: ' + self.recognized_user:^80}")
        print(f"{'⌨️  按 Ctrl+C 退出':^80}")
        print("═" * 80 + "\n")
        
        try:
            # 初始欢迎语
            try:
                await self.tts.speak_text(f"11{self.recognized_user}，甘薯知识问答系统已启动。", wait=True)
                # 确保欢迎语播放完毕后再继续
                await asyncio.sleep(0.5)
                await self.clear_audio_buffer()
            except Exception as e:
                logging.error(f"⚠️ 播放欢迎消息失败: {e}")
                print(f"👋 {self.recognized_user}，甘薯知识问答系统已启动。")

            while not self.shutdown_event.is_set():
                # 获取用户问题
                question = await self.process_user_input()
                
                # 检查是否收到退出信号
                if self.shutdown_event.is_set():
                    break
                
                # 处理问题并回答
                if question:
                    try:
                        answer_loader = LoadingAnimation("正在思考")
                        answer_loader.start()
                        
                        # 获取答案
                        answer = await self.qa.ask_stream(question)
                        
                        # 停止加载动画
                        answer_loader.stop()
                        # 播放答案
                        if answer:
                            logging.info(f"💡 答案: {answer}")
                            print(f"💡 答案: {answer}")
                            try:
                                await self.tts.speak_text(answer, wait=False)
                            except Exception as e:
                                logging.error(f"⚠️ 播放答案失败: {e}")
                                print(f"⚠️ 播放答案失败: {e}")
                    except Exception as e:
                        logging.error(f"❌ 处理问题时出错: {e}")
                        print(f"❌ 处理问题时出错: {e}")
                    
        except KeyboardInterrupt:
            logging.info("⌨️ 收到键盘中断信号")
            print("\n⌨️ 收到键盘中断信号")
            self.shutdown_event.set()
        except asyncio.CancelledError:
            logging.info("🛑 任务被取消")
            print("\n🛑 任务被取消")
            self.shutdown_event.set()
        except Exception as e:
            logging.error(f"❌ 运行时发生错误: {e}")
            print(f"\n❌ 运行时发生错误: {e}")
        finally:
            # 清理资源
            await self.shutdown()
            
    async def shutdown(self):
        """清理资源并关闭系统"""
        logging.info("🧹 正在关闭系统...")
        print("\n🧹 正在关闭系统...")
        
        # 显示关闭动画
        shutdown_animation = LoadingAnimation("正在清理系统资源")
        shutdown_animation.start()
        
        try:
            # 播放告别消息
            if self.tts and not self.shutdown_event.is_set():
                try:
                    await self.tts.speak_text("11感谢使用甘薯知识助手再见！", wait=True)
                except Exception as e:
                    logging.error(f"⚠️ 播放告别语音失败: {e}")
            
            # 先关闭TTS (最重要的资源释放)
            if self.tts:
                await self.tts.shutdown()
                
            # 关闭ASR
            if self.asr:
                self.asr.stop_recording()
                logging.info("✅ ASR资源已释放")
                
            # 停止关闭动画
            shutdown_animation.stop()
                
            logging.info("✅ 所有资源已清理，系统已安全关闭")
            print("\n" + "═" * 80)
            print(f"{'👋 系统已安全关闭，感谢使用！👋':^80}")
            print("═" * 80)
            
        except Exception as e:
            # 确保动画停止
            if 'shutdown_animation' in locals() and shutdown_animation.thread and shutdown_animation.thread.is_alive():
                shutdown_animation.stop()
                
            logging.error(f"❌ 清理资源时出错: {e}")
            print(f"\n❌ 清理资源时出错: {e}")

async def main():
    """程序入口点"""
    # 显示启动横幅
    print("\n" + "═" * 80)
    print(f"{'🚀 甘薯知识问答系统 v2.0 🚀':^80}")
    print(f"{'启动中...':^80}")
    print("═" * 80 + "\n")
    
    parser = argparse.ArgumentParser(description="甘薯知识问答系统")
    parser.add_argument("--model", default="qwen2.5:7b", help="LLM模型名称")
    parser.add_argument("--voice", default="zh-CN-XiaoyiNeural", help="TTS语音")
    parser.add_argument("--debug", action="store_true", help="启用调试模式")
    args = parser.parse_args()
    
    try:
        chatbox = SweetPotatoChatbox(
            model=args.model,
            voice=args.voice,
            debug=args.debug
        )
        
        await chatbox.run()
    except KeyboardInterrupt:
        print("\n⌨️ 程序被用户中断")
    except Exception as e:
        logging.error(f"❌ 程序出错: {e}")
        print(f"\n❌ 程序出错: {e}")
    finally:
        print("\n👋 程序已完全退出")
    
        os._exit(0)

if __name__ == "__main__":
    asyncio.run(main())