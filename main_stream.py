import sys
import os
sys.path.append(os.path.abspath("/home/wuye/vscode/chatbox"))
import asyncio
import logging
import signal
import argparse
import time
from qa_model.qa_model_easy import KnowledgeQA
from ASR.asr import ASRhelper
from TTS.tts_stream import TTSStreamer  
from face.face_recognize import FaceRecognizer

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("chatbox.log"), logging.StreamHandler()]
)

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
        
    async def authenticate_user(self):
        """使用人脸识别进行用户认证"""
        logging.info("开始人脸认证...")
        
        # 初始化TTS用于提示信息
        temp_tts = TTSStreamer(voice=self.voice)
        await temp_tts.speak_text("11开始人脸认证，请面向摄像头", wait=True)
        
        # 初始化人脸识别系统
        face_system = FaceRecognizer()
        if not face_system.initialize():
            logging.error("人脸识别系统初始化失败")
            await temp_tts.speak_text("11人脸识别系统初始化失败，请检查人脸模型", wait=True)
            await temp_tts.shutdown()
            return False, None
        
        # 执行人脸认证
        auth_success, user_name = face_system.recognize_face()
        
        # 根据认证结果提供语音反馈
        if auth_success:
            welcome_message = f"11欢迎你，{user_name}。已进入甘薯知识系统。"
            logging.info(f"认证成功: {user_name}")
            await temp_tts.speak_text(welcome_message, wait=True)
        else:
            deny_message = "11你是谁？我不认识你。系统将退出。"
            logging.info("认证失败，拒绝访问")
            await temp_tts.speak_text(deny_message, wait=True)
        
        # 关闭临时TTS
        await temp_tts.shutdown()
        
        return auth_success, user_name
        
    async def initialize(self):
        """初始化所有组件"""
        try:
            logging.info("正在初始化甘薯问答系统...")
            self.qa = KnowledgeQA(llm_model=self.model)
            self.tts = TTSStreamer(voice=self.voice)
            self.asr = ASRhelper()
            logging.info("系统初始化完成")
            return True
        except Exception as e:
            logging.error(f"初始化失败: {e}")
            return False
            
    def setup_signal_handlers(self):
        """设置信号处理器用于优雅退出"""
        def signal_handler(*args):
            logging.info("收到退出信号")
            self.shutdown_event.set()
            
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, lambda s, f: asyncio.create_task(
                asyncio.to_thread(signal_handler)))
    
    async def process_user_input(self):
        """处理用户语音输入"""
        logging.info("等待用户输入...")
        
        # 确保TTS完全结束并额外等待，防止自我收听
        await self.tts.wait_until_done()
        await asyncio.sleep(1.0)  # 增加额外等待时间以确保音频播放完全结束
        
        
        # 清空任何排队的音频缓冲
        try:
            if hasattr(self.asr, 'stream'):
                # 清除缓冲区中残留的音频数据
                time.sleep(0.2)  # 短暂等待以确保任何未处理的数据都到达缓冲区
                while self.asr.stream.get_read_available() > 0:
                    self.asr.stream.read(self.asr.CHUNK, exception_on_overflow=False)
        except Exception as e:
            logging.warning(f"清理音频缓冲区时出错: {e}")
        
        # 执行语音识别
        question_result = self.asr.real_time_recognition()
        
        # 处理结果
        if not question_result or 'result' not in question_result or not question_result['result']:
            logging.info("未检测到有效语音输入")
            return None
            
        question = question_result["result"][0]
        logging.info(f"用户问题: {question}")
        
        # 检查是否是退出命令
        if question.lower() in ["退出", "关闭", "停止", "拜拜", "再见"]:
            logging.info("收到退出命令")
            self.shutdown_event.set()
            return None
            
        return question
        
    async def process_llm_response(self, question):
        """处理LLM响应并流式输出"""
        if not question:
            return
            
        logging.info("正在处理问题...")
        
        # 统计收到的文本块
        total_response = ""
        current_segment = ""
        segment_size = 60  # 每个语音段的大致字符数量
        first_segment = True  # 标记是否是第一个段落
        
        try:
            # 开始计时
            start_time = time.time()
            
            # 使用流式接口获取回复
            async for chunk in self.qa.ask_stream(question):
                # 添加到总响应
                total_response += chunk
                current_segment += chunk
                
                # 当前段达到一定长度，启动TTS
                if len(current_segment) >= segment_size:
                    if self.debug:
                        print(f"播放段落: {current_segment}")
                        
                
                    if first_segment:
                        await self.tts.speak_segment('11' + current_segment)
                        first_segment = False
                    else:
                        await self.tts.speak_segment('11'+current_segment)
                        
                    current_segment = ""
            
            # 处理最后剩余部分
            if current_segment:
                if self.debug:
                    print(f"播放最后段落: {current_segment}")
                    
       
                if first_segment:
                    await self.tts.speak_segment('11' + current_segment)
                else:
                    await self.tts.speak_segment('11'+current_segment)
                
            # 输出完整响应到控制台
            logging.info(f"完整响应 ({time.time() - start_time:.2f}秒)：{total_response}")
            
            # 等待所有语音播放完成
            await self.tts.wait_until_done()
            
        except Exception as e:
            logging.error(f"处理响应时出错: {e}")
            try:
                # 尝试播放错误提示
                await self.tts.speak_text("11抱歉，处理您的问题时出现了错误。", wait=True)
            except:
                pass
    
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
        print("\n" + "=" * 80)
        print(f"{'甘薯知识问答系统已启动':^80}")
        print(f"{'用户: ' + self.recognized_user:^80}")
        print(f"{'按 Ctrl+C 退出':^80}")
        print("=" * 80 + "\n")
        
        try:
            # 初始欢迎语
            await self.tts.speak_text(f"{self.recognized_user}，甘薯知识问答系统已启动，请问您有什么关于甘薯的问题？", wait=True)
            
            while not self.shutdown_event.is_set():
                # 获取用户问题
                question = await self.process_user_input()
                
                # 处理问题并回答
                if question:
                    await self.process_llm_response(question)
                    # 对话间短暂暂停
                    await asyncio.sleep(1.0)
                else:
                    # 如果没有输入，间隔较长
                    await asyncio.sleep(0.5)
                    
        except Exception as e:
            logging.error(f"运行时发生错误: {e}")
        finally:
            # 清理资源
            await self.shutdown()
            
    async def shutdown(self):
        """清理资源并关闭系统"""
        logging.info("正在关闭系统...")
        
        try:
            # 先关闭TTS (最重要的资源释放)
            if self.tts:
                await self.tts.shutdown()
                
            # 关闭ASR
            if self.asr:
                self.asr.stop_recording()
                
            logging.info("所有资源已清理，系统已安全关闭")
            print("\n系统已安全关闭，感谢使用！")
            
        except Exception as e:
            logging.error(f"清理资源时出错: {e}")

async def main():
    """程序入口点"""
    parser = argparse.ArgumentParser(description="甘薯知识问答系统")
    parser.add_argument("--model", default="qwen2.5:7b", help="LLM模型名称")
    parser.add_argument("--voice", default="zh-CN-XiaoyiNeural", help="TTS语音")
    parser.add_argument("--debug", action="store_true", help="启用调试模式")
    args = parser.parse_args()
    
    chatbox = SweetPotatoChatbox(
        model=args.model,
        voice=args.voice,
        debug=args.debug
    )
    
    await chatbox.run()

if __name__ == "__main__":
    asyncio.run(main())