import asyncio
import os
import time
import threading
import pyaudio
import edge_tts
import subprocess
import logging
import re
import webrtcvad
import array
import math
from aip import AipSpeech

# 百度API配置
APP_ID = '118613302'
API_KEY = '7hSl10mvmtaCndZoab0S3BXQ'
SECRET_KEY = 'Fv10TxiFLmWb4UTAdLeA2eaTIE56QtkW'

client = AipSpeech(APP_ID, API_KEY, SECRET_KEY)

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

class SimpleInterruptibleTTS:
    def __init__(self, voice="zh-CN-XiaoyiNeural", rate="+0%", volume="+0%"):
        # TTS配置
        self.voice = voice
        self.rate = rate
        self.volume = volume
        self.energy_threshold = 1000
        
        # 中断控制
        self.is_speaking = False
        self.should_interrupt = False
        self.listen_thread = None
        self.playback_process = None
        
        # 音频输入配置（固定为16kHz）
        self.CHUNK = 320  # 20ms at 16kHz
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000
        
        # 始终使用 VAD 级别 3（嘈杂环境）
        try:
            self.vad = webrtcvad.Vad(3)  # 总是使用最高灵敏度
            logging.info("VAD已初始化，使用高灵敏度模式")
        except Exception as e:
            logging.error(f"初始化VAD失败: {e}")
            self.vad = None
        
        # 初始化PyAudio
        self.p = pyaudio.PyAudio()
        self.input_stream = None
        
        # 直接设置输入流
        self.setup_input_stream()
    
    def preprocess_text(self, text):
        """预处理文本，替换标点符号"""
        if not text:
            return ""
            
        text = text.replace("，", ",")
        text = text.replace("。", ",")
        text = text.replace("、", ",")
        text = text.replace("；", ",")
        text = text.replace("：", ",")
        text = text.replace("*", ',')
        text = text.replace(".", ',')
        text = text.replace("#", ',')
        text = text.replace("？", ',')
        text = re.sub(r'[\x00-\x1F\x7F]', '', text)
        
        return text
        
    def setup_input_stream(self):
        """简化的音频输入流设置方法"""
        if self.input_stream is None:
            try:
                # 直接使用默认输入设备
                self.input_stream = self.p.open(
                    format=self.FORMAT,
                    channels=self.CHANNELS,
                    rate=self.RATE,
                    input=True,
                    frames_per_buffer=self.CHUNK
                )
                logging.info(f"成功打开输入流，使用采样率: {self.RATE}Hz")
                return True
            except Exception as e:
                logging.error(f"设置输入流出错: {e}")
                return False
        return True  # 如果输入流已经存在
    
    def close_input_stream(self):
        """关闭音频输入流"""
        if self.input_stream:
            try:
                self.input_stream.stop_stream()
                self.input_stream.close()
                self.input_stream = None
                logging.info("输入流已关闭")
            except Exception as e:
                logging.error(f"关闭输入流出错: {e}")
    
    def is_speech(self, data):
        """检测音频数据是否包含语音"""
        try:
            if self.vad.is_speech(data, self.RATE):
                return True
            return False
        except Exception as e:
            logging.error(f"语音检测出错: {e}")
            return False
    
    def listen_for_interruption(self):
        """监听用户语音，检测是否需要中断"""
        if not self.setup_input_stream():
            logging.error("无法设置输入流进行中断监听")
            return
        
        consecutive_speech_frames = 0
        required_speech_frames = 3  # 连续检测到3帧语音才触发中断
        
        logging.info("开始监听中断...")
        
        while self.is_speaking:
            try:
                data = self.input_stream.read(self.CHUNK, exception_on_overflow=False)
                
                if self.is_speech(data):
                    consecutive_speech_frames += 1
                    
                    # 如果连续检测到语音帧达到阈值，触发中断
                    if consecutive_speech_frames >= required_speech_frames:
                        logging.info(f"检测到用户语音，准备中断... (连续帧数: {consecutive_speech_frames})")
                        self.should_interrupt = True
                        self.stop_playback()
                        break
                else:
                    consecutive_speech_frames = 0  # 重置计数器
                    
            except Exception as e:
                logging.error(f"监听中断时出错: {e}")
                break
                
        logging.info("停止监听中断")
    
    def stop_playback(self):
        """停止正在播放的音频"""
        if self.playback_process and self.playback_process.poll() is None:
            try:
                self.playback_process.terminate()
                time.sleep(0.2)
                if self.playback_process.poll() is None:
                    self.playback_process.kill()
                logging.info("音频播放已停止")
            except Exception as e:
                logging.error(f"停止音频播放出错: {e}")
    
    async def get_user_input(self):
        """获取用户输入（打断后）"""
        logging.info("请问您有什么需要？")
        
        if not self.setup_input_stream():
            logging.error("无法设置输入流获取用户输入")
            return None
        
        # 配置录音参数
        input_frames = []
        start_time = time.time()
        speech_started = False
        last_speech_time = time.time()
        silence_duration = 1.0  # 1秒无语音则结束录音
        max_record_seconds = 7.0  # 最长录音7秒
        
        logging.info("请说话...")
        
        while True:
            try:
                data = self.input_stream.read(self.CHUNK, exception_on_overflow=False)
                
                if self.is_speech(data):
                    if not speech_started:
                        speech_started = True
                        logging.info("检测到语音输入...")
                    last_speech_time = time.time()
                    input_frames.append(data)
                else:
                    if speech_started and (time.time() - last_speech_time) >= silence_duration:
                        logging.info("语音输入结束")
                        break
                        
                if (time.time() - start_time) >= max_record_seconds:
                    logging.info("达到最大录音时间")
                    break
                    
            except Exception as e:
                logging.error(f"录音出错: {e}")
                break
        
        if input_frames:
            try:
                audio_data = b"".join(input_frames)
                logging.info(f"上传 {len(audio_data)} 字节到百度API")
                result = client.asr(audio_data, 'pcm', self.RATE, {'dev_pid': 1537})
                
                if result['err_no'] == 0 and len(result['result']) > 0:
                    user_question = result['result'][0]
                    logging.info(f"用户说: {user_question}")
                    return user_question
            except Exception as e:
                logging.error(f"语音识别API调用出错: {e}")
                return None
        else:
            logging.info("没有录到语音")
            return None
        
    async def prepare_audio_file(self, text):
        """准备音频文件，返回文件路径"""
        try:
            communicate = edge_tts.Communicate(
                text=text,
                voice=self.voice,
                rate=self.rate,
                volume=self.volume
            )
            
            temp_file = f"temp_audio_{int(time.time())}_{id(self)}.mp3"
            
            await communicate.save(temp_file)
            logging.info(f"音频文件已保存: {temp_file}")
            
            return temp_file
        except Exception as e:
            logging.error(f"准备音频文件出错: {e}")
            return None
    
    async def play_audio_with_interrupt(self, audio_file):
        """播放音频文件，同时监听中断"""
        if not os.path.exists(audio_file):
            logging.error(f"音频文件不存在: {audio_file}")
            return False, None
        
        try:
            # 设置状态
            self.is_speaking = True
            self.should_interrupt = False
            
            # 启动监听线程
            self.listen_thread = threading.Thread(target=self.listen_for_interruption)
            self.listen_thread.daemon = True
            self.listen_thread.start()
            
            # 启动播放进程
            self.playback_process = subprocess.Popen(
                ["mpg123", audio_file], 
                stdout=subprocess.DEVNULL, 
                stderr=subprocess.DEVNULL
            )
            
            # 监听中断事件
            was_interrupted = False
            while self.playback_process.poll() is None:
                if self.should_interrupt:
                    self.stop_playback()
                    was_interrupted = True
                    logging.info("语音输出已被用户中断")
                    break
                await asyncio.sleep(0.1)
            
            # 等待线程完成
            if self.listen_thread and self.listen_thread.is_alive():
                self.is_speaking = False
                self.listen_thread.join(timeout=1.0)
            
            # 检查是否播放完成还是被中断
            if was_interrupted:
                logging.info("处理中断...")
                await asyncio.sleep(0.5)
                
                # 播放"怎么了?"
                await self.text_to_speech("怎么了?")
                
                # 获取新的用户输入
                user_input = await self.get_user_input()
                return True, user_input
            else:
                logging.info("音频播放完成")
                return False, None
                
        except Exception as e:
            logging.error(f"播放音频出错: {e}")
            return False, None
        finally:
            # 重置状态和清理
            self.is_speaking = False
            self.should_interrupt = False
            self.playback_process = None
            
            # 尝试删除临时文件
            try:
                if os.path.exists(audio_file):
                    os.remove(audio_file)
                    logging.info(f"临时音频文件已删除: {audio_file}")
            except Exception as e:
                logging.error(f"删除临时文件出错: {e}")
    
    async def speak_with_interrupt(self, text):
        """启用中断功能的语音输出主函数"""
        processed_text = self.preprocess_text(text)
        if not processed_text:
            logging.warning("文本为空，跳过播放")
            return False, None
            
        logging.info("-" * 50)
        logging.info(f"开始播放: {processed_text[:100]}...")
        logging.info("-" * 50)
            
        try:
            # 准备音频文件
            audio_file = await self.prepare_audio_file(processed_text)
            if not audio_file:
                logging.error("无法准备音频文件")
                return False, None
                
            # 播放并监听中断
            return await self.play_audio_with_interrupt(audio_file)
                
        except Exception as e:
            logging.error(f"语音播放出错: {e}")
            return False, None
    
    async def text_to_speech(self, text):
        """基本的文本到语音转换（不带中断功能，用于短句）"""
        processed_text = self.preprocess_text(text)
        if not processed_text:
            return
            
        try:
            # 创建edge-tts通信实例
            communicate = edge_tts.Communicate(
                text=processed_text,
                voice=self.voice,
                rate=self.rate,
                volume=self.volume
            )
            
            # 生成临时文件
            output_file = f"temp_simple_{int(time.time())}_{id(self)}.mp3"
            await communicate.save(output_file)
            
            # 播放
            subprocess.run(
                ["mpg123", output_file], 
                stdout=subprocess.DEVNULL, 
                stderr=subprocess.DEVNULL
            )
            
            # 删除临时文件
            if os.path.exists(output_file):
                os.remove(output_file)
                
        except Exception as e:
            logging.error(f"简单语音播放失败: {e}")
    
    def cleanup(self):
        """清理资源"""
        self.close_input_stream()
        self.stop_playback()
        
        if self.p:
            self.p.terminate()
            logging.info("已清理所有音频资源")


# 简单演示代码
async def demo():
    tts = SimpleInterruptibleTTS()
    
    try:
        print("\n" + "="*60)
        print("演示可中断的语音输出")
        print("在听到语音输出时，请尝试说话来中断它")
        print("="*60 + "\n")
        
        long_text = """甘薯又名地瓜、红薯、番薯等，是世界上重要的粮食作物之一。它富含淀粉、膳食纤维、胡萝卜素、维生素C和矿物质，营养价值非常高。甘薯品种繁多，根据肉色可分为白心、黄心、橙心、紫心等多种类型。甘薯适应性强，抗逆性好，可在多种气候条件下种植。"""
        
        print("将开始播放语音，请准备随时打断...")
        await asyncio.sleep(1)  # 给用户准备的时间
        
        # 播放长文本，并等待可能的中断
        was_interrupted, user_question = await tts.speak_with_interrupt(long_text)
        
        if was_interrupted and user_question:
            print(f"\n用户提问: {user_question}")
            # 这里可以调用QA模型处理用户问题
            answer = f"关于{user_question}的回答是：这是一个示例回答，实际应用中这里会调用您的QA模型。"
            
            # 继续对话
            await tts.speak_with_interrupt(answer)
        else:
            print("\n语音播放完成，没有被中断")
        
        print("\n演示结束")
            
    except KeyboardInterrupt:
        print("\n程序被手动中断")
    except Exception as e:
        print(f"\n演示过程中出错: {e}")
    finally:
        tts.cleanup()


if __name__ == "__main__":
    try:
        asyncio.run(demo())
    except KeyboardInterrupt:
        print("\n程序被手动中断")
    except Exception as e:
        print(f"\n程序出错: {e}")