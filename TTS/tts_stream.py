import sys
import os
sys.path.append(os.path.abspath("/home/wuye/vscode/chatbox"))
import asyncio
import edge_tts
import subprocess
import io
import logging
import time
import re


class TTStreaming:
    def __init__(self, voice="zh-CN-XiaoyiNeural", rate="+0%", volume="+0%"):
        self.voice = voice
        self.rate = rate  
        self.volume = volume  
        self.is_speaking = False
        self.audio_queue = asyncio.Queue()
        self.mpg123_process = None
        self.min_buffer_size = 6400
        self._lock = asyncio.Lock()
        self._speech_complete_event = asyncio.Event()
        
        
        self._text_buffer = ""
        self._buffer_threshold = 50  
        self._last_speak_time = 0
        self._buffer_timeout = 1.0  

    def preprocess_text(self, text):
        """
        预处理文本，替换不标准的标点并清理可能导致问题的字符
        """
        if not text:
            return ""
            
        text = text.replace("，", ",")
        text = text.replace("。", ".")  # Changed to period for better speech pauses
        text = re.sub(r'[\x00-\x1F\x7F]', '', text)
        text = text.strip("，。！？")
        return text

    async def start_audio_player(self):
        """开启mpg123进程，确保只有一个实例在运行"""
        async with self._lock:
            if not self.mpg123_process or self.mpg123_process.poll() is not None:
                try:
                    self.mpg123_process = subprocess.Popen(
                        ["mpg123", "-q", "-"], 
                        stdin=subprocess.PIPE,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        bufsize=4096  
                    )
                    logging.info("mpg123 进程开启")
                except Exception as e:
                    logging.error(f"mpg123进程开启失败: {e}")
                    self.mpg123_process = None
                    raise

    async def stop_audio_player(self):
        """安全关闭 mpg123 进程"""
        async with self._lock:
            if self.mpg123_process:
                try:
                    self.mpg123_process.stdin.close()
                    self.mpg123_process.terminate()
                    await asyncio.sleep(0.2)
                    if self.mpg123_process.poll() is None:
                        self.mpg123_process.kill()
                    self.mpg123_process.wait()
                    self.mpg123_process = None
                    logging.info("mpg123 进程已关闭")
                except Exception as e:
                    logging.error(f"关闭mpg123进程时出错: {e}")

    async def _buffer_audio_data(self, text):
        """
        缓冲并预处理整个音频数据，确保即使在网络延迟下也不会丢失开头部分
        """
        if not text or not text.strip():
            return None
            
        logging.info(f"正在准备TTS音频: {text[:30]}...")
        start_time = time.time()
        
        try:
            # 先将整个文本转换为音频数据
            communicate = edge_tts.Communicate(
                text, 
                self.voice,
                rate=self.rate,
                volume=self.volume
            )
            
            # 存储完整音频数据
            full_audio = io.BytesIO()
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    full_audio.write(chunk["data"])
                    
            # 重置指针并返回完整数据
            if full_audio.tell() > 0:
                full_audio.seek(0)
                logging.info(f"TTS音频准备完成，耗时: {time.time() - start_time:.2f}秒")
                return full_audio.getvalue()
            else:
                logging.warning("未生成任何音频数据")
                return None
                
        except Exception as e:
            logging.error(f"准备音频时出错: {e}")
            return None

    async def stream_tts(self, text):
        """
        改进的TTS流式处理，确保不会丢失开头部分
        """
        if not text or not text.strip():
            logging.warning("文本为空，跳过语音播放")
            return
            
        start_time = time.time()
        
        # 获取完整预处理的音频数据
        audio_data = await self._buffer_audio_data(text)
        if not audio_data:
            logging.error("无法生成音频数据")
            return
            
        # 启动音频播放器
        await self.start_audio_player()
        
        try:
            # 直接写入完整音频数据到播放器
            if self.mpg123_process and self.mpg123_process.poll() is None:
                self.mpg123_process.stdin.write(audio_data)
                self.mpg123_process.stdin.flush()
                
                # 估算播放时间 (粗略估计: 每10个字符约1秒)
                estimated_duration = len(text) * 0.1
                # 等待估计的播放时间结束
                await asyncio.sleep(estimated_duration)
            else:
                logging.error("mpg123进程未启动或已关闭")
        except Exception as e:
            logging.error(f"播放音频时出错: {e}")
        finally:
            logging.info(f"TTS 处理总耗时: {time.time() - start_time:.2f}秒")

    async def speak(self, text):
        """
        增强的speak方法，处理文本缓冲以解决延迟问题
        """
        # 移除可能的前缀 (例如 '11')
        if text.startswith('11'):
            text = text[2:]
            
        # 如果文本为空，直接返回
        if not text or not text.strip():
            return
            
        # 设置状态标志
        self.is_speaking = True
        self._speech_complete_event.clear()
        
        try:
            # 预处理文本
            text = self.preprocess_text(text)
            await self.stream_tts(text)
        except Exception as e:
            logging.error(f"语音合成失败: {e}")
        finally:
            # 重置状态
            self.is_speaking = False
            self._speech_complete_event.set()
            self._last_speak_time = time.time()

    async def wait_until_done(self):
        """等待直到语音播放完成"""
        if self.is_speaking:
            await self._speech_complete_event.wait()
            # 额外等待一小段时间，确保完全完成
            await asyncio.sleep(0.2)

    async def shutdown(self):
        """清理资源"""
        await self.stop_audio_player()
        # 清空队列
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except:
                pass