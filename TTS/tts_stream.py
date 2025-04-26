import sys
import os
sys.path.append(os.path.abspath("/home/wuye/vscode/chatbox"))
import asyncio
import edge_tts
import subprocess
import io
import logging
import re



class TTSStreamer:
    def __init__(self, voice="zh-CN-XiaoyiNeural", rate="+0%", volume="+0%"):
        self.voice = voice
        self.rate = rate
        self.volume = volume
        self.is_speaking = False
        self._lock = asyncio.Lock()
        self.mpg123_process = None
        self.min_buffer_size = 4096
        self.buffered_audio = bytearray()
        self.speech_queue = asyncio.Queue()
        self._speech_complete_event = asyncio.Event()
        self._speech_complete_event.set()  
        self.speech_task = None

    def preprocess_text(self, text):
        """
        预处理文本，替换不标准的标点并清理可能导致问题的字符
        """
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
 
        # print(f"预处理后的文本：{text}")
        return text

    async def start_player(self):
        """启动mpg123进程，确保只存在一个实例"""
        async with self._lock:
            if self.mpg123_process is None or self.mpg123_process.poll() is not None:
                try:
                    self.mpg123_process = subprocess.Popen(
                        ["mpg123", "-q", "-"],  # 安静模式
                        stdin=subprocess.PIPE,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        bufsize=1024*8  # 大缓冲区
                    )
                    logging.info("mpg123播放器已启动")
                except Exception as e:
                    logging.error(f"启动mpg123失败: {e}")
                    self.mpg123_process = None
                    raise

    async def stop_player(self):
        """安全关闭播放器进程"""
        async with self._lock:
            if self.mpg123_process:
                try:
                    self.mpg123_process.stdin.flush()
                    self.mpg123_process.stdin.close()
                    self.mpg123_process.terminate()
                    await asyncio.sleep(0.3)
                    if self.mpg123_process.poll() is None:
                        self.mpg123_process.kill()
                    await asyncio.sleep(0.2)
                    self.mpg123_process = None
                    logging.info("mpg123播放器已关闭")
                except Exception as e:
                    logging.error(f"关闭mpg123时出错: {e}")

    async def _generate_speech(self, text):
        """生成语音数据（完整缓冲）"""
        if not text or not text.strip():
            return None
            
        try:
            communicate = edge_tts.Communicate(
                text, 
                self.voice,
                rate=self.rate,
                volume=self.volume
            )
            
            audio_data = io.BytesIO()
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_data.write(chunk["data"])
                    
            # 返回完整数据
            if audio_data.tell() > 0:
                audio_data.seek(0)
                return audio_data.getvalue()
            else:
                return None
        except Exception as e:
            logging.error(f"生成语音时出错: {e}")
            return None
            
    async def _speech_processor(self):
        """处理语音队列的后台任务"""
        try:
            await self.start_player()
            
            while True:
                # 获取下一段语音文本
                text = await self.speech_queue.get()
                
                # None作为结束信号
                if text is None:
                    break
                    
                try:
                    # 生成语音数据
                    audio_data = await self._generate_speech(text)
                    
                    if audio_data:
                        # 播放语音
                        if self.mpg123_process and self.mpg123_process.poll() is None:
                            self.mpg123_process.stdin.write(audio_data)
                            self.mpg123_process.stdin.flush()
                            
                            # 估计播放时间
                            estimated_duration = len(text) * 0.1
                            await asyncio.sleep(estimated_duration)
                        else:
                            # 重启播放器
                            await self.start_player()
                            if self.mpg123_process:
                                self.mpg123_process.stdin.write(audio_data)
                                self.mpg123_process.stdin.flush()
                                await asyncio.sleep(len(text) * 0.1)
                except Exception as e:
                    logging.error(f"播放语音时出错: {e}")
                
                # 标记此项已完成
                self.speech_queue.task_done()
                
        except Exception as e:
            logging.error(f"语音处理任务出错: {e}")
        finally:
            # 确保播放器关闭
            await self.stop_player()
            
    async def start_speech_processor(self):
        """启动语音处理任务"""
        if self.speech_task is None or self.speech_task.done():
            self.speech_task = asyncio.create_task(self._speech_processor())
            
    async def stop_speech_processor(self):
        """停止语音处理任务"""
        if self.speech_task and not self.speech_task.done():
            # 发送结束信号
            await self.speech_queue.put(None)
            # 等待任务结束
            await self.speech_task
            self.speech_task = None

    async def speak_segment(self, text):
        """将文本段加入语音队列"""
        if not text or not text.strip():
            return
            
        # 确保处理器运行
        await self.start_speech_processor()
        
        # 修改状态
        self.is_speaking = True
        self._speech_complete_event.clear()
        
        # 放入队列
        text = self.preprocess_text(text)
        await self.speech_queue.put(text)
        
        # 注意：这里不等待播放完成，允许并行处理
        
    async def wait_until_done(self):
        """等待所有语音播放完成"""
        if self.speech_queue.qsize() > 0:
            await self.speech_queue.join()
        
        self.is_speaking = False
        self._speech_complete_event.set()
        
    async def speak_text(self, text, wait=False):
        """流式处理较长文本，分段播放"""
        text = self.preprocess_text(text)
        
        # 分割文本为自然段落
        segments = []
        
        # 基于句子分割
        sentences = re.split(r'([.!?。！？])', text)
        
        # 重新组合句子和标点
        current_segment = ""
        for i in range(0, len(sentences)-1, 2):
            if i+1 < len(sentences):
                sentence = sentences[i] + sentences[i+1]
            else:
                sentence = sentences[i]
                
            # 如果当前段落已经太长，存储并开始新段落
            if len(current_segment) + len(sentence) > 100:  # 适当的段落长度
                segments.append(current_segment)
                current_segment = sentence
            else:
                current_segment += sentence
                
        # 添加最后一个段落
        if current_segment:
            segments.append(current_segment)
            
        # 如果没有找到好的分割点，作为一个段落处理
        if not segments:
            segments = [text]
            
        # 播放所有段落
        for segment in segments:
            if segment.strip():
                await self.speak_segment(segment)
                
        # 如果需要等待完成
        if wait:
            await self.wait_until_done()
    
    async def shutdown(self):
        """清理资源"""
        # 停止处理新的语音请求
        await self.stop_speech_processor()
        # 确保播放器关闭
        await self.stop_player()