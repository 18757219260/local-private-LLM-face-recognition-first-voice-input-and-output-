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
    def __init__(self, voice="zh-CN-XiaoyiNeural"):
        self.voice = voice
        self.is_speaking = False
        self.audio_queue = asyncio.Queue()
        self.mpg123_process = None
        self.min_buffer_size = 6400

    def preprocess_text(self, text):
        """
        预处理文本，替换不标准的标点并清理可能导致问题的字符
        """
        text = text.replace("，", ",")
        text = text.replace("。", ",")
        text = re.sub(r'[\x00-\x1F\x7F]', '', text)
        text = text.strip("，。！？")
        print(f"预处理后的文本：{text}")
        return text

    async def start_audio_player(self):
        """启动单一的 mpg123 进程"""
        if not self.mpg123_process:
            self.mpg123_process = subprocess.Popen(
                ["mpg123", "-"],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            print("mpg123 进程启动")

    async def stop_audio_player(self):
        """关闭 mpg123 进程"""
        if self.mpg123_process:
            self.mpg123_process.stdin.close()
            self.mpg123_process.wait()
            self.mpg123_process = None
            print("mpg123 进程关闭")

    async def stream_tts(self, text):
        if not text.strip():
            logging.warning("文本为空!!，跳过语音播放‼️")
            return
        start_time = time.time()
        communicate = edge_tts.Communicate(text, self.voice,)

        async def audio_stream_handler():
            current_audio = io.BytesIO()
            try:
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        current_audio.write(chunk["data"])
                        if current_audio.tell() >= self.min_buffer_size:
                            current_audio.seek(0)
                            await self.audio_queue.put(current_audio.getvalue())
                            current_audio = io.BytesIO()
                if current_audio.tell() > 0:
                    current_audio.seek(0)
                    await self.audio_queue.put(current_audio.getvalue())
                await self.audio_queue.put(None)
            except Exception as e:
                logging.error(f"音频流处理错误: {e}")

        async def audio_player():
            await self.start_audio_player()
            try:
                while True:
                    chunk = await self.audio_queue.get()
                    if chunk is None:
                        break
                    try:
                        self.mpg123_process.stdin.write(chunk)
                        self.mpg123_process.stdin.flush()
                    except Exception as e:
                        logging.error(f"音频播放错误: {e}")
            except Exception as e:
                logging.error(f"音频播放任务错误: {e}")

        await asyncio.gather(audio_stream_handler(), audio_player())
        print(f"TTS 处理耗时: {time.time() - start_time:.2f}秒")

    async def speak(self, text):
        self.is_speaking = True
        try:
            await self.stream_tts(text)
        except Exception as e:
            logging.error(f"语音合成失败: {e}")
        finally:
            self.is_speaking = False

    async def shutdown(self):
        """清理资源"""
        await self.stop_audio_player()

