import asyncio
import edge_tts
import subprocess
import io
import logging
from collections import deque
from qa_model_easy import KnowledgeQA
import time

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("chat.log")]
)

class EdgeTTSStreaming:
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
        # text = re.sub(r'[\x00-\x1F\x7F]', '', text)
        # text = text.strip("，。！？")
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
            logging.info("mpg123 进程启动")

    async def stop_audio_player(self):
        """关闭 mpg123 进程"""
        if self.mpg123_process:
            self.mpg123_process.stdin.close()
            self.mpg123_process.wait()
            self.mpg123_process = None
            logging.info("mpg123 进程关闭")

    async def stream_tts(self, text):
        if not text.strip():
            logging.warning("文本为空，跳过语音合成")
            return
        start_time = time.time()
        communicate = edge_tts.Communicate(text, self.voice)

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
        logging.info(f"TTS 处理耗时: {time.time() - start_time:.2f}秒")

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

async def main():
    # 初始化 QA 和 TTS
    qa = KnowledgeQA()
    tts = EdgeTTSStreaming(voice="zh-CN-XiaoyiNeural")
    
    # 提问
    question = "甘薯的贮藏特性"
    logging.info(f"提问: {question}")

    # 流式处理回答并实时语音输出
    buffer = ""
    min_length = 20  # 最小文本长度
    trigger_length = 40  # 触发语音的字符数
    timeout = 0.5  # 0.5秒超时触发语音
    last_chunk_time = time.time()

    async def text_producer():
        """从 ask_stream 获取文本并放入队列"""
        async for chunk in qa.ask_stream(question):
            logging.info(f"🧠 输出块: {chunk}")
            yield chunk
      
    async def text_consumer():
        """处理文本并触发语音"""
        nonlocal buffer, last_chunk_time
        async for chunk in text_producer():
            chunk=tts.preprocess_text(chunk)  # 预处理文本
            buffer += chunk
            last_chunk_time = time.time()

            # 触发语音：达到 trigger_length 或超时
            if len(buffer) >= trigger_length or (time.time() - last_chunk_time) >= timeout:
                if len(buffer) >= min_length:
                    logging.info(f"语音输出: {buffer}")
                    await tts.speak(buffer)
                    buffer = ""  # 清空缓冲区

        # 处理剩余文本
        if buffer and len(buffer) >= min_length:
            logging.info(f"语音输出（剩余）: {buffer}")
            await tts.speak('111'+buffer)

    try:
        await text_consumer()
    finally:
        await tts.shutdown()

if __name__ == "__main__":
    asyncio.run(main())