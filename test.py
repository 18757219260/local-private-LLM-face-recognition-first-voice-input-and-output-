
import asyncio
import edge_tts
import pygame
import io
from edge_tts import VoicesManager
from collections import deque
from concurrent.futures import ThreadPoolExecutor

class EdgeTTSStreaming:
    def __init__(self):
        self.voice = "zh-CN-XiaoyiNeural"
        pygame.mixer.init(frequency=24000)  # 设置更适合的采样率
        self.is_speaking = False
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.audio_buffer = deque(maxlen=50)  # 使用双端队列缓存音频
        self.buffer_ready = asyncio.Event()
        self.min_buffer_size = 6400  # 增加缓冲区大小

    async def stream_tts(self, text):
        communicate = edge_tts.Communicate(text, self.voice)
        audio_queue = asyncio.Queue()
        
        async def audio_stream_handler():
            current_audio = io.BytesIO()
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    current_audio.write(chunk["data"])
                    # 积累更多数据再处理
                    if current_audio.tell() > self.min_buffer_size:
                        current_audio.seek(0)
                        await audio_queue.put(current_audio.getvalue())
                        current_audio = io.BytesIO()
            
            # 处理最后的音频片段
            if current_audio.tell() > 0:
                current_audio.seek(0)
                await audio_queue.put(current_audio.getvalue())
            await audio_queue.put(None)

        async def buffer_manager():
            """预先缓存一定数量的音频片段"""
            buffer_threshold = 3  # 开始播放前预缓存的片段数
            
            while True:
                chunk = await audio_queue.get()
                if chunk is None:
                    break
                
                self.audio_buffer.append(chunk)
                if len(self.audio_buffer) >= buffer_threshold:
                    self.buffer_ready.set()
                
                # 控制缓冲区大小
                if len(self.audio_buffer) > 10:
                    await asyncio.sleep(0.1)

        async def audio_player():
            """平滑播放音频"""
            await self.buffer_ready.wait()  # 等待缓冲区准备好
            
            while True:
                if not self.audio_buffer:
                    if not self.is_speaking:
                        break
                    await asyncio.sleep(0.05)
                    continue
                
                chunk = self.audio_buffer.popleft()
                
                def play_audio(audio_chunk):
                    try:
                        sound = pygame.mixer.Sound(io.BytesIO(audio_chunk))
                        sound.play()
                        # 等待当前片段接近播放完成再播放下一个
                        duration = sound.get_length() * 0.95  # 95%的播放时间
                        pygame.time.wait(int(duration * 1000))
                    except Exception as e:
                        print(f"播放错误: {e}")
                
                # 在线程池中执行播放
                await asyncio.get_event_loop().run_in_executor(
                    self.executor, 
                    play_audio,
                    chunk
                )

        # 同时运行所有协程
        await asyncio.gather(
            audio_stream_handler(),
            buffer_manager(),
            audio_player()
        )

    def speak(self, text):
        """开始语音合成和播放"""
        if self.is_speaking:
            return
        
        self.is_speaking = True
        self.audio_buffer.clear()
        self.buffer_ready.clear()
        
        try:
            asyncio.run(self.stream_tts(text))
        finally:
            self.is_speaking = False

# 使用示例
if __name__ == "__main__":
    tts = EdgeTTSStreaming()
    
    # 测试长文本
    text = """前面五点更新都是关于右键菜单的，这也是Cool Papers扩展程序最主要的功能，它的工作逻辑是：依次检查用户右击鼠标时的所选文字、超链接、页面路径中是否包含论文ID，如果包含则跳转到Cool Papers相应页面。

v0.1.0版写在Cool Papers建立之初，当时只支持识别单一的arXiv论文ID，现在支持识别多个论文ID在同一页打开，并且支持arXiv、OpenReview、ACL、IJCAI、PMLR五个论文源（后面再陆续增加，这几个是图方便先补充上去）。还有一点比较关键的改进，就是之前在PDF页面是无法进行跳转的，比如你访问了arXiv的PDF如“https://arxiv/org/pdf/xxxx.xxxxx”，那么右击时就会发现“Redirect to Cool Papers”根本不显示，这是旧版遗留的一个问题，新版把它解决了。

最后，如果在所选文字、超链接、页面路径中都检测不到论文ID，那么就直接以所选文字为Query，跳转到Cool Papers的站内搜索页，这时候就变成了一个划词搜索功能了。。"""
    
    tts.speak(text)

