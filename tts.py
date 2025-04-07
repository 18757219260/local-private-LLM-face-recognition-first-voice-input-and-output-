import asyncio
import io
import pygame
import edge_tts
import speech_recognition as sr

class VoiceAssistant:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        pygame.mixer.init()

    async def speek(self, text: str):
        communicate = edge_tts.Communicate(
            text=text,
            voice="zh-CN-XiaoxiaoNeural",
            rate="+10%"
        )
        audio_stream = io.BytesIO()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_stream.write(chunk["data"])
        audio_stream.seek(0)
        pygame.mixer.music.load(audio_stream)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            await asyncio.sleep(0.1)

    def listen(self):
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
            try:
                audio = self.recognizer.listen(source, timeout=5)
                message=self.recognizer.recognize_google(audio, language="zh-CN")
                print("用户提问：\n"+message)
                return message
            except:
                return None
