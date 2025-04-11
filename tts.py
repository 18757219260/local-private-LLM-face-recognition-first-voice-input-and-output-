import asyncio
import io
import pygame
import edge_tts
import speech_recognition as sr
from qa_model import KnowledgeQA

class VoiceAssistant:
    def __init__(self, voice="zh-CN-XiaoxiaoNeural"):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.voice = voice
        pygame.mixer.init()

    async def speak(self, text: str):
        communicate = edge_tts.Communicate(
            text=text,
            voice=self.voice,
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

    def listen(self, timeout=6, phrase_time_limit=8):
        with self.microphone as source:
            print("🎤 请开始说话...")
            self.recognizer.adjust_for_ambient_noise(source, duration=0.8)
            try:
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
                message = self.recognizer.recognize_google(audio, language="zh-CN")
                print("✅ 用户说：", message)
                return message
            except sr.WaitTimeoutError:
                print("⌛ 没有检测到语音输入。")
            except sr.UnknownValueError:
                print("🤔 无法识别语音内容。")
            except sr.RequestError as e:
                print(f"❌ 请求识别服务出错: {e}")
            return None

    async def chat_once(self):
        question = self.listen()
        if question:
            qa= KnowledgeQA()
            answer = qa.ask(question)
            question = f"你刚才说的是：{question}"
            print("用户提问：", question)
            answer = f"回答是：{answer}"
            print("🤖 助手回答：", answer)
            await self.speak('11'+answer)

if __name__ == "__main__":
    va = VoiceAssistant()
    asyncio.run(va.chat_once())