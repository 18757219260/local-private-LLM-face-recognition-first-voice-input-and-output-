import asyncio
import edge_tts
import re
import os
import subprocess
import logging
from qa_model_easy import KnowledgeQA
from asr import ASRhelper

class RealTimeTTSHelper:
    def __init__(self, voice, rate, volume):
        self.voice = voice
        self.rate = rate    # 语速
        self.volume = volume  # 音量

    def preprocess_text(self, text):
        """
        预处理文本，替换不标准的标点并清理可能导致问题的字符
        """
        text = text.replace("，", ",")
        text = text.replace("。", ",")
        text = re.sub(r'[\x00-\x1F\x7F]', '', text)
        # text = text.strip("，。！？")
        print(f"预处理后的文本：{text}")
        return text

    async def speak(self, text):
        """
        异步输出语音，使用 edge-tts 生成音频文件并通过系统播放器播放
        """
        processed_text = self.preprocess_text(text)
        if not processed_text:
            logging.warning("文本为空，跳过语音播放")
            return
        print(f"开始播放语音：{processed_text}")
        try:
            # 创建 edge-tts Communicate 实例
            communicate = edge_tts.Communicate(
                text=processed_text,
                voice=self.voice,
                rate=self.rate,
                volume=self.volume
            )
            # 保存音频到临时文件
            output_file = f"temp_audio_{id(self)}.mp3"  # 唯一文件名避免冲突
            await communicate.save(output_file)
            print(f"音频已保存到：{output_file}")

            # 使用 mpg123 播放音频
            result = subprocess.run(["mpg123", output_file], capture_output=True, text=True)
            if result.returncode != 0:
                logging.error(f"mpg123 播放失败：{result.stderr}")
                raise RuntimeError(f"mpg123 播放失败：{result.stderr}")
            print(f"语音播放完成：{processed_text}")

            # 删除临时音频文件
            os.remove(output_file)
            print(f"已删除临时文件：{output_file}")
        except Exception as e:
            logging.error(f"语音播放失败：{e}")
            raise

    async def text_to_speech(self, text):
        """
        文本转语音的入口，调用语音播放
        """
        await self.speak(text)

    async def main(self, text_stream):
        """
        主逻辑方法，接收大模型的流式文本（异步生成器）并实时语音播放
        """
        buffer = ""  # 累积文本缓冲区
        sentence_end_marks = {"。", "！", "？"}  # 句子结束标点
        min_length = 5  # 最小文本长度触发语音
        max_length = 50  # 最大缓冲长度触发语音

        async for chunk in text_stream:
            print(f"接收到文本块：{chunk}")
            buffer += chunk

            # 检查是否需要触发语音
            while buffer:
                # 查找句子结束标点
                sentence_end = -1
                for mark in sentence_end_marks:
                    pos = buffer.find(mark)
                    if pos != -1 and (sentence_end == -1 or pos < sentence_end):
                        sentence_end = pos

                # 如果找到句子结束标点或缓冲区过长
                if sentence_end != -1:
                    # 提取完整句子（包括标点）
                    sentence = buffer[:sentence_end + 1]
                    buffer = buffer[sentence_end + 1:]
                    if len(sentence) >= min_length:
                        await self.text_to_speech(sentence)
                elif len(buffer) >= max_length:
                    # 如果缓冲区过长，截取并播放
                    sentence = buffer[:max_length]
                    buffer = buffer[max_length:]
                    await self.text_to_speech(sentence)
                else:
                    break  

        if buffer and len(buffer) >= min_length:
            await self.text_to_speech(buffer)


if __name__ == "__main__":
    async def test():
        asr=ASRhelper()
        question=asr.main()
        qa=KnowledgeQA()
        
        answer=qa.ask(question)
        tts_helper = RealTimeTTSHelper(voice="zh-CN-XiaoyiNeural", rate="+0%", volume="+0%")

        await tts_helper.text_to_speech('11'+answer)

    asyncio.run(test())