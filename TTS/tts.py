import sys
import os
sys.path.append(os.path.abspath("/home/wuye/vscode/chatbox"))
from qa_model.qa_model_easy import KnowledgeQA
import asyncio
import edge_tts
import re
import os
import subprocess
import logging
from qa_model.qa_model_easy import KnowledgeQA
import pyaudio


class TTSHelper:
    def __init__(self, voice, rate, volume):
        self.voice = voice
        self.rate = rate    # 语速
        self.volume = volume  # 音量
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=24000,  # edge-tts 默认采样率
            output=True
        )

    def preprocess_text(self, text):
        """
        预处理文本，替换不标准的标点并清理可能导致问题的字符
        """
        text = text.replace("，", ",")
        text = text.replace("。", ",")
        text = text.replace("、", ",")
        text = text.replace("；", ",")
        text = text.replace("：", ",")
        text = text.replace("！", ",")
        text = text.replace("？", ",")
        text = text.replace("“", ',')
        text = text.replace("”", ',')
        text = text.replace("‘", ',')
        text = text.replace("’", ',')
        text = text.replace("《", ',')
        text = text.replace("》", ',')
        text = re.sub(r'[\x00-\x1F\x7F]', '', text)
 
        # print(f"预处理后的文本：{text}")
        return text

    async def speak(self, text):
        """
        异步输出语音，使用 edge-tts 生成音频文件并通过系统播放器播放
        """
        processed_text = self.preprocess_text(text)
        if  processed_text==None:
            logging.warning("文本为空，跳过语音播放‼️")
            return
        print(50*"*","开始回答啦！😁😁😁😁",50*"*")
        print(f"{processed_text}")
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
            # print(f"音频已保存到：{output_file}")

            # 使用 mpg123 播放音频
            result = subprocess.run(["mpg123", output_file], capture_output=True, text=True)
            if result.returncode != 0:
                print(f"mpg123 播放失败：{result.stderr}")
                raise RuntimeError(f"mpg123 播放失败：{result.stderr}")
            print(f"语音播放完成：{processed_text}")

            # 删除临时音频文件
            os.remove(output_file)
            # print(f"已删除临时文件：{output_file}")
        except Exception as e:
            logging.error(f"语音播放失败：{e}")
            raise


    async def text_to_speech(self, text):
        """
        文本转语音的入口，调用语音播放
        """
        await self.speak(text)

#以上是直接生成mp3文件的语音播放方法


    def play_audio_chunk(self, chunk):
        """
        实时播放音频数据
        """
        pyaudio_instance = pyaudio.PyAudio()
        stream = pyaudio_instance.open(
            format=pyaudio.paInt16,  # 16-bit audio format
            channels=1,  # 单声道
            rate=16000,  # 采样率
            output=True
        )
        stream.write(chunk)
    

    

 



# if __name__ == "__main__":
#     async def test():
#         asr=ASRhelper()
#         question=asr.main()
#         qa=KnowledgeQA()
        
#         answer=qa.ask(question)
#         tts_helper = TTSHelper(voice="zh-CN-XiaoyiNeural", rate="+0%", volume="+0%")

#         await tts_helper.text_to_speech('11'+answer)

#     asyncio.run(test())


async def main():
    qa=KnowledgeQA()
    tts=TTSHelper(voice="zh-CN-XiaoyiNeural", rate="+0%", volume="+0%")
    question="甘薯的贮藏特性"
    answer=qa.ask(question)
    # print(answer)
    await tts.text_to_speech('11'+answer)


if __name__=="__main__":
    asyncio.run(main())
    