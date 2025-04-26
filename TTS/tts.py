import sys
import os
sys.path.append(os.path.abspath("/home/wuye/vscode/chatbox"))
import asyncio
import edge_tts
import re
import os
import subprocess
import logging
import time

class TTSHelper:
    def __init__(self, voice="zh-CN-XiaoyiNeural", rate="+0%", volume="+0%"):
        self.voice = voice
        self.rate = rate    # 语速
        self.volume = volume  # 音量
        self.max_retries = 3  # 最大重试次数
        self.retry_delay = 1  # 重试间隔秒数
        self.is_speaking = False  # 是否正在说话

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
 
        return text

    async def speak(self, text):
        """
        异步输出语音，使用 edge-tts 生成音频文件并通过系统播放器播放
        增加错误处理和重试机制
        """
        processed_text = self.preprocess_text(text)
        if not processed_text:
            logging.warning("文本为空，跳过语音播放‼️")
            return
            
        # print(50*"*","开始回答啦！😁😁😁😁",50*"*")
        # print(f"{processed_text}")
        
        # 设置为正在说话状态
        self.is_speaking = True
        
        # 重试循环
        for attempt in range(self.max_retries):
            try:
                # 创建 edge-tts Communicate 实例
                communicate = edge_tts.Communicate(
                    text=processed_text,
                    voice=self.voice,
                    rate=self.rate,
                    volume=self.volume
                )
                
                # 保存音频到临时文件
                output_file = f"temp_audio_{id(self)}_{int(time.time())}.mp3"  # 唯一文件名避免冲突
                
                # 尝试保存音频
                await communicate.save(output_file)
                
                # 使用 mpg123 播放音频
                result = subprocess.run(["mpg123", "-q", output_file], capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"mpg123 播放失败：{result.stderr}")
                    raise RuntimeError(f"mpg123 播放失败：{result.stderr}")
                    
                # print(f"语音播放完成：{processed_text}")
                
                # 删除临时音频文件
                try:
                    os.remove(output_file)
                except Exception as e:
                    print(f"删除临时文件失败：{e}")
                
                # 成功退出重试循环
                break
                
            except Exception as e:
                # 记录错误
                logging.error(f"TTS尝试 {attempt+1}/{self.max_retries} 失败: {e}")
                
                # 如果是最后一次尝试，使用备用方法或者打印错误
                if attempt == self.max_retries - 1:
                    print(f"语音生成多次失败，将直接显示文本: {processed_text}")
                    # 这里可以实现备用TTS方法，比如本地TTS或其他服务
                    # 或者只显示文本
                    break
                    
                # 等待一段时间后重试
                await asyncio.sleep(self.retry_delay)
        
        # 设置为结束说话状态
        self.is_speaking = False

    async def text_to_speech(self, text, wait=True):
        """
        文本转语音的入口，调用语音播放
        增加了等待选项
        """
        task = asyncio.create_task(self.speak(text))
        
        if wait:
            await task
        
        return task

    async def wait_until_done(self):
        """
        等待直到所有语音播放完成
        """
        # 等待直到不再处于说话状态
        while self.is_speaking:
            await asyncio.sleep(0.1)


# 备用的简易TTS函数，当Edge TTS服务不可用时使用
async def fallback_print_text(text):
    """
    当TTS失败时的备用方法：仅打印文本
    """
    clean_text = text
    if text.startswith("11"):
        clean_text = text[2:]  # 移除前缀
    
    print("\n" + "="*50)
    print(f"语音合成失败，显示文本: {clean_text}")
    print("="*50 + "\n")
    # 模拟语音播放时间
    await asyncio.sleep(len(clean_text) * 0.05)  # 假设每个字符需要0.05秒读出
    return True

async def main():
    tts = TTSHelper(voice="zh-CN-XiaoyiNeural", rate="+0%", volume="+0%")
    await tts.text_to_speech("11这是一个测试语音。如果您能听到这段话，说明语音系统正常工作。")

if __name__ == "__main__":
    asyncio.run(main())