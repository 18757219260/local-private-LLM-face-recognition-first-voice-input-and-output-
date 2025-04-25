import sys
import os
sys.path.append(os.path.abspath("/home/wuye/vscode/chatbox"))
import asyncio
from qa_model.qa_model_easy import KnowledgeQA
from ASR.asr import ASRhelper
from TTS.tts_stream import TTStreaming
import time
import argparse
import signal
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("chatbox.log"), logging.StreamHandler()]
)

async def main():
    # 初始化 QA 和 TTS
    parser = argparse.ArgumentParser(description="甘薯知识问答系统")
    parser.add_argument("--voice", default="zh-CN-XiaoyiNeural", help="TTS voice to use")
    parser.add_argument("--model", default="qwen2.5:7b", help="LLM model to use")
    parser.add_argument("--wait-after-response", type=float, default=1.5, 
                      help="等待等模型说完再次开始监听的时")
    args = parser.parse_args()
    
    # 控制是否正在说话的标志
    is_speaking = False
    
    try:
        qa = KnowledgeQA(llm_model=args.model)
        tts = TTStreaming(voice=args.voice)
        asr = ASRhelper()
        
        print("甘薯知识问答系统已启动！按 Ctrl+C 退出")
        
        # 设置优雅关闭
        shutdown_event = asyncio.Event()
        
        def signal_handler(*args):
            print("\n正在关闭系统...")
            shutdown_event.set()
            
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, lambda s, f: asyncio.create_task(asyncio.to_thread(signal_handler)))
        
        while not shutdown_event.is_set():
            buffer = ""
            min_length = 5  # 最小文本长度
            trigger_length = 40  # 触发语音的字符数
            timeout = 0.5  # 0.5秒超时触发语音
            
            # 确保系统不在说话时才开始语音识别
            print("\n请开始说话...")
            question_result = asr.real_time_recognition()
            
            # 检查语音识别结果
            if not question_result or 'result' not in question_result or not question_result['result']:
                print("无输入，请重新尝试")
                await asyncio.sleep(1)
                continue
            
            question = question_result["result"][0]
            print(f"用户问题: {question}")
            
            # 检测特殊命令
            if question.lower() in ["退出", "关闭", "停止"]:
                print("收到退出指令")
                break
            
            # 设置正在说话标志
            is_speaking = True
            tts_complete_event = asyncio.Event()
            
            # 文本生成器
            async def text_producer():
                try:
                    async for chunk in qa.ask_stream(question):
                        print(f"🧠 输出块: {chunk}")
                        yield chunk
                except Exception as e:
                    logging.error(f"生成回答时出错: {e}")
                    yield "抱歉，处理您的问题时出现了错误。"
            
            # 文本处理和TTS
            async def text_consumer():
                nonlocal buffer
                last_chunk_time = time.time()
                chunks_received = False
                
                try:
                    async for chunk in text_producer():
                        chunks_received = True
                        chunk = tts.preprocess_text(chunk)
                        buffer += chunk
                        last_chunk_time = time.time()
                        
                        # 触发语音：达到指定长度或超时
                        if len(buffer) >= trigger_length or (time.time() - last_chunk_time) >= timeout:
                            if len(buffer) >= min_length:
                                segment = buffer
                                buffer = ""
                                print(f"语音输出: {segment}")
                                await tts.speak(segment)  # 移除'11'前缀，除非确实需要
                    
                    # 处理剩余文本
                    if buffer and len(buffer) >= min_length:
                        print(f"语音输出（剩余）: {buffer}")
                        await tts.speak(buffer)  # 移除'11'前缀
                    
                    # 如果没有收到任何内容
                    if not chunks_received:
                        await tts.speak("抱歉，我没能找到相关的甘薯知识。")
                        
                except Exception as e:
                    logging.error(f"处理文本时出错: {e}")
                    await tts.speak("抱歉，处理您的问题时出现了错误。")
                finally:
                    # 无论如何都需要重置状态
                    tts_complete_event.set()
            
            # 执行文本消费者任务
            await text_consumer()
            
            # 标记语音已结束
            is_speaking = False
            
            # 等待额外的时间，确保用户有时间理解响应
            logging.info(f"等待 {args.wait_after_response} 秒后再次开始监听...")
            await asyncio.sleep(args.wait_after_response)
            
    except Exception as e:
        logging.error(f"系统运行时出错: {e}")
    finally:
        # 清理资源
        try:
            await tts.shutdown()
            asr.stop_recording()
            print("程序已安全退出，资源已清理")
        except Exception as e:
            logging.error(f"清理资源时出错: {e}")

if __name__ == "__main__":
    asyncio.run(main())