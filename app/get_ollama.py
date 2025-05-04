import streamlit as st
import time
import os
import asyncio
import edge_tts
import sys
sys.path.append(os.path.abspath("/home/wuye/vscode/chatbox"))
from qa_model.qa_model import KnowledgeQA

def preprocess_text( text):
    """
    预处理文本，替换不标准的标点并清理可能导致问题的字符
    """
    text = text.replace("，", ",")
    text = text.replace("。", ",")
    text = text.replace("、", ",")
    # text = re.sub(r'[\x00-\x1F\x7F]', '', text)
    text = text.strip("，。！？")
    # print(f"预处理后的文本：{text}")
    return text

async def async_tts(text: str, filename: str = "audio.mp3"):
    """
    异步语音合成
    
    参数:
        text: 要转换为语音的文本
        filename: 输出音频文件名
    """
    communicate = edge_tts.Communicate(
        text=text,
        voice="zh-CN-XiaoxiaoNeural",
        rate="+10%"
    )
    await communicate.save(filename)

def speak(text: str, filename: str = "audio.mp3"):
    """
    语音合成包装函数
    
    参数:
        text: 要转换为语音的文本
        filename: 输出音频文件名
    """
    asyncio.run(async_tts(text, filename))

def main():
    """主函数，定义Streamlit应用界面"""
    
    # 页面配置
    st.set_page_config(page_title="甘薯知识助手", layout="wide")

    # 页面标题和图片
    col1, col2 = st.columns([1, 3])
    with col1:
        st.image("sweetpotato.png", width=200)
    with col2:
        st.title("🍠 甘薯知识助手 🍠")
    
    # 提示文本
    st.markdown('<p style="font-size:20px; font-weight:bold;">请输入关于甘薯的问题，例如：甘薯的储存方法</p>', unsafe_allow_html=True)
    
    # 用户输入框
    query = st.text_input("", key="input")
    
    # 状态提示
    talk = st.empty()
    talk.text("🤖等待投喂问题ing...😴")
    
    # 当用户输入问题时
    if query:
        # 进度条
        my_bar = st.empty()
        my_bar.progress(0)
        
        # 状态更新
        talk.text("🧠 正在进行头脑风暴...🥱")
        
        try:
            # 初始化问答系统
            qa_system = KnowledgeQA(
                knowledge_path="knowledge.json",
                faiss_index_path="faiss_index",
                llm_model="qwen2.5:7b"
            )
            
            # 检查并更新知识库（如果需要）
            qa_system.vector_manager.check_and_update_if_needed()
            
            time.sleep(1)
            my_bar.progress(30)
            talk.text("😈好像找到答案了？！🤔")
            my_bar.progress(60)
            
            # 获取答案
            response = qa_system.ask(query)
            answer = response["answer"]
            answer = preprocess_text(answer)
            sources = response["sources"]
            
            talk.text("🎉 答案已找到！😻")
            my_bar.progress(90)
            
            # 显示答案
            st.markdown(f"### 答案\n{answer}")
            
            # 显示参考来源（如果有）
            if sources:
                st.markdown("### 参考来源")
                for src in sources:
                    st.markdown(f"- {src['question']}")
            
            # 语音合成
            talk.text("🔊 生成语音中...")
            speak(answer)
            st.audio("audio.mp3")
            my_bar.progress(100)
            
            # 清理临时音频文件
            if os.path.exists("audio.mp3"):
                os.remove("audio.mp3")
            
            # 清空进度条和状态提示
            my_bar.empty()
            talk.empty()
            
        except Exception as e:
            # 错误处理
            st.error(f"发生错误: {str(e)}")
            my_bar.empty()
            talk.text("😿 很抱歉，出现了错误，请重试")

if __name__ == "__main__":
    main()