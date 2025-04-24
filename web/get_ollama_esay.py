import json
import os
import logging
from datetime import datetime
from typing import List, Dict, Any
import edge_tts
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM
import nest_asyncio
import time
import asyncio
nest_asyncio.apply()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("chat.log"), logging.StreamHandler()]
)

class KnowledgeQA:
    def __init__(
        self,
        knowledge_path = "knowledge.json",
        faiss_index_path = "../faiss_index",
        llm_model = "qwen2.5:7b",
    ):
        """
        初始化知识问答系统，加载对话历史、嵌入模型、向量库和问答链。
        """
        # self.knowledge_path = knowledge_path
        self.faiss_index_path = faiss_index_path
        self.llm_model = llm_model
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="../bge-base-zh-v1.5",
            encode_kwargs={'normalize_embeddings': True}
        )
        # self.embedding_model = self._init_embeddings()
        self.vectorstore = self.load_or_create_vectorstore()
        self.qa_chain = self.init_qa_chain()

    # def _init_embeddings(self):
    #     """
    #     初始化本地 HuggingFace 向量模型（用于文本向量化）。
    #     """
    #     return HuggingFaceEmbeddings(
    #         model_name="./bge-base-zh-v1.5",
    #         encode_kwargs={'normalize_embeddings': True}
    #     )

    

    # def load_data(self):
    #     """
    #     加载本地知识库（JSON 格式），返回问答对组成的列表。
    #     """
    #     with open(self.knowledge_path, 'r', encoding='utf-8') as f:
    #         return json.load(f)

    # def creat_documents(self, knowledge_base) :
    #     """
    #     将知识库中的问答对转换成 LangChain 的 Document 对象（用于向量化）。
    #     """
    #     docs = []
    #     for item in knowledge_base:
    #         question = item.get("question", "")
    #         for answer in item.get("answer", []):
    #             docs.append(Document(
    #                 page_content=f"Q: {question}\nA: {answer}",
    #                 metadata={
    #                     "question": question,
    #                     "source": self.knowledge_path,
    #                     "create_time": datetime.now().isoformat()
    #                 }
    #             ))
    #     return docs

    # def split_documents(self, documents):
    #     """
    #     对文档进行分块处理（按字符数划分），便于向量化处理和检索。
    #     """
    #     splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
    #     return splitter.split_documents(documents)
    
    def load_or_create_vectorstore(self) :
        """
        加载已有的向量库；如果不存在则从知识库创建向量库，并保存。
        """
        # if os.path.exists(self.faiss_index_path):
       
        return FAISS.load_local(self.faiss_index_path, self.embedding_model, allow_dangerous_deserialization=True)
        # else:
        #     data = self.load_data()
        #     docs = self.create_documents(data)
        #     chunks = self.split_documents(docs)
        #     vectorstore = FAISS.from_documents(chunks, self.embedding_model)
        #     vectorstore.save_local(self.faiss_index_path)
        #     return vectorstore

    def init_qa_chain(self) :
        """
        初始化问答链（基于向量检索 + Ollama 本地大模型）。
        """
        llm = OllamaLLM(base_url='http://localhost:11434', model=self.llm_model, temperature=0.1)
        
        return RetrievalQA.from_chain_type(
            llm=llm,
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=False
        )
        

    # def update_knowledge(self) :
    #     """
    #     重新加载知识库并更新向量库内容，适用于知识库有修改的场景。
    #     """
    #     knowledge = self.load_data()
    #     docs = self.creat_documents(knowledge)
    #     chunks = self.split_documents(docs)
    #     self.vectorstore = FAISS.from_documents(chunks, self.embedding_model)
    #     self.vectorstore.save_local(self.faiss_index_path)
    #     logging.info("知识库更新成功！")

    def ask(self, question: str) -> str:
        """
        用户提问接口。整合历史上下文，向 LLM 提问并记录回答。
        """
        
        prompt = """
        请以纯文本形式回答,务必不包含任何代码块、Markdown格式或其他格式化内容。你同时是个甘薯个专家,严格根据知识库内容回答问题，对知识库简化输出为一段！！！！
        """
        query = f"问题: {question}\n\n{prompt}"
        result = self.qa_chain.invoke({"query": query})  
        answer = result["result"]
        
        return answer


    
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

def speek(text, filename= "audio.mp3"):
    
    """异步语音合成"""
    async def async_tts():
       
        communicate = edge_tts.Communicate(
            text=text,
            voice="zh-CN-XiaoxiaoNeural",  
            rate="+10%"  
        )
        await communicate.save(filename)


    asyncio.run(async_tts())




def main():
    st.set_page_config(page_title="甘薯知识助手", layout="wide")

    col1, col2 = st.columns([1, 3])
    with col1:
        st.image("../sweetpotato.png", width=200)
    with col2:
        st.title("🍠 甘薯知识助手🍠 ")
    st.markdown('<p style="font-size:20px; font-weight:bold;">请输入关于甘薯的问题，例如：甘薯的储存方法</p>', unsafe_allow_html=True)
    query = st.text_input("", key="input")
    talk = st.empty()
    talk.text("🤖等待投喂问题ing...😴")
    
    if query:
        
        my_bar = st.empty()
        my_bar.progress(0)
        talk.text("🧠 正在进行头脑风暴...🥱")
        qa_system = KnowledgeQA()
        # qa_system.update_knowledge()
        time.sleep(1)
        my_bar.progress(30)
        talk.text("😈好像找到答案了？！🤔")
        my_bar.progress(60)
        answer = qa_system.ask(query)
        processed_answer = preprocess_text(answer)
        talk.text("🎉 答案已找到！😻")
        my_bar.progress(90)
        st.markdown(f"### 答案\n{answer}")

        talk.text("🔊 生成语音中...")
        
        speek(processed_answer)
        st.audio("audio.mp3")
        my_bar.progress(100)

        if os.path.exists("audio.mp3"):
            os.remove("audio.mp3")
        my_bar.empty()
        talk.empty()

if __name__ == "__main__":
    main()