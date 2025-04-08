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
import asyncio
import nest_asyncio
import time


nest_asyncio.apply()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("chat.log"), logging.StreamHandler()]
)

class KnowledgeQA:
    def __init__(
        self,
        knowledge_path: str = "knowledge.json",
        faiss_index_path: str = "faiss_index",
        llm_model: str = "qwen2.5:7b",
        history_log: str = "chat_history.json",
    ):
        """
        初始化知识问答系统，加载对话历史、嵌入模型、向量库和问答链。
        """
        self.knowledge_path = knowledge_path
        self.faiss_index_path = faiss_index_path
        self.llm_model = llm_model
        self.history_log = history_log

        self.conversation_history = self._load_history()
        self.embedding_model = self._init_embeddings()
        self.vectorstore = self._load_or_create_vectorstore()
        self.qa_chain = self._init_qa_chain()

    def _init_embeddings(self) -> HuggingFaceEmbeddings:
        """
        初始化本地 HuggingFace 向量模型（用于文本向量化）。
        """
        return HuggingFaceEmbeddings(
            model_name="./bge-base-zh-v1.5",
            encode_kwargs={'normalize_embeddings': True}
        )

    def _load_knowledge(self) -> List[Dict[str, Any]]:
        """
        加载本地知识库（JSON 格式），返回问答对组成的列表。
        """
        with open(self.knowledge_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _create_documents(self, knowledge_base: List[Dict]) :
        """
        将知识库中的问答对转换成 LangChain 的 Document 对象（用于向量化）。
        """
        docs = []
        for item in knowledge_base:
            question = item.get("question", "")
            for answer in item.get("answer", []):
                docs.append(Document(
                    page_content=f"Q: {question}\nA: {answer}",
                    metadata={
                        "question": question,
                        "source": self.knowledge_path,
                        "create_time": datetime.now().isoformat()
                    }
                ))
        return docs

    def _split_documents(self, documents: List[Document]):
        """
        对文档进行分块处理（按字符数划分），便于向量化处理和检索。
        """
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
        return splitter.split_documents(documents)
    
    def _load_or_create_vectorstore(self) -> FAISS:
        """
        加载已有的向量库；如果不存在则从知识库创建向量库，并保存。
        """
        if os.path.exists(self.faiss_index_path):
            return FAISS.load_local(self.faiss_index_path, self.embedding_model, allow_dangerous_deserialization=True)
        else:
            knowledge = self._load_knowledge()
            docs = self._create_documents(knowledge)
            chunks = self._split_documents(docs)
            vectorstore = FAISS.from_documents(chunks, self.embedding_model)
            vectorstore.save_local(self.faiss_index_path)
            return vectorstore

    def _init_qa_chain(self) -> RetrievalQA:
        """
        初始化问答链（基于向量检索 + Ollama 本地大模型）。
        """
        llm = OllamaLLM(base_url='http://localhost:11434', model=self.llm_model, temperature=0.3)
        
        return RetrievalQA.from_chain_type(
            llm=llm,
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=False
        )
        
        
    def _load_history(self) :
        """
        加载历史对话记录（如文件存在）；否则返回空列表。
        """
        if os.path.exists(self.history_log):
            try:
                with open(self.history_log, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return []
        return []

    def _save_history(self) :
        """
        将对话记录保存到本地文件（最多保存最新100条）。
        """
        with open(self.history_log, 'w', encoding='utf-8') as f:
            json.dump(self.conversation_history[-100:], f, ensure_ascii=False, indent=2)

    def update_knowledge(self) :
        """
        重新加载知识库并更新向量库内容，适用于知识库有修改的场景。
        """
        knowledge = self._load_knowledge()
        docs = self._create_documents(knowledge)
        chunks = self._split_documents(docs)
        self.vectorstore = FAISS.from_documents(chunks, self.embedding_model)
        self.vectorstore.save_local(self.faiss_index_path)
        logging.info("知识库更新成功！")

    def ask(self, question: str) -> str:
        """
        用户提问接口。整合历史上下文，向 LLM 提问并记录回答。
        """
        
        prompt = """
        请以纯文本形式回答，务必不包含任何代码块、Markdown格式或其他格式化内容。你同时是个甘薯个专家，严格根据知识库内容回答问题。
        """

        knowledge_last_modified = os.path.getmtime(self.knowledge_path)
        if hasattr(self, "last_knowledge_update") and self.last_knowledge_update != knowledge_last_modified:
            logging.info("知识库有更新，正在重新加载...")
            self.update_knowledge()
            self.last_knowledge_update = knowledge_last_modified  # 更新最后更新时间
        history = "".join(
            f"[{h.get('timestamp', 'N/A')}] User: {h.get('user', '')}\nBot: {h.get('bot', '')}"
            for h in self.conversation_history[-5:]
        )
        full_query = f"Conversation History:\n{history}\n\n新问题: {question}\n\n{prompt}"
        result = self.qa_chain.invoke({"query": full_query})  
        answer = result["result"]

        # 保存对话历史
        record = {
            "timestamp": datetime.now().isoformat(),
            "user": question,
            "bot": answer
        }
        self.conversation_history.append(record)
        self._save_history()
        return answer

    


def speek(text: str, filename: str = "audio.mp3"):
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
        st.image("/home/wuye/vscode/chatbox/images/a9b65894-4916-4291-aec5-083e8db149d1.png", width=200)
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
        qa_system.update_knowledge()
        time.sleep(1)
        my_bar.progress(30)
        talk.text("😈好像找到答案了？！🤔")
        my_bar.progress(60)
        answer = qa_system.ask(query)
        talk.text("🎉 答案已找到！😻")
        my_bar.progress(90)
        st.markdown(f"### 答案\n{answer}")

        talk.text("🔊 生成语音中...")
        speek(answer)
        st.audio("audio.mp3")
        my_bar.progress(100)

        if os.path.exists("audio.mp3"):
            os.remove("audio.mp3")
        my_bar.empty()
        talk.empty()

if __name__ == "__main__":
    main()