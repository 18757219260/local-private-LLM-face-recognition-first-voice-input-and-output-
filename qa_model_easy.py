import json
import os
import logging
from datetime import datetime
from typing import List, Dict, Any
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM
import nest_asyncio
import time

nest_asyncio.apply()

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
    ):
        """
        初始化知识问答系统，加载对话历史、嵌入模型、向量库和问答链。
        """
        # self.knowledge_path = knowledge_path
        self.faiss_index_path = faiss_index_path
        self.llm_model = llm_model
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="./bge-base-zh-v1.5",
            encode_kwargs={'normalize_embeddings': True}
        )
        # self.embedding_model = self._init_embeddings()
        self.vectorstore = self.load_or_create_vectorstore()
        self.qa_chain = self.init_qa_chain()


    
    def load_or_create_vectorstore(self) :
        """
        加载已有的向量库；如果不存在则从知识库创建向量库，并保存。
        """
       
        return FAISS.load_local(self.faiss_index_path, self.embedding_model, allow_dangerous_deserialization=True)


    def init_qa_chain(self) :
        """
        初始化问答链（基于向量检索 + Ollama 本地大模型）。
        """
        llm = OllamaLLM(base_url='http://localhost:11434', model=self.llm_model, temperature=0.4)
        
        return RetrievalQA.from_chain_type(
            llm=llm,
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 2}),
            return_source_documents=False
        )
        


    def ask(self, question) -> str:
        start=time.time()   
        """
        用户提问接口。整合历史上下文，向 LLM 提问并记录回答。
        """
        
        prompt = """
        你是个甘薯个专家请以纯文本形式回答,输出为一段。如果输入问题和甘薯一点关系都没有，请直接回答“我不知道”。输出的内容要简洁明了，避免使用复杂的术语和长句子。请确保回答是准确的，并且与问题相关。请不要添加任何额外的解释或背景信息。
        """
        query = f"问题: {question}\n\n{prompt}"
        result = self.qa_chain.invoke({"query": query})  
        answer = result["result"]
        end=time.time()
        logging.info(f"模型问答耗时: {end-start:.2f}秒")
        
        return answer
