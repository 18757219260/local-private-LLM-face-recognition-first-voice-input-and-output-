import sys
import os
sys.path.append(os.path.abspath("/home/wuye/vscode/chatbox"))
import logging
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM
import nest_asyncio
import time
import asyncio
import random

nest_asyncio.apply()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("chat.log"), logging.StreamHandler()]
)


class KnowledgeQA:
    def __init__(
        self,
        faiss_index_path= "faiss_index",
        llm_model = "qwen2.5:7b",
        temperature = 0.4,
        ollama_url = 'http://localhost:11434',
        k_documents = 2
    ):
        """初始化qa配置"""
        self.faiss_index_path = faiss_index_path
        self.llm_model = llm_model
        self.temperature = temperature
        self.ollama_url = ollama_url
        self.k_documents = k_documents

        self.embedding_model = self._init_embeddings()
        self.vectorstore = self._load_vectorstore_with_retry()
        self.llm = self._init_llm()
        self.qa_chain = self._init_qa_chain()
        self.unknown_responses  = [
    "我不知道",
    "这个问题我无法回答",
    "抱歉我不太会",
    "我还不了解这方面。",
    "对不起，我没有这方面的资料。",
    "我不知道这个答案，不过你可以去问吴家卓",
    "好像不太会？",
    "我里个豆阿，你问出这么难的问题我怎么会呢？"
]
    
    def _init_embeddings(self):
        """初始化向量模型"""
        try:
            return HuggingFaceEmbeddings(
                model_name="bge-base-zh-v1.5",
                encode_kwargs={'normalize_embeddings': True}
            )
        except Exception as e:
            logging.error(f"错误初始化向量化模型: {e}")
            raise
    
    def _load_vectorstore_with_retry(self, max_retries=3):
        """下载向量模型"""
        for attempt in range(max_retries):
            try:
                return FAISS.load_local(
                    self.faiss_index_path, 
                    self.embedding_model, 
                    allow_dangerous_deserialization=True
                )
            except Exception as e:
                if attempt < max_retries - 1:
                    logging.warning(f"重新加载向量模型{attempt+1}/{max_retries} : {e}")
                    time.sleep(1)
                else:
                    logging.error(f" 多次尝试失败加载{max_retries} : {e}")
                    raise
    
    def _init_llm(self):
        """初始化大模型"""
        try:
            return OllamaLLM(
                base_url=self.ollama_url,
                model=self.llm_model,
                temperature=self.temperature
            )
        except Exception as e:
            logging.error(f"初始化llm错误: {e}")
            raise
    
    def _init_qa_chain(self):
        """初始化问答链（基于向量检索 + Ollama 本地大模型）"""
        try:
            return RetrievalQA.from_chain_type(
                llm=self.llm,
                retriever=self.vectorstore.as_retriever(search_kwargs={"k": self.k_documents}),
                return_source_documents=False
            )
        except Exception as e:
            logging.error(f"QA模型链初始化失败: {e}")
            raise
    

    async def ask_stream(self, question):
        """流式回答"""
        if not question or not question.strip():
            yield "我没有听清楚您的问题，请重新提问。"
            return
        
        try:
            query="你是一个甘薯专家，请你以说话的标准回答，请你根据参考内容回答，回答输出为一段，回答内容简洁，如果参考内容中没有相关信息，请回答'{}'。".format(random.choice(self.unknown_responses))
            
            retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.k_documents})
            docs = await asyncio.to_thread(retriever.invoke, question)
            
            if not docs:
                yield "我没有找到相关的甘薯知识，请尝试其他问题。"
                return
                
            context = "\n\n".join([doc.page_content for doc in docs])
            final_prompt = f"已知内容:\n{context}\n\n问题: {question}\n\n{query}"
            
            start_time = time.time()
            async for chunk in self.llm.astream(final_prompt):
                yield chunk
            logging.info(f"流式回答花费了 {time.time() - start_time:.2f} seconds")
            
        except Exception as e:
            logging.error(f"Error in ask_stream: {e}")
            yield "抱歉，处理您的问题时出现了错误，请稍后再试。"

    def ask(self, question):
        """
        用户提问接口。向LLM提问并直接返回完整答案，不使用流式输出。
        """
        if not question or not question.strip():
            return "我没有听清楚您的问题，请重新提问。"
        
        try:
            # 设置提示词
            query="你是一个甘薯专家，请你以说话的标准回答，请你根据参考内容回答，回答输出为一段，回答内容简洁，如果参考内容中没有相关信息，请回答'{}'。".format(random.choice(self.unknown_responses))
            
            # 获取相关文档
            retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.k_documents})
            docs = retriever.invoke(question)
            
            if not docs:
                return "我没有找到相关的甘薯知识，请尝试其他问题。"
                
            # 构建上下文
            context = "\n\n".join([doc.page_content for doc in docs])
            final_prompt = f"已知内容:\n{context}\n\n问题: {question}\n\n{query}"
            
            # 计时
            start_time = time.time()
            
            # 调用模型获取完整回答
            result = self.llm.invoke(final_prompt)
            
            # 记录耗时
            logging.info(f"问答耗时: {time.time() - start_time:.2f}秒")
            
            return result
            
        except Exception as e:
            logging.error(f"问答出错: {e}")
            return "抱歉，处理您的问题时出现了错误，请稍后再试。"


async def main():
    qa = KnowledgeQA()
    question = "甘薯的未来"
    async for chunk in qa.ask_stream(question):
        print(chunk, end='', flush=True)  

if __name__ == "__main__":
    asyncio.run(main())
    



