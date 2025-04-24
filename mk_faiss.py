import json
import os
import logging
from datetime import datetime
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("mk_faiss.log"), logging.StreamHandler()]
)

class MkFaiss:
    def __init__(
        self,
        knowledge_path: str = "knowledge.json",
        faiss_index_path: str = "faiss_index",
        embedding_model_name: str = "bge-base-zh-v1.5"
    ):
 
        self.knowledge_path = knowledge_path
        self.faiss_index_path = faiss_index_path
        self.embedding_model_name = embedding_model_name
        try:
            self.embedding_model = self._init_embeddings()
        except Exception as e:
            logging.error(f"初始化嵌入模型失败: {e}")
            raise
        self.vectorstore = self.load_or_create_vectorstore()

    def _init_embeddings(self) :

        try:
            return HuggingFaceEmbeddings(
                model_name=self.embedding_model_name,
                encode_kwargs={'normalize_embeddings': True}
            )
        except Exception as e:
            logging.error(f"加载嵌入模型 {self.embedding_model_name} 失败: {e}")
            raise RuntimeError(f"嵌入模型加载失败: {e}")

    def load_data(self) :
        
        try:
            if not os.path.exists(self.knowledge_path):
                logging.error(f"知识库文件 {self.knowledge_path} 不存在")
                raise FileNotFoundError(f"知识库文件 {self.knowledge_path} 不存在")
            with open(self.knowledge_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if not isinstance(data, list):
                logging.error("知识库必须是 JSON 数组")
                raise ValueError("知识库必须是 JSON 数组")
            if not data:
                logging.warning("知识库为空")
            logging.info(f"成功加载知识库: {len(data)} 条记录")
            return data
        except json.JSONDecodeError as e:
            logging.error(f"解析 {self.knowledge_path} 失败: {e}")
            raise
        except Exception as e:
            logging.error(f"加载知识库失败: {e}")
            raise

    def create_documents(self, knowledge_base) :
        docs = []
        for item in knowledge_base:
            question = item.get("question", "")
            answers = item.get("answer", [])
            if not question or not answers:
                logging.warning(f"无效的问答对: question={question}, answers={answers}")
                continue
            for answer in answers:
                if not isinstance(answer, str):
                    logging.warning(f"答案必须是字符串: {answer}")
                    continue
                docs.append(Document(
                    page_content=f"Q: {question}\nA: {answer}",
                    metadata={
                        "question": question,
                        "source": self.knowledge_path,
                        "create_time": datetime.now().isoformat()
                    }
                ))
        logging.info(f"生成 {len(docs)} 个文档")
        return docs

    def split_documents(self, documents):
      
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,  
            chunk_overlap=100,
            add_start_index=True
        )
        chunks = splitter.split_documents(documents)
        logging.info(f"生成 {len(chunks)} 个文档分块")
        return chunks

    def load_or_create_vectorstore(self) :
        try:
            if os.path.exists(self.faiss_index_path):
                logging.info(f"加载现有向量数据库: {self.faiss_index_path}")
                return FAISS.load_local(
                    self.faiss_index_path,
                    self.embedding_model,
                    allow_dangerous_deserialization=True
                )
            logging.info("创建新的向量数据库")
            data = self.load_data()
            docs = self.create_documents(data)
            if not docs:
                logging.error("无法创建向量数据库：文档为空")
                raise ValueError("文档为空")
            chunks = self.split_documents(docs)
            vectorstore = FAISS.from_documents(chunks, self.embedding_model)
            vectorstore.save_local(self.faiss_index_path)
            logging.info(f"向量数据库已保存到 {self.faiss_index_path}")
            return vectorstore
        except Exception as e:
            logging.error(f"加载或创建向量数据库失败: {e}")
            raise RuntimeError(f"向量数据库处理失败: {e}")

    def update_knowledge(self) :
      
        try:
            logging.info("开始更新知识库")
            data = self.load_data()
            docs = self.create_documents(data)
            if not docs:
                logging.error("无法更新向量数据库：文档为空")
                raise ValueError("文档为空")
            chunks = self.split_documents(docs)
            self.vectorstore = FAISS.from_documents(chunks, self.embedding_model)
            self.vectorstore.save_local(self.faiss_index_path)
            logging.info("知识库更新成功！")
        except Exception as e:
            logging.error(f"更新知识库失败: {e}")
            raise

    def get_vectorstore(self):
        return self.vectorstore

if __name__ == "__main__":

    mk_faiss = MkFaiss()
    mk_faiss.update_knowledge()
    vectorstore = mk_faiss.get_vectorstore()
    logging.info("向量数据库测试成功")
