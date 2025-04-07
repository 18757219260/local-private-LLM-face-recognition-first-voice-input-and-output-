import json
import os
from datetime import datetime
from typing import List, Dict
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM

class KnowledgeQA:
    def __init__(self, knowledge_path="knowledge.json", faiss_index_path="faiss_index",
                 llm_model="mistral", history_log="chat_history.json"):
        self.knowledge_path = knowledge_path
        self.faiss_index_path = faiss_index_path
        self.llm_model = llm_model
        self.history_log = history_log
        self.conversation_history = self.load_history()
        self.embedding_model = self._init_embeddings()
        self.vectorstore = self._load_or_create_vectorstore()
        self.qa_chain = self._init_qa_chain()

    def _init_embeddings(self):
        return HuggingFaceEmbeddings(
            model_name="./bge-base-zh-v1.5",
            encode_kwargs={'normalize_embeddings': True}
        )

    def _load_knowledge(self):
        with open(self.knowledge_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def create_documents(self, knowledge_base: List[Dict]):
        docs = []
        for item in knowledge_base:
            question = item.get("question", "")
            for answer in item.get("answer", []):
                docs.append(Document(
                    page_content=answer,
                    metadata={
                        "question": question,
                        "source": self.knowledge_path,
                        "create_time": datetime.now().isoformat()
                    }
                ))
        return docs

    def split_documents(self, documents: List[Document]):
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
        return splitter.split_documents(documents)

    def _load_or_create_vectorstore(self):
        if os.path.exists(self.faiss_index_path):
            return FAISS.load_local(self.faiss_index_path, self.embedding_model, allow_dangerous_deserialization=True)
        else:
            knowledge = self._load_knowledge()
            docs = self.create_documents(knowledge)
            chunks = self.split_documents(docs)
            vectorstore = FAISS.from_documents(chunks, self.embedding_model)
            vectorstore.save_local(self.faiss_index_path)
            return vectorstore

    def _init_qa_chain(self):
        llm = OllamaLLM(base_url='http://localhost:11434', model=self.llm_model, temperature=0.3)
        return RetrievalQA.from_chain_type(
            llm=llm,
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=False
        )

    def load_history(self):
        if os.path.exists(self.history_log):
            try:
                with open(self.history_log, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return []
        return []

    def save_history(self):
        with open(self.history_log, 'w', encoding='utf-8') as f:
            json.dump(self.conversation_history[-100:], f, ensure_ascii=False, indent=2)

    def ask(self, question: str) -> str:
        history = "\n".join(
            f"[{h.get('timestamp', 'N/A')}] User: {h.get('user', '')}\nBot: {h.get('bot', '')}"
            for h in self.conversation_history[-5:]
        )
        full_query = f"Conversation History:\n{history}\n\n新问题: {question}"
        result = self.qa_chain.invoke({"query": full_query})
        answer = result["result"]
        self.conversation_history.append({
            "timestamp": datetime.now().isoformat(),
            "user": question,
            "bot": answer
        })
        self.save_history()
        return answer