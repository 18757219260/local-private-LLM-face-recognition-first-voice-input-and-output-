import sys
import os
import json
import logging
from typing import List, Dict, Any
from datetime import datetime
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM
import nest_asyncio
from mk_faiss import MkFaiss

# 应用nest_asyncio以确保在notebook环境中asyncio兼容性
nest_asyncio.apply()

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("chatqa.log"), logging.StreamHandler()]
)

class KnowledgeQA:
    """基于本地LLM的知识问答类"""
    
    def __init__(
        self,
        knowledge_path = "knowledge.json",
        faiss_index_path= "faiss_index",
        embedding_model_path= "./bge-base-zh-v1.5",
        llm_model= "qwen2.5:7b",
        ollama_base_url= "http://localhost:11434",
        history_log_path = "chat_history.json",
        max_history_items= 100,
        max_history_context = 5,
        chunk_size = 1000,  # 增加块大小为1000
        chunk_overlap = 200,  # 增加重叠为200
        temperature = 0.1,
        top_k = 3
    ):
        """
        初始化知识问答系统
        
        参数:
            knowledge_path: 知识库JSON文件路径
            faiss_index_path: FAISS索引存储/加载路径
            embedding_model_path: 嵌入模型路径或名称
            llm_model: 使用Ollama的LLM模型名称
            ollama_base_url: Ollama API的基础URL
            history_log_path: 对话历史存储路径
            max_history_items: 存储的最大对话条目数
            max_history_context: 上下文中包含的最近对话条目数
            chunk_size: 文档分块大小
            chunk_overlap: 连续分块之间的重叠
            temperature: LLM的温度（越高=越有创意）
            top_k: 检索的相似文档数量
        """
        self.history_log_path = history_log_path
        self.max_history_items = max_history_items
        self.max_history_context = max_history_context
        self.llm_model = llm_model
        self.ollama_base_url = ollama_base_url
        self.temperature = temperature
        self.top_k = top_k
        
        # 加载对话历史
        self.conversation_history = self._load_history()
        
        # 初始化向量存储管理器
        self.vector_manager = MkFaiss(
            knowledge_path=knowledge_path,
            faiss_index_path=faiss_index_path,
            embedding_model_path=embedding_model_path,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # 初始化问答链
        self.qa_chain = self._init_qa_chain()
    
    def _init_qa_chain(self):
        """使用Ollama LLM和向量检索初始化问答链"""
        try:
            llm = OllamaLLM(
                base_url=self.ollama_base_url,
                model=self.llm_model,
                temperature=self.temperature
            )
            
            return RetrievalQA.from_chain_type(
                llm=llm,
                retriever=self.vector_manager.get_vectorstore().as_retriever(
                    search_kwargs={"k": self.top_k}
                ),
                return_source_documents=True  # 确保返回源文档
            )
            
        except Exception as e:
            logging.error(f"初始化问答链失败: {e}")
            raise RuntimeError(f"问答链初始化失败: {e}")
    
    def _load_history(self) -> List[Dict[str, Any]]:
        """从文件加载对话历史"""
        if os.path.exists(self.history_log_path):
            try:
                with open(self.history_log_path, 'r', encoding='utf-8') as f:
                    history = json.load(f)
                logging.info(f"已加载 {len(history)} 条对话历史记录")
                return history
            except json.JSONDecodeError:
                logging.warning(f"无法解析 {self.history_log_path}，将使用空历史记录")
                return []
        return []
    
    def _save_history(self):
        """保存对话历史到文件"""
        try:
            # 只保留最近的记录，最多保存max_history_items条
            history_to_save = self.conversation_history[-self.max_history_items:]
            
            with open(self.history_log_path, 'w', encoding='utf-8') as f:
                json.dump(history_to_save, f, ensure_ascii=False, indent=2)
                
            logging.info(f"已保存 {len(history_to_save)} 条对话历史记录")
            
        except Exception as e:
            logging.error(f"保存对话历史失败: {e}")
    
    def _format_history_context(self):
        """将最近的对话历史格式化为LLM的上下文"""
        if not self.conversation_history:
            return ""
            
        # 获取最近的对话项
        recent_history = self.conversation_history[-self.max_history_context:]
        
        # 格式化历史记录
        formatted_history = []
        for item in recent_history:
            timestamp = item.get("timestamp", "N/A")
            user_query = item.get("user", "")
            bot_response = item.get("bot", "")
            formatted_history.append(f"[{timestamp}] User: {user_query}\nBot: {bot_response}")
            
        return "\n\n".join(formatted_history)
    
    def ask(self, question: str) -> Dict[str, Any]:
        """
        处理用户问题并生成答案
        
        参数:
            question: 用户的问题
            
        返回:
            包含答案和元数据的字典
        """
        # 检查知识库是否需要更新
        self.vector_manager.check_and_update_if_needed()
        
        # 准备系统提示
        system_prompt = """
        请以纯文本形式回答。你是个专业的甘薯知识问答助手，严格根据知识库内容回答问题。
        如果知识库中没有相关信息，请明确告知用户"知识库中没有关于甘薯的这方面信息"，不要编造答案。
        回答要简洁明了，直接针对问题给出答案。
        """
        
        # 获取对话历史上下文
        history_context = self._format_history_context()
        
        # 创建完整查询，包含历史上下文和系统提示
        if history_context:
            full_query = f"{system_prompt}\n\n---\n\n历史对话:\n{history_context}\n\n---\n\n新问题: {question}"
        else:
            full_query = f"{system_prompt}\n\n---\n\n问题: {question}"
        
        # 调用问答链
        try:
            result = self.qa_chain.invoke({"query": full_query})
            
            answer = result["result"]
            source_documents = result.get("source_documents", [])
            
            # 格式化来源信息
            sources = []
            for doc in source_documents:
                if doc.metadata and "question" in doc.metadata:
                    sources.append({
                        "question": doc.metadata["question"],
                        "content": doc.page_content
                    })
            
            # 创建响应
            response = {
                "answer": answer,
                "sources": sources
            }
            
            # 记录对话
            record = {
                "timestamp": datetime.now().isoformat(),
                "user": question,
                "bot": answer,
                "sources": [s["question"] for s in sources]
            }
            
            self.conversation_history.append(record)
            self._save_history()
            
            return response
            
        except Exception as e:
            error_msg = f"处理问题时出错: {str(e)}"
            logging.error(error_msg)
            return {"answer": error_msg, "sources": []}


# CLI示例
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="知识问答系统")
    parser.add_argument("--knowledge", default="knowledge.json", help="知识库JSON文件路径")
    parser.add_argument("--index", default="faiss_index", help="FAISS索引目录路径")
    parser.add_argument("--model", default="qwen2.5:7b", help="Ollama模型名称")
    parser.add_argument("--question", help="要提问的问题（如不提供则进入交互模式）")
    args = parser.parse_args()
    
    try:
        qa_system = KnowledgeQA(
            knowledge_path=args.knowledge,
            faiss_index_path=args.index,
            llm_model=args.model
        )
        
        if args.question:
            # 单问题模式
            response = qa_system.ask(args.question)
            print(f"\n回答: {response['answer']}\n")
            if response['sources']:
                print("参考来源:")
                for src in response['sources']:
                    print(f"- {src['question']}")
        else:
            # 交互模式
            print("知识问答系统已启动。输入'退出'或'exit'结束对话。")
            while True:
                user_input = input("\n请输入您的问题: ")
                if user_input.lower() in ['退出', 'exit', 'quit', 'q']:
                    break
                    
                response = qa_system.ask(user_input)
                print(f"\n回答: {response['answer']}\n")
                if response['sources']:
                    print("参考来源:")
                    for src in response['sources']:
                        print(f"- {src['question']}")
                        
    except Exception as e:
        logging.error(f"程序执行出错: {e}")
        print(f"程序执行出错: {e}")