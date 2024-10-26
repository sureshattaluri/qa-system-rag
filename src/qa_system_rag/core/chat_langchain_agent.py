from typing import List

from langchain_community.document_loaders import DirectoryLoader
from langchain_core.tools import create_retriever_tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_vertexai import ChatVertexAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from src.qa_system_rag.app.config import Config
from langchain.embeddings.base import Embeddings
from vertexai.language_models import TextEmbeddingModel

agentExecutorMemory = MemorySaver()
llmForAgent = ChatVertexAI(model="gemini-1.5-pro")


class VertexAITextEmbeddings(Embeddings):
    def __init__(self, model_name: str = "textembedding-gecko@001"):
        self.model = TextEmbeddingModel.from_pretrained(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.get_embeddings(texts)
        return [embedding.values for embedding in embeddings]

    def embed_query(self, text: str) -> List[float]:
        embedding = self.model.get_embeddings([text])[0]
        return embedding.values

def agent_executor():
    loader = DirectoryLoader(Config.PDF_FOLDER_PATH, glob="**/*.pdf")
    docs = loader.load()
    if not docs:
        raise ValueError("No documents loaded. Please check the PDF folder path.")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # Initialize the embedding model
    embeddings = VertexAITextEmbeddings(model_name="text-embedding-004")

    # Create the vector store
    vectorstore = InMemoryVectorStore.from_documents(docs, embedding=embeddings)
    retriever = vectorstore.as_retriever()

    ### Build retriever tool ###
    tool = create_retriever_tool(
        retriever,
        "planton cloud support agent",
        "Provides excerpts from the Planton Cloud documentation.",
    )
    tools = [tool]
    return create_react_agent(llmForAgent, tools, checkpointer=agentExecutorMemory)
