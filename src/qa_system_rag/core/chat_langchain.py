from typing import Sequence

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_vertexai import ChatVertexAI
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from vertexai.language_models import TextEmbeddingModel

from src.qa_system_rag.app.config import Config
from typing_extensions import Annotated, TypedDict
from langgraph.graph.message import add_messages

from langchain.embeddings.base import Embeddings
from typing import List

llm = ChatVertexAI(model="gemini-1.5-flash")

class VertexAITextEmbeddings(Embeddings):
    def __init__(self, model_name: str = "text-embedding-004"):
        self.model = TextEmbeddingModel.from_pretrained(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Embedding for multiple documents
        embeddings = self.model.get_embeddings(texts)
        return [embedding.values for embedding in embeddings]

    def embed_query(self, text: str) -> List[float]:
        # Embedding for a single query
        embedding = self.model.get_embeddings([text])[0]
        return embedding.values

def build_rag_chain():
    contextualize_q_system_prompt = """
    Role Description:
        You are an expert support agent for Planton Cloud. You will be provided with various types of documents, such as 
        product documentation and guideline documents for different types of API resources.

    Primary Task:
        Given a chat history and the latest user question which might reference context in the chat history, 
        Your task is to assist users by answering their questions. These questions may be:

        General Inquiries: Questions regarding the general use, features, or capabilities of Planton Cloud.
        API Resource Requests: Requests to create, modify, or delete an API resource within Planton Cloud.
        For API resource requests, you are expected to follow the instructions outlined in the guideline documents. 
        This includes:
            Prompting the user for required input parameters.
            Guiding the user through the necessary steps as per the guidelines.

    Important Guidelines:
        Prioritize Accuracy:
            Provide precise and accurate information.
            If you are uncertain about any detail, state "Unknown" instead of guessing.        
        Avoid Adding Unverified Information:
            Do not include information that is not directly supported by the provided documents.
            Refrain from making assumptions or adding personal interpretations.
        Be Specific and Clear:
            Use precise language to address the user's query.
            Ensure that all information is relevant and directly answers their question.
        Consider Context:
            Take into account the context of the user's question, including any previous interactions or information provided.
            If the user provides specific details or references, incorporate that information into your response.
    
    Follow Guidelines for API Resource Management:
        Adhere strictly to the instructions in the guideline documents when assisting with API resource creation, modification, or deletion.
        Prompt the user for all necessary input parameters as specified.
        Provide step-by-step guidance to ensure the user can successfully complete their request.
    Objective:
        By following these guidelines, you will effectively assist users in resolving their inquiries and managing API resources within Planton Cloud, ensuring a high level of support and customer satisfaction.
        """

    loader = DirectoryLoader(Config.PDF_FOLDER_PATH, glob="**/*.pdf")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    vectorstore = InMemoryVectorStore.from_documents(
        documents=splits, embedding=VertexAITextEmbeddings()
    )
    retriever = vectorstore.as_retriever()

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    ### Answer question ###
    system_prompt = (
        """You are an expert support agent for Planton Cloud 
        Use the following pieces of retrieved context to answer the question. If you don't know the answer, say that 
        you don't know. 
        \n\n
        "{context}"""
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return rag_chain


### Statefully manage chat history ###
class State(TypedDict):
    input: str
    chat_history: Annotated[Sequence[BaseMessage], add_messages]
    context: str
    answer: str


rag_chain = build_rag_chain()


def call_model(state: State):
    response = rag_chain.invoke(state)
    return {
        "chat_history": [
            HumanMessage(state["input"]),
            AIMessage(response["answer"]),
        ],
        "context": response["context"],
        "answer": response["answer"],
    }
