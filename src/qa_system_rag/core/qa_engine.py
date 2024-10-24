from typing import Sequence

import vertexai
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_vertexai import ChatVertexAI
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from vertexai.generative_models import (
    GenerationConfig,
    GenerativeModel,
    HarmBlockThreshold,
    HarmCategory,
    Image,
)
from vertexai.language_models import TextEmbeddingModel
from vertexai.vision_models import MultiModalEmbeddingModel

from src.qa_system_rag.utils.multimodal_qa_with_rag_utils import get_document_metadata, set_global_variable, \
    get_similar_text_from_query, \
    get_similar_image_from_query, get_gemini_response
import pickle
from src.qa_system_rag.app.config import Config
from typing_extensions import Annotated, TypedDict
from langgraph.graph.message import add_messages

from langchain.embeddings.base import Embeddings
from typing import List

# Initialize Vertex AI
vertexai.init(location="us-central1")

# # Instantiate text model with appropriate name and version
# text_model = GenerativeModel("gemini-1.0-pro")  # works with text, code

# Multimodal models: Choose based on your performance/cost needs
multimodal_model_15 = GenerativeModel(
    "gemini-1.5-pro"
)  # works with text, code, images, video(with or without audio) and audio(mp3) with 1M input context - complex reasoning

# Multimodal models: Choose based on your performance/cost needs
multimodal_model_15_flash = GenerativeModel(
    "gemini-1.5-flash"
)  # works with text, code, images, video(with or without audio) and audio(mp3) with 1M input context - faster inference

multimodal_model_10 = GenerativeModel(
    "gemini-1.0-pro-vision-001"
)  # works with text, code, video(without audio) and images with 16k input context

# Load text embedding model from pre-trained source
text_embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-004")

# Load multimodal embedding model from pre-trained source
multimodal_embedding_model = MultiModalEmbeddingModel.from_pretrained(
    "multimodalembedding"
)  # works with image, image with caption(~32 words), video, video with caption(~32 words)

set_global_variable("text_embedding_model", text_embedding_model)
set_global_variable("multimodal_embedding_model", multimodal_embedding_model)

safety_settings = {
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
}

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


def load_metadata_from_pickle(filename):
    """Loads the text and image metadata DataFrames from a pickle file."""
    with open(filename, "rb") as f:
        data = pickle.load(f)
    text_metadata_df = data["text_metadata"]
    image_metadata_df = data["image_metadata"]

    # Now you can use text_metadata_df and image_metadata_df
    print("\nText Metadata DataFrame:")
    print(text_metadata_df.head())

    print("\nImage Metadata DataFrame:")
    print(image_metadata_df.head())

    return text_metadata_df, image_metadata_df


def save_metadata_to_pickle(text_df, image_df, filename):
    """Saves the text and image metadata DataFrames to a pickle file."""
    data = {
        "text_metadata": text_df,
        "image_metadata": image_df,
    }
    with open(filename, "wb") as f:
        pickle.dump(data, f)
    print(f"Metadata saved to {filename}")


def get_answer(query):
    text_metadata_df, image_metadata_df = load_metadata_from_pickle(Config.PICKLE_FILE)
    matching_results_chunks_data = get_similar_text_from_query(
        query,
        text_metadata_df,
        column_name="text_embedding_chunk",
        top_n=10,
        chunk_text=True,
    )

    # Get all relevant images based on user query
    # matching_results_image_fromdescription_data = get_similar_image_from_query(
    #     text_metadata_df,
    #     image_metadata_df,
    #     query=query,
    #     column_name="text_embedding_from_image_description",
    #     image_emb=False,
    #     top_n=10,
    #     embedding_size=1408,
    # )

    instruction = """Task: Answer the following question in detail, providing clear reasoning .
                Instructions:

                1. **Analyze:** Carefully examine the provided text context.
                2. **Synthesize:** Integrate information from textual elements.
                3. **Reason:**  Deduce logical connections and inferences to address the question.
                4. **Respond:** Provide a concise, accurate answer in the following format:

                   * **Question:** [Question]
                   * **Answer:** [Direct response to the question]
                   * **Explanation:** [Bullet-point reasoning steps if applicable]
                   * **Source** [name of the file, page, image from where the information is citied]

                5. **Ambiguity:** If the context is insufficient to answer, respond "Not enough context to answer."

                """

    # combine all the selected relevant text chunks
    context_text = ["Text Context: "]
    for key, value in matching_results_chunks_data.items():
        context_text.extend(
            [
                "Text Source: ",
                f"""file_name: "{value["file_name"]}" Page: "{value["page_num"]}""",
                "Text",
                value["chunk_text"],
            ]
        )

    # combine all the selected relevant images
    gemini_content = [
        instruction,
        "Questions: ",
        query,
    ]
    # for key, value in matching_results_image_fromdescription_data.items():
    #     gemini_content.extend(
    #         [
    #             "Image Path: ",
    #             value["img_path"],
    #             "Image Description: ",
    #             value["image_description"],
    #             "Image:",
    #             value["image_object"],
    #         ]
    #     )
    gemini_content.extend(context_text)

    response = get_gemini_response(
        multimodal_model_15,
        model_input=gemini_content,
        stream=True,
        safety_settings=safety_settings,
        generation_config=GenerationConfig(temperature=1, max_output_tokens=8192),
    )

    return response


def load_text_embeddings():
    # Specify the image description prompt. Change it
    # image_description_prompt = """Explain what is going on in the image.
    # If it's a table, extract all elements of the table.
    # If it's a graph, explain the findings in the graph.
    # Do not include any numbers that are not mentioned in the image.
    # """

    image_description_prompt = """You are expert support agent of planton cloud. You will be provided with various types of documents like product documentation.
        Your task is to answer the questions that project planton user ask

        Important Guidelines:
        * Prioritize accuracy:  If you are uncertain about any detail, state "Unknown" or "Not visible" instead of guessing.
        * Avoid hallucinations: Do not add information that is not directly supported by the image.
        * Be specific: Use precise language to describe shapes, colors, textures, and any interactions depicted.
        * Consider context: If the image is a screenshot or contains text, incorporate that information into your description.
        """

    # Extract text and image metadata from the PDF document
    text_metadata_df, image_metadata_df = get_document_metadata(
        multimodal_model_15,  # we are passing Gemini 1.5 Pro
        Config.PDF_FOLDER_PATH,
        image_save_dir="images",
        image_description_prompt=image_description_prompt,
        embedding_size=1408,
        # add_sleep_after_page = True, # Uncomment this if you are running into API quota issues
        # sleep_time_after_page = 5,
        add_sleep_after_document=True,  # Uncomment this if you are running into API quota issues
        sleep_time_after_document=5,
        # Increase the value in seconds, if you are still getting quota issues. It will slow down the processing.
        # generation_config = # see next cell
        # safety_settings =  # see next cell
    )

    print("\n\n --- Completed processing. ---")

    # Save the DataFrames using pickle
    save_metadata_to_pickle(text_metadata_df, image_metadata_df, Config.PICKLE_FILE)


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
