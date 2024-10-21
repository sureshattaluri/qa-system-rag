import argparse

import vertexai
from rich import print as rich_print
from rich.markdown import Markdown as rich_Markdown
from vertexai.generative_models import (
    GenerationConfig,
    GenerativeModel,
    HarmBlockThreshold,
    HarmCategory,
    Image,
)
from vertexai.language_models import TextEmbeddingModel
from vertexai.vision_models import MultiModalEmbeddingModel
from utils.multimodal_qa_with_rag_utils import get_document_metadata, set_global_variable, get_similar_text_from_query, \
    get_similar_image_from_query, get_gemini_response
import pickle
import os

def get_answer(query, pickle_file):
    vertexai.init(location="us-central1")
    # Instantiate text model with appropriate name and version
    text_model = GenerativeModel("gemini-1.0-pro")  # works with text, code

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

    if not args.regenerate or os.path.exists(pickle_file):
        text_metadata_df, image_metadata_df = load_metadata_from_pickle(pickle_file)
    else:
        pdf_folder_path = "/Users/suresh/Downloads/planton-cloud-pdf"

        # Specify the image description prompt. Change it
        # image_description_prompt = """Explain what is going on in the image.
        # If it's a table, extract all elements of the table.
        # If it's a graph, explain the findings in the graph.
        # Do not include any numbers that are not mentioned in the image.
        # """

        image_description_prompt = """You are expert support agent of project planton. You will be provided with various types of documents like product documentation.
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
            pdf_folder_path,
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
        save_metadata_to_pickle(text_metadata_df, image_metadata_df, pickle_file)

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

    rich_Markdown(
        response
    )

    print(response)

def save_metadata_to_pickle(text_df, image_df, filename):
    """Saves the text and image metadata DataFrames to a pickle file."""
    data = {
        "text_metadata": text_df,
        "image_metadata": image_df,
    }
    with open(filename, "wb") as f:
        pickle.dump(data, f)
    print(f"Metadata saved to {filename}")

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




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and load metadata.")
    parser.add_argument("--regenerate", action="store_true", help="Force data regeneration.")
    # parser.add_argument("--query", type=str, required=True, help="question")
    args = parser.parse_args()

    pickle_file = "mrag_metadata.pkl"

    get_answer('what is project planton', pickle_file)


