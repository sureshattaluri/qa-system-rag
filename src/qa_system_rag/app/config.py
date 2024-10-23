from vertexai.generative_models import (
    HarmBlockThreshold,
    HarmCategory,
)

class Config:
    PDF_FOLDER_PATH = "/Users/suresh/Downloads/planton-cloud-pdf"
    PICKLE_FILE = "./mrag_metadata.pkl"
    SAFETY_SETTINGS = {
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    }
