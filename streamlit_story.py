import streamlit as st
from transformers import pipeline
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceEndpoint
from langchain.chains import LLMChain
from huggingface_hub import login
from PIL import Image
from dotenv import find_dotenv, load_dotenv
import os

load_dotenv(verbose=True, override=True)
load_dotenv(find_dotenv())


HUGGING_TOKEN = os.getenv('HUGGING_FACE_TOKEN')


login(token=HUGGING_TOKEN)

# Define the functions
def imag2text(image):
    image_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    text = image_to_text(image)[0]["generated_text"]
    return text

def generate_story(scenario):
    template = """
        You are a story teller;
        You can generate short story from a simple narration, the story should be no more than 100 words;

        CONTEXT: {scenario}
        STORY:
        """
    
    prompt = PromptTemplate(
        input_variables=["scenario"],
        template=template
    )

    repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"

    llm = HuggingFaceEndpoint(
        repo_id=repo_id, max_length=128, temperature=0.5, huggingfacehub_api_token=HUGGING_TOKEN
    )
    llm_chain = LLMChain(prompt=prompt, llm=llm)

    story = llm_chain.predict(scenario=scenario)
    return story

# Streamlit app start
st.title("Image to Story Generator")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    
    st.write("Generating story...")
    
    scenario = imag2text(image)
    story = generate_story(scenario)
    
    st.write("Scenario:", scenario)
    st.write("Story:", story)
