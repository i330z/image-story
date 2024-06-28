from dotenv import find_dotenv, load_dotenv
from transformers import pipeline
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceEndpoint
from langchain.chains import LLMChain
# from huggingface_hub import login


# login(token=HUGGING_TOKEN)


load_dotenv(find_dotenv())

def imag2text(url):
    image_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

    text = image_to_text(url)[0]["generated_text"]

    print(text)
    return text




def generate_story(scenario):
    template="""
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

    print(story)
    return story



scenario = imag2text("photo.jpeg")
story = generate_story(scenario)