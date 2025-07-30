import os
import json
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
import re
from dotenv import load_dotenv
load_dotenv()

def load_model_config(path='model_config.json'):
    with open(path, 'r') as f:
        return json.load(f)

def extract_json(text):
    """Extract a JSON object from text, handling extra content around it."""
    if '{' not in text or '}' not in text:
        print("No JSON found in the text.")
        return text
    try:
        cleaned_text = re.sub(r"^```(?:json)?\n?", "", text.strip())
        cleaned_text = re.sub(r"\n?```$", "", cleaned_text.strip())
        print("cleaned_text",cleaned_text)
        start = text.index('{')
        end = text.rindex('}') + 1
        json_str = text[start:end]
        return json.loads(json_str)
    except (ValueError, json.JSONDecodeError) as e:
        return None
    
def get_llm_instance(config):

    provider = config['provider']
    model = config['model']
    temperature = config.get('temperature', 0.1)
    try:

        if provider == "openai":
            return ChatOpenAI(model=model, openai_api_key=os.getenv("OPENAI_API_KEY"), temperature=temperature)
        elif provider == "groq":
            return ChatGroq(model=model, groq_api_key=os.getenv("GROQ_API_KEY"), temperature=temperature)
        elif provider == "gemini":
            return ChatGoogleGenerativeAI(model=model, google_api_key=os.getenv("GEMINI_API_KEY"), temperature=temperature)
        elif provider == "anthropic":
            return ChatAnthropic(model=model, anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"), temperature=temperature)
        elif provider == "huggingface":
            llm= HuggingFaceEndpoint(
                repo_id=model,
                huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_KEY"),
                temperature=temperature
            )
            return ChatHuggingFace(llm=llm)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
        
    except Exception as e:
        print(f"[LLM ERROR] Failed to initialize LLM ({provider}-{model}): {e}")
        raise
