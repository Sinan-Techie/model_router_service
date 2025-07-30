import os
import json
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
load_dotenv()

def load_model_config(path='model_config.json'):
    with open(path, 'r') as f:
        return json.load(f)

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
        else:
            raise ValueError(f"Unsupported provider: {provider}")
        
    except Exception as e:
        print(f"[LLM ERROR] Failed to initialize LLM ({provider}-{model}): {e}")
        raise
