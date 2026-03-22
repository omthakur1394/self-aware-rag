import os 
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(
    model="gpt-5-nano-2025-08-07",temperature=0.2
)

