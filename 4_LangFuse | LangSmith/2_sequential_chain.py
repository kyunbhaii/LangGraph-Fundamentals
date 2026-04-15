from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langfuse.langchain import CallbackHandler
import os

os.environ['LANGCHAIN_PROJECT'] = 'Sequential APP'

load_dotenv()

langfuse_handler = CallbackHandler()

model = ChatGroq(
    model= 'llama-3.1-8b-instant',
    temperature= 0.7
)

prompt1 = PromptTemplate(
    template='Generate a detailed report on topic - {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a 5 pointer summary on the following text - {text}',
    input_variables=['text']
)

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser

config = {
    'run_name': 'sequential_chain',
    'tags': ['llm_app', 'report_generation', 'sequential_chain'],
    'metadata': {'model': 'llama-3.1-8b-instant', 'model_temp': 0.7, 'parser': 'StrOutputParser'},
    'callbacks': [langfuse_handler]
}

for message in chain.stream({'topic': 'Attention is all you need'}, config = config):
    print(message, end="", flush=True)