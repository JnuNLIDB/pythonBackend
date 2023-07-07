import os

import openai
from langchain import OpenAI, ElasticVectorSearch
from langchain.agents import create_vectorstore_agent
from langchain.agents.agent_toolkits import VectorStoreInfo, VectorStoreToolkit
from langchain.embeddings.openai import OpenAIEmbeddings

from config import OPENAI_API_KEY

openai.proxy = {
    "http": "http://127.0.0.1:10809",
    "https": "http://127.0.0.1:10809"
}

os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

llm = OpenAI(temperature=0)

embedding = OpenAIEmbeddings()
elastic_vector_search = ElasticVectorSearch(
    elasticsearch_url="http://localhost:9200",
    index_name="opinion",
    embedding=embedding
)

vector_store_info = VectorStoreInfo(
    name="opinion",
    description="the most recent opinions from people all around the world",
    vectorstore=elastic_vector_search,
)
toolkit = VectorStoreToolkit(vectorstore_info=vector_store_info)
agent_executor = create_vectorstore_agent(llm=llm, toolkit=toolkit, verbose=True)

if __name__ == '__main__':
    agent_executor.run("Prompt")
