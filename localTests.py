import json
import os

import openai

from openai_embedding import OpenAIEmbeddings
from langchain import OpenAI
from langchain.agents import create_vectorstore_agent
from langchain.agents.agent_toolkits import VectorStoreInfo, VectorStoreToolkit
from langchain.vectorstores import Chroma

from config import OPENAI_API_KEY

openai.proxy = {
    "http": "http://127.0.0.1:10809",
    "https": "http://127.0.0.1:10809"
}

os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

llm = OpenAI(temperature=0)
embedding = OpenAIEmbeddings(max_retries=999999999999999999999999999999)

# print("Loading documents...")
# loader = TextLoader("./preprocessed.txt")
# documents = loader.load()
#
# print("Splitting documents...")
# text_splitter = CharacterTextSplitter()
# texts = text_splitter.split_documents(documents)

print("Loading vector store...")
embeddings = OpenAIEmbeddings()
chroma = Chroma(
    collection_name="news", persist_directory="./news", embedding_function=embeddings
)

# print("Persisting vector store...")
# chroma.persist()

print("Creating vector store info...")
vector_store_info = VectorStoreInfo(
    name="news",
    description="the most recent news from people all around the world",
    vectorstore=chroma,
)

print("Creating agent executor...")
toolkit = VectorStoreToolkit(vectorstore_info=vector_store_info)
agent_executor = create_vectorstore_agent(llm=llm, toolkit=toolkit, verbose=True)

if __name__ == '__main__':
    print("Running agent executor...")
    result = []
    for q in [
        "What views on the South China Sea have been expressed by U.S. President Joe Biden?",
        "What are the main South China Sea issues in focus?",
        "What has U.S. Secretary of State Blinken said about the South China Sea? ",
        "What did Fumio Kishida say about the South China Sea issue？",
        "What has Biden said about China in US media reports？",
        "What has the Philippine media done about the Chinese navy?",
        "What has the Philippine media reported on the Taiwan issue?",
        "What Philippine media say about China's military drills",
        "What does Medel Aguilar think of Balikatan?",
        "what is Priority defense platform?",
        "What are Marcos' views on the South China Sea reported by Philippine media",
        "What is Robin Padilla’s opinion on drugs?",
        "What is Martin Romualdez's views on China and the United States respectively? What are the differences?",
        "Cooperation between China and the National Grid Corporation of the Philippines (NGCP)"
    ]:
        resp = agent_executor.run(q)
        result.append(resp)

    print(json.dumps(result, indent=4, ensure_ascii=False))
