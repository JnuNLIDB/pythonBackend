import json
import os

import openai

from openai_embedding import OpenAIEmbeddings
from langchain import OpenAI
from langchain.agents import create_vectorstore_agent, create_vectorstore_router_agent
from langchain.agents.agent_toolkits import VectorStoreInfo, VectorStoreToolkit, VectorStoreRouterToolkit
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

# print("Persisting vector store...")
# chroma.persist()

print("Creating vector store info...")
names = [
    'philippines', 'south_china_sea', 'tibet', 'xin_jiang', 'hong_kong', 'taiwan'
]

vector_infos = []
for name in names:
    vector = Chroma(
        collection_name=name, persist_directory="./news", embedding_function=embeddings
    )
    vector_info = VectorStoreInfo(
        name=name,
        description="the most recent news of " + name,
        vectorstore=vector,
    )
    vector_infos.append(vector_info)


print("Creating agent executor...")
router_toolkit = VectorStoreRouterToolkit(
    vectorstores=vector_infos, llm=llm
)
agent_executor = create_vectorstore_router_agent(
    llm=llm, toolkit=router_toolkit, verbose=True
)

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
