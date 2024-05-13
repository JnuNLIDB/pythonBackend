import os

import chromadb
import openai
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAI
from langchain_chroma import Chroma

from config import OPENAI_API_KEY

openai.proxy = {
    "http": "http://127.0.0.1:10809",
    "https": "http://127.0.0.1:10809"
}

os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

llm = OpenAI(temperature=0)
embedding = OpenAIEmbeddings(max_retries=999999999999999999999999999999)

if __name__ == '__main__':
    client = chromadb.Client(chromadb.config.Settings(
        persist_directory="./news2"
    ))
    print(list(client.list_collections()))

    queue = [n.replace("_preprocessed.txt", "") for n in os.listdir("./data") if n.endswith("_preprocessed.txt")]
    print(queue)
    for q in queue:
        print(f"Loading documents of {q}...")
        loader = TextLoader(f"./data/{q}_preprocessed.txt", encoding="utf-8")
        documents = loader.load()

        print("Splitting documents...")
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(documents)
        print(len(texts))

        print("Loading vector store...")
        embeddings = OpenAIEmbeddings()
        chroma = Chroma.from_documents(
            texts, collection_name=q, persist_directory="./news2", embedding=embeddings
        )

    client = chromadb.Client(chromadb.config.Settings(
        persist_directory="./news2"
    ))
    print(client.list_collections())

