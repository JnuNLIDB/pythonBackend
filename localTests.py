import json
import os

import openai

from langchain.agents import create_vectorstore_agent, create_vectorstore_router_agent
from langchain.agents.agent_toolkits import VectorStoreInfo, VectorStoreToolkit, VectorStoreRouterToolkit
from langchain_core.callbacks import BaseCallbackManager, BaseCallbackHandler
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAI, ChatOpenAI
from langchain_chroma import Chroma

from config import OPENAI_API_KEY

openai.proxy = {
    "http": "http://127.0.0.1:10809",
    "https": "http://127.0.0.1:10809"
}

os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

llm = OpenAI(temperature=0, model_name="gpt-3.5-turbo-instruct")

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
    'theory'
]

vector_infos = []
for name in names:
    vector = Chroma(
        collection_name=name, persist_directory="./news2", embedding_function=embeddings
    )
    vector_info = VectorStoreInfo(
        name=name,
        description="the most recent news of " + name,
        vectorstore=vector,
    )
    vector_infos.append(vector_info)

print("Creating agent executor...")
router_toolkit = VectorStoreToolkit(
    vectorstore_info=vector_infos[0], llm=llm
)


class Foo(BaseCallbackHandler):
    def on_chain_start(
            self,
            serialized,
            inputs,
            *,
            run_id=None,
            parent_run_id=None,
            tags=None,
            metadata=None,
            **kwargs,
    ):
        for (i, count) in [('summaries', 2000), ('context', 3000)]:
            if i in inputs:
                inputs[i] = inputs[i][:count]
        if 'question' in inputs:
            inputs['question'] += " Answer and use tools in chinese!"
        return BaseCallbackManager(handlers=[Foo()])


agent_executor = create_vectorstore_agent(
    llm=llm, toolkit=router_toolkit, verbose=True, callback_manager=BaseCallbackManager(handlers=[], inheritable_handlers=[Foo()])
)

if __name__ == '__main__':
    print("Running agent executor...")
    result = []
    for q in [
        "新时代新在哪里？",
    ]:
        resp = agent_executor.invoke(q, config={"callbacks": BaseCallbackManager(handlers=[], inheritable_handlers=[Foo()])})
        result.append(resp)

    print(json.dumps(result, indent=4, ensure_ascii=False))
