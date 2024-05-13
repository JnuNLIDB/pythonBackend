import json.decoder
import logging
import os

import openai
from fastapi import FastAPI, Request, HTTPException, status
from langchain_core.callbacks import BaseCallbackHandler, BaseCallbackManager
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAI, ChatOpenAI
from langchain_chroma import Chroma

from langchain.agents import create_vectorstore_agent, create_vectorstore_router_agent
from langchain.agents.agent_toolkits import VectorStoreInfo, VectorStoreToolkit, VectorStoreRouterToolkit
from langchain.callbacks import get_openai_callback
from langchain.schema import HumanMessage
from starlette.middleware.cors import CORSMiddleware

from asyncDBChain import AsyncSQLDatabaseChain
from asyncDatabase import AsyncSQLDatabase
from asyncTools import AsyncSQLDatabaseToolkit, create_sql_agent
from config import OPENAI_API_KEY, POSTGRES_URI

openai.proxy = {
    "http": "http://127.0.0.1:10809",
    "https": "http://127.0.0.1:10809"
}

app = FastAPI()
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

origins = [
    "http://localhost",
    "http://127.0.0.1:5173",
]
logger = logging.getLogger(__name__)
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

db = None

embeddings = OpenAIEmbeddings()

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


@app.post("/v1/embedding")
async def embedding(request: Request):
    j, llm = await get_params(request)
    router_toolkit = VectorStoreToolkit(
        vectorstore_info=vector_infos[0], llm=llm
    )

    agent_executor = create_vectorstore_agent(
        llm=llm, toolkit=router_toolkit, verbose=True,
        callback_manager=BaseCallbackManager(handlers=[], inheritable_handlers=[Foo()])
    )

    with get_openai_callback() as cb:
        try:
            result = agent_executor.invoke(
                j['question'],
                config={
                    "callbacks": BaseCallbackManager(handlers=[], inheritable_handlers=[Foo()])
                }
            )
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
        finally:
            total_token = cb.total_tokens
            prompt_token = cb.prompt_tokens
            completion_token = cb.completion_tokens
            total_cost = cb.total_cost

    return {
        'detail': result,
        'cost': {
            'total_token': total_token,
            'prompt_token': prompt_token,
            'completion_token': completion_token,
            'total_cost': total_cost
        }
    }


async def get_params(request):
    try:
        j = await request.json()
    except json.decoder.JSONDecodeError:
        j = None
    if j is None or 'question' not in j or 'llm' not in j:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid JSON")
    if j['llm'] != 'openai':
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid provider")
    llm = OpenAI(temperature=0 if 'temperature' not in j else int(j['temperature']))
    print("Received request: " + str(j))
    return j, llm


@app.post("/v1/nlidb")
async def nlidb(request: Request):
    global db
    if db is None:
        db = await AsyncSQLDatabase.from_uri(POSTGRES_URI)
    j, llm = await get_params(request)
    chat = ChatOpenAI(temperature=1)

    with get_openai_callback() as cb:
        print("Using GPT...")
        resp = await chat.agenerate([[HumanMessage(content=j['question'])]])
        resp = resp.generations[0][0].text
        result = "GPT: " + resp + "\n====================\nWith Postgres: "
        try:
            print("Trying to use LLM Toolkit...")
            toolkit = AsyncSQLDatabaseToolkit(db=db, llm=llm)
            agent_executor = create_sql_agent(
                llm=llm,
                toolkit=toolkit,
                verbose=False
            )
            k = await agent_executor.arun(j['question'])
            if "Not related to database" in k:
                print("Not related to database, using GPT...")
                k = await chat.agenerate([[HumanMessage(content=j['question'])]])
                k = k.generations[0][0].text
            result += k
        except Exception as e1:
            try:
                print("LLM Toolkit failed, using LLM Chain...")
                chain = AsyncSQLDatabaseChain.from_llm(llm, db)
                result += await chain.arun(j['question'])
            except Exception as e2:
                raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e1) + str(e2))
        finally:
            total_token = cb.total_tokens
            prompt_token = cb.prompt_tokens
            completion_token = cb.completion_tokens
            total_cost = cb.total_cost

    return {
        'detail': result,
        'cost': {
            'total_token': total_token,
            'prompt_token': prompt_token,
            'completion_token': completion_token,
            'total_cost': total_cost
        }
    }
