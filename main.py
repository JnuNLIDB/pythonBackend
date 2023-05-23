import json.decoder
import logging
import os

from fastapi import FastAPI, Request, HTTPException, status
from langchain import OpenAI
from langchain.callbacks import get_openai_callback
from starlette.middleware.cors import CORSMiddleware

from asyncDBChain import AsyncSQLDatabaseChain
from asyncDatabase import AsyncSQLDatabase
from asyncTools import AsyncSQLDatabaseToolkit, create_sql_agent
from config import OPENAI_API_KEY, POSTGRES_URI

import openai
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


@app.post("/v1/nlidb")
async def nlidb(request: Request):
    global db
    if db is None:
        db = await AsyncSQLDatabase.from_uri(POSTGRES_URI)
    try:
        j = await request.json()
    except json.decoder.JSONDecodeError:
        j = None
    if j is None or 'question' not in j or 'llm' not in j:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid JSON")

    if j['llm'] != 'openai':
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid provider")
    llm = OpenAI(temperature=0 if 'temperature' not in j else int(j['temperature']))

    with get_openai_callback() as cb:
        result = "GPT: " + llm.generate([j['question']])[0] + "\n====================\nWith Postgres: "
        try:
            toolkit = AsyncSQLDatabaseToolkit(db=db, llm=llm)
            agent_executor = create_sql_agent(
                llm=llm,
                toolkit=toolkit,
                verbose=False
            )
            result += await agent_executor.arun(j['question'])
        except Exception as e1:
            try:
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
