import json.decoder
import os

from fastapi import FastAPI, Request, HTTPException, status
from langchain import OpenAI, SQLDatabase, SQLDatabaseChain

from asyncDBChain import AsyncSQLDatabaseChain
from asyncDatabase import AsyncSQLDatabase
from config import OPENAI_API_KEY

app = FastAPI()
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY


@app.get("/v1/nlidb")
async def nlidb(request: Request):
    try:
        j = await request.json()
    except json.decoder.JSONDecodeError:
        j = None
    if j is None or 'question' not in j or 'llm' not in j:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid JSON")

    if j['llm'] != 'openai':
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid provider")

    try:
        db = await AsyncSQLDatabase.from_uri('sqlite+aiosqlite:///./new.db')
        llm = OpenAI(temperature=0 if 'temperature' not in j else int(j['temperature']))
        chain = AsyncSQLDatabaseChain.from_llm(llm, db)
        result = await chain.arun(j['question'])
    except Exception as e:
        raise e

    return {'message': result}
