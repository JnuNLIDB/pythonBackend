"""Chain for interacting with SQL Database."""
from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional

from langchain.base_language import BaseLanguageModel
from langchain.callbacks.manager import CallbackManagerForChainRun, AsyncRunManager
from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.chains.sql_database.prompt import DECIDER_PROMPT, PROMPT, SQL_PROMPTS
from langchain.prompts.base import BasePromptTemplate
from pydantic import Extra, Field, root_validator

from asyncDatabase import AsyncSQLDatabase


class AsyncSQLDatabaseChain(Chain):
    """Chain for interacting with SQL Database.

    Example:
        .. code-block:: python

            from langchain import AsyncSQLDatabaseChain, OpenAI, AsyncSQLDatabase
            db = AsyncSQLDatabase(...)
            db_chain = AsyncSQLDatabaseChain.from_llm(OpenAI(), db)
    """

    llm_chain: LLMChain
    llm: Optional[BaseLanguageModel] = None
    """[Deprecated] LLM wrapper to use."""
    database: AsyncSQLDatabase = Field(exclude=True)
    """SQL Database to connect to."""
    prompt: Optional[BasePromptTemplate] = None
    """[Deprecated] Prompt to use to translate natural language to SQL."""
    top_k: int = 5
    """Number of results to return from the query"""
    input_key: str = "query"  #: :meta private:
    output_key: str = "result"  #: :meta private:
    return_intermediate_steps: bool = False
    """Whether or not to return the intermediate steps along with the final answer."""
    return_direct: bool = False
    """Whether or not to return the result of querying the SQL table directly."""

    def _call(self, inputs: Dict[str, Any], run_manager: Optional[CallbackManagerForChainRun] = None) -> Dict[str, Any]:
        raise NotImplementedError

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @root_validator(pre=True)
    def raise_deprecation(cls, values: Dict) -> Dict:
        if "llm" in values:
            warnings.warn(
                "Directly instantiating an AsyncSQLDatabaseChain with an llm is deprecated. "
                "Please instantiate with llm_chain argument or using the from_llm "
                "class method."
            )
            if "llm_chain" not in values and values["llm"] is not None:
                database = values["database"]
                prompt = values.get("prompt") or SQL_PROMPTS.get(
                    database.dialect, PROMPT
                )
                values["llm_chain"] = LLMChain(llm=values["llm"], prompt=prompt)
        return values

    @property
    def input_keys(self) -> List[str]:
        """Return the singular input key.

        :meta private:
        """
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        """Return the singular output key.

        :meta private:
        """
        if not self.return_intermediate_steps:
            return [self.output_key]
        else:
            return [self.output_key, "intermediate_steps"]

    async def _acall(
            self,
            inputs: Dict[str, Any],
            run_manager: Optional[AsyncRunManager] = None,
    ) -> Dict[str, Any]:
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        input_text = f"{inputs[self.input_key]}\nSQLQuery:"
        await _run_manager.on_text(input_text, verbose=self.verbose)
        # If not present, then defaults to None which is all tables.
        table_names_to_use = inputs.get("table_names_to_use")
        table_info = await self.database.get_table_info(table_names=table_names_to_use)
        llm_inputs = {
            "input": input_text,
            "top_k": self.top_k,
            "dialect": self.database.dialect,
            "table_info": table_info,
            "stop": ["\nSQLResult:"],
        }
        intermediate_steps = []
        sql_cmd = await self.llm_chain.apredict(
            callbacks=_run_manager.get_child(), **llm_inputs
        )
        sql_cmd = sql_cmd.strip(' "')
        intermediate_steps.append(sql_cmd)
        await _run_manager.on_text(sql_cmd, color="green", verbose=self.verbose)
        result = await self.database.run(sql_cmd) if sql_cmd != "N/A" else "N/A"
        intermediate_steps.append(result)
        await _run_manager.on_text("\nSQLResult: ", verbose=self.verbose)
        await _run_manager.on_text(result, color="yellow", verbose=self.verbose)
        # If return direct, we just set the final result equal to the sql query
        if self.return_direct:
            final_result = result
        else:
            await _run_manager.on_text("\nAnswer:", verbose=self.verbose)
            input_text += f"{sql_cmd}\nSQLResult: {result}\nAnswer:"
            llm_inputs["input"] = input_text
            final_result = self.llm_chain.predict(
                callbacks=_run_manager.get_child(), **llm_inputs
            )
            await _run_manager.on_text(final_result, color="green", verbose=self.verbose)
        chain_result: Dict[str, Any] = {self.output_key: final_result}
        if self.return_intermediate_steps:
            chain_result["intermediate_steps"] = intermediate_steps
        return chain_result

    @property
    def _chain_type(self) -> str:
        return "sql_database_chain"

    @classmethod
    def from_llm(
            cls,
            llm: BaseLanguageModel,
            db: AsyncSQLDatabase,
            prompt: Optional[BasePromptTemplate] = None,
            **kwargs: Any,
    ) -> AsyncSQLDatabaseChain:
        prompt = prompt or SQL_PROMPTS.get(db.dialect, PROMPT)
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        return cls(llm_chain=llm_chain, database=db, **kwargs)


class AsyncSQLDatabaseSequentialChain(Chain):
    """Chain for querying SQL database that is a sequential chain.

    The chain is as follows:
    1. Based on the query, determine which tables to use.
    2. Based on those tables, call the normal SQL database chain.

    This is useful in cases where the number of tables in the database is large.
    """

    def _call(self, inputs: Dict[str, Any], run_manager: Optional[CallbackManagerForChainRun] = None) -> Dict[str, Any]:
        raise NotImplementedError

    decider_chain: LLMChain
    sql_chain: AsyncSQLDatabaseChain
    input_key: str = "query"  #: :meta private:
    output_key: str = "result"  #: :meta private:
    return_intermediate_steps: bool = False

    @classmethod
    def from_llm(
            cls,
            llm: BaseLanguageModel,
            database: AsyncSQLDatabase,
            query_prompt: BasePromptTemplate = PROMPT,
            decider_prompt: BasePromptTemplate = DECIDER_PROMPT,
            **kwargs: Any,
    ) -> AsyncSQLDatabaseSequentialChain:
        """Load the necessary chains."""
        sql_chain = AsyncSQLDatabaseChain(
            llm=llm, database=database, prompt=query_prompt, **kwargs
        )
        decider_chain = LLMChain(
            llm=llm, prompt=decider_prompt, output_key="table_names"
        )
        return cls(sql_chain=sql_chain, decider_chain=decider_chain, **kwargs)

    @property
    def input_keys(self) -> List[str]:
        """Return the singular input key.

        :meta private:
        """
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        """Return the singular output key.

        :meta private:
        """
        if not self.return_intermediate_steps:
            return [self.output_key]
        else:
            return [self.output_key, "intermediate_steps"]

    async def _acall(
            self,
            inputs: Dict[str, Any],
            run_manager: Optional[AsyncRunManager] = None,
    ) -> Dict[str, Any]:
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        _table_names = self.sql_chain.database.get_usable_table_names()
        table_names = ", ".join(_table_names)
        llm_inputs = {
            "query": inputs[self.input_key],
            "table_names": table_names,
        }
        table_names_to_use = await self.decider_chain.apredict_and_parse(
            callbacks=_run_manager.get_child(), **llm_inputs
        )
        await _run_manager.on_text("Table names to use:", end="\n", verbose=self.verbose)
        await _run_manager.on_text(
            str(table_names_to_use), color="yellow", verbose=self.verbose
        )
        new_inputs = {
            self.sql_chain.input_key: inputs[self.input_key],
            "table_names_to_use": table_names_to_use,
        }
        return self.sql_chain(
            new_inputs, callbacks=_run_manager.get_child(), return_only_outputs=True
        )

    @property
    def _chain_type(self) -> str:
        return "sql_database_sequential_chain"
