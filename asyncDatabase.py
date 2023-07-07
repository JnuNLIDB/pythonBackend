"""SQLAlchemy wrapper around a database."""
from __future__ import annotations

import warnings
from typing import Any, Iterable, List, Optional

import sqlalchemy
from sqlalchemy import (
    MetaData,
    Table,
    inspect,
    select,
    text, Connection,
)
from sqlalchemy.engine import CursorResult
from sqlalchemy.exc import ProgrammingError, SQLAlchemyError
from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine, AsyncConnection
from sqlalchemy.schema import CreateTable


def _format_index(index: sqlalchemy.engine.interfaces.ReflectedIndex) -> str:
    return (
        f'Name: {index["name"]}, Unique: {index["unique"]},'
        f' Columns: {str(index["column_names"])}'
    )


class AsyncSQLDatabase:
    """SQLAlchemy wrapper around a database."""

    def __init__(
            self,
            engine: AsyncEngine,
            schema: Optional[str] = None,
            metadata: Optional[MetaData] = None,
            ignore_tables: Optional[List[str]] = None,
            include_tables: Optional[List[str]] = None,
            sample_rows_in_table_info: int = 3,
            indexes_in_table_info: bool = False,
            custom_table_info: Optional[dict] = None,
            view_support: bool = False,
    ):
        """Create engine from database URI."""
        self._usable_tables = None
        self._engine = engine
        self._schema = schema
        if include_tables and ignore_tables:
            raise ValueError("Cannot specify both include_tables and ignore_tables")

        if not isinstance(sample_rows_in_table_info, int):
            raise TypeError("sample_rows_in_table_info must be an integer")

        self._sample_rows_in_table_info = sample_rows_in_table_info
        self._indexes_in_table_info = indexes_in_table_info
        self._include_tables = set(include_tables) if include_tables else set()
        self._ignore_tables = set(ignore_tables) if ignore_tables else set()

        self._custom_table_info = custom_table_info
        if self._custom_table_info:
            if not isinstance(self._custom_table_info, dict):
                raise TypeError(
                    "table_info must be a dictionary with table names as keys and the "
                    "desired table info as values"
                )
            # only keep the tables that are also present in the database
            intersection = set(self._custom_table_info).intersection(self._all_tables)
            self._custom_table_info = dict(
                (table, self._custom_table_info[table])
                for table in self._custom_table_info
                if table in intersection
            )
        self._all_tables = set()

        self._metadata = metadata or MetaData()
        # including view support if view_support = true

    @staticmethod
    def set_inspector(conn: Connection, self: AsyncSQLDatabase, view_support: bool) -> AsyncSQLDatabase:
        self._inspector = inspect(conn)

        self._all_tables = set(
            self._inspector.get_table_names(schema=self._schema)
            + (self._inspector.get_view_names(schema=self._schema) if view_support else [])
        )

        if self._include_tables:
            missing_tables = self._include_tables - self._all_tables
            if missing_tables:
                raise ValueError(
                    f"include_tables {missing_tables} not found in database"
                )
        if self._ignore_tables:
            missing_tables = self._ignore_tables - self._all_tables
            if missing_tables:
                raise ValueError(
                    f"ignore_tables {missing_tables} not found in database"
                )
        usable_tables = self.get_usable_table_names()
        self._usable_tables = set(usable_tables) if usable_tables else self._all_tables

        self._metadata.reflect(
            views=view_support,
            bind=conn,
            only=list(self._usable_tables),
            schema=self._schema,
        )
        return self

    async def reflect_metadata(self, view_support: bool) -> AsyncSQLDatabase:
        """Reflect metadata from database."""
        async with self._engine.begin() as conn:
            await conn.run_sync(self.set_inspector, self, view_support)
            conn: AsyncConnection = conn
        return self

    @classmethod
    async def from_uri(
            cls, database_uri: str, engine_args: Optional[dict] = None, **kwargs: Any
    ) -> AsyncSQLDatabase:
        """Construct a SQLAlchemy engine from URI."""
        _engine_args = engine_args or {}
        return await cls(create_async_engine(database_uri, **_engine_args), **kwargs) \
            .reflect_metadata(kwargs['view_support'] if 'view_support' in kwargs else False)

    @property
    def dialect(self) -> str:
        """Return string representation of dialect to use."""
        return self._engine.dialect.name

    def get_usable_table_names(self) -> Iterable[str]:
        """Get names of tables available."""
        if self._include_tables:
            return self._include_tables
        return self._all_tables - self._ignore_tables

    def get_table_names(self) -> Iterable[str]:
        """Get names of tables available."""
        warnings.warn(
            "This method is deprecated - please use `get_usable_table_names`."
        )
        return self.get_usable_table_names()

    @property
    async def table_info(self) -> str:
        """Information about all tables in the database."""
        return await self.get_table_info()

    async def get_table_info(self, table_names: Optional[List[str]] = None) -> str:
        """Get information about specified tables.

        Follows best practices as specified in: Rajkumar et al, 2022
        (https://arxiv.org/abs/2204.00498)

        If `sample_rows_in_table_info`, the specified number of sample rows will be
        appended to each table description. This can increase performance as
        demonstrated in the paper.
        """
        all_table_names = self.get_usable_table_names()
        if table_names is not None:
            missing_tables = set(table_names).difference(all_table_names)
            if missing_tables:
                raise ValueError(f"table_names {missing_tables} not found in database")
            all_table_names = table_names

        meta_tables = [
            tbl
            for tbl in self._metadata.sorted_tables
            if tbl.name in set(all_table_names)
               and not (self.dialect == "sqlite" and tbl.name.startswith("sqlite_"))
        ]

        tables = []
        async with self._engine.begin() as conn:
            for table in meta_tables:
                if self._custom_table_info and table.name in self._custom_table_info:
                    tables.append(self._custom_table_info[table.name])
                    continue

                # add create table command
                create_table = str(CreateTable(table).compile(conn))
                table_info = f"{create_table.rstrip()}"
                has_extra_info = (
                        self._indexes_in_table_info or self._sample_rows_in_table_info
                )
                if has_extra_info:
                    table_info += "\n\n/*"
                if self._indexes_in_table_info:
                    table_info += f"\n{self._get_table_indexes(table)}\n"
                if self._sample_rows_in_table_info:
                    table_info += f"\n{await self._get_sample_rows(table)}\n"
                if has_extra_info:
                    table_info += "*/"
                tables.append(table_info)
        final_str = "\n\n".join(tables)
        return final_str

    def _get_table_indexes(self, table: Table) -> str:
        indexes = self._inspector.get_indexes(table.name)
        indexes_formatted = "\n".join(map(_format_index, indexes))
        return f"Table Indexes:\n{indexes_formatted}"

    async def _get_sample_rows(self, table: Table) -> str:
        # build the select command
        command = select(table).limit(self._sample_rows_in_table_info)

        # save the columns in string format
        columns_str = "\t".join([col.name for col in table.columns])

        try:
            # get the sample rows
            async with self._engine.connect() as connection:
                sample_rows_result: CursorResult = await connection.execute(command)
                # shorten values in the sample rows
                sample_rows = list(
                    map(lambda ls: [str(i)[:100] for i in ls], sample_rows_result)
                )

            # save the sample rows in string format
            sample_rows_str = "\n".join(["\t".join(row) for row in sample_rows])

        # in some dialects when there are no rows in the table a
        # 'ProgrammingError' is returned
        except ProgrammingError:
            sample_rows_str = ""

        return (
            f"{self._sample_rows_in_table_info} rows from {table.name} table:\n"
            f"{columns_str}\n"
            f"{sample_rows_str}"
        )

    async def run(self, command: str, fetch: str = "all") -> str:
        """Execute a SQL command and return a string representing the results.

        If the statement returns rows, a string of the results is returned.
        If the statement returns no rows, an empty string is returned.
        """
        async with self._engine.begin() as connection:
            connection: AsyncConnection
            if self._schema is not None:
                await connection.exec_driver_sql(f"SET search_path TO {self._schema}")
            cursor = await connection.execute(text(command))
            if cursor.returns_rows:
                if fetch == "all":
                    result = cursor.fetchall()
                elif fetch == "one":
                    result = cursor.fetchone()[0]  # type: ignore
                else:
                    raise ValueError("Fetch parameter must be either 'one' or 'all'")
                return str(result)
        return ""

    async def get_table_info_no_throw(self, table_names: Optional[List[str]] = None) -> str:
        """Get information about specified tables.

        Follows best practices as specified in: Rajkumar et al, 2022
        (https://arxiv.org/abs/2204.00498)

        If `sample_rows_in_table_info`, the specified number of sample rows will be
        appended to each table description. This can increase performance as
        demonstrated in the paper.
        """
        try:
            return await self.get_table_info(table_names)
        except ValueError as e:
            """Format the error message"""
            return f"Error: {e}"

    async def run_no_throw(self, command: str, fetch: str = "all") -> str:
        """Execute a SQL command and return a string representing the results.

        If the statement returns rows, a string of the results is returned.
        If the statement returns no rows, an empty string is returned.

        If the statement throws an error, the error message is returned.
        """
        try:
            return await self.run(command, fetch)
        except SQLAlchemyError as e:
            """Format the error message"""
            return f"Error: {e}"
