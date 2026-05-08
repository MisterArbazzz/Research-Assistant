"""PLACEHOLDER — use-case-specific @tool definitions.

Replace during the build. Conventions:

- All tools async
- Pydantic args_schema and Pydantic return type
- Tools that hit Neo4j use parametrized Cypher from neo4j_client
- Tools that hit external services delegate to src/adapters/<service>.py
- LLM-driven tools use `with_structured_output(SomeModel)` on the chat client

Reference shape:

    from __future__ import annotations
    from langchain_core.tools import tool
    from pydantic import BaseModel, Field

    class MyToolArgs(BaseModel):
        param: str = Field(...)

    class MyToolResult(BaseModel):
        result: str

    @tool(args_schema=MyToolArgs)
    async def my_tool(param: str) -> MyToolResult:
        '''Single-sentence description of what the tool does (used by LLMs that bind it).'''
        # ...
        return MyToolResult(result="...")
"""

from __future__ import annotations
