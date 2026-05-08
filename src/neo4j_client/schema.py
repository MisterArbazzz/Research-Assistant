"""Apply Cypher schema and static seed files to Neo4j.

Splits multi-statement Cypher files on `;` after stripping `//` line comments,
runs each statement in its own session, then awaits indexes so vector queries
issued seconds after seed don't race the index build.

Idempotent: every CREATE uses IF NOT EXISTS; every node uses MERGE.

Run as: `uv run python -m src.neo4j_client.schema`
"""

from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path

from .driver import Neo4jClient

logger = logging.getLogger(__name__)

SEED_DIR = Path(__file__).resolve().parents[2] / "data" / "seed"


def _split_statements(cypher_text: str) -> list[str]:
    """Split a multi-statement Cypher file on top-level `;`.

    Quote-aware so semicolons inside `"..."` / `'...'` string literals don't
    break statements (e.g. `voice.excerpt = "Never sell; always inform."`).
    Honors backslash escapes inside strings.
    """
    # Strip `//` line comments. Our seed files do not embed `//` inside string
    # literals, so a regex is safe; revisit if that changes.
    text = re.sub(r"//[^\n]*", "", cypher_text)

    statements: list[str] = []
    buf: list[str] = []
    in_single = False
    in_double = False
    i = 0
    while i < len(text):
        ch = text[i]
        # Honor backslash escapes inside string literals (pass through pair).
        if (in_single or in_double) and ch == "\\" and i + 1 < len(text):
            buf.append(ch)
            buf.append(text[i + 1])
            i += 2
            continue
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        if ch == ";" and not in_single and not in_double:
            stmt = "".join(buf).strip()
            if stmt:
                statements.append(stmt)
            buf = []
        else:
            buf.append(ch)
        i += 1
    tail = "".join(buf).strip()
    if tail:
        statements.append(tail)
    return statements


async def apply_cypher_file(client: Neo4jClient, path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    statements = _split_statements(text)
    logger.info(
        "applying cypher file",
        extra={"path": str(path), "statement_count": len(statements)},
    )
    async with client.session() as session:
        for stmt in statements:
            await session.run(stmt)


async def apply_schema_and_static(client: Neo4jClient) -> None:
    await apply_cypher_file(client, SEED_DIR / "01_schema.cypher")
    # Wait for indexes to come online before any seed step issues vector queries.
    async with client.session() as session:
        await session.run("CALL db.awaitIndexes()")
    await apply_cypher_file(client, SEED_DIR / "02_seed_static.cypher")


async def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    client = Neo4jClient()
    await client.connect()
    try:
        await apply_schema_and_static(client)
        logger.info("schema + static seed applied")
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
