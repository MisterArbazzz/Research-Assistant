.PHONY: help install neo4j-up neo4j-down seed dev test test-integration lint format clean

help:
	@echo "make install         — uv sync the dependencies"
	@echo "make neo4j-up        — start Neo4j via docker compose"
	@echo "make neo4j-down      — stop Neo4j and remove volumes"
	@echo "make seed            — run the use-case-specific seed script"
	@echo "make dev             — run uvicorn with --reload"
	@echo "make test            — pytest non-integration"
	@echo "make test-integration — pytest live integration tests"
	@echo "make lint            — ruff format + ruff check + mypy strict"
	@echo "make format          — ruff format only"
	@echo "make clean           — remove caches"

install:
	uv sync

neo4j-up:
	docker compose up -d neo4j
	@echo "Waiting for Neo4j..."
	@until docker compose ps neo4j 2>/dev/null | grep -q "healthy"; do sleep 2; done
	@echo "Neo4j is ready at http://localhost:7474"

neo4j-down:
	docker compose down -v

seed:
	@if [ -f data/seed/run_seed.sh ]; then \
		bash data/seed/run_seed.sh; \
	else \
		echo "No seed script — write data/seed/run_seed.sh for your use case."; \
	fi

dev:
	uv run uvicorn src.server:app --reload --host 0.0.0.0 --port 8000

test:
	uv run pytest -m "not integration" -v

test-integration:
	uv run pytest -m integration -v

lint:
	uv run ruff format src/ tests/
	uv run ruff check --fix src/ tests/
	uv run mypy --strict src/

format:
	uv run ruff format src/ tests/

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
