"""Port-adapters for external integrations.

Each adapter exposes one async function. Live mode hits the real API;
mock mode calls `emit_mock(...)` from `_mock_helpers.py` which writes a
Mock node to Neo4j AND emits an SSE event so the UI renders the side-effect.
Mode selected via `settings.<X>_MODE`. Default is mock.

Use-case-specific adapters are added here as `<service>.py` files.
"""
