"""Streamlit showcase app for the Research Assistant.

Run via:
    uv run streamlit run streamlit_app/app.py --server.fileWatcherType=none

Layout: single page with `st.tabs` covering Chat / Pipeline Trace /
Retrieval Lens / Memory / Eval Console / Settings / Architecture.

The whole app binds to a SINGLE persistent asyncio event loop stored in
`st.session_state["loop"]`. Every async call uses
`loop.run_until_complete(...)` — see `streamlit_app/state.py` for why.
"""
