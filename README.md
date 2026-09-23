# Multi-Agent Customer Support (CrewAI + Streamlit)

A customer support system built with three CrewAI agents that run **sequentially**, wrapped in a Streamlit UI.

## How it works

1. **Assistant** — answers the user's query using a RAG (Retrieval-Augmented Generation) tool that searches a FAISS vector store built from the company's own `.txt` knowledge base (FAQs, policies, etc.) — not open internet or generic model knowledge.
2. **Web Search Assistant** — searches the web (via Serper) and answers based on the results.
3. **Entry Agent** — writes the query and both answers to `answers.txt`, and returns both answers to be shown in the UI.

The three agents run in order using CrewAI's `Process.sequential`. The Assistant and Web Search Assistant each return a structured `AnswerOutput` (a Pydantic model with a single `answer` field) instead of free-form text, so the Entry Agent gets a reliable, well-defined value to copy into the file — not text it has to re-interpret.

### Knowledge base (RAG)

On startup, the app loads every `.txt` file in the `data/` folder (with automatic encoding detection), splits it into chunks (`RecursiveCharacterTextSplitter`), embeds the chunks with OpenAI embeddings, and stores them in a FAISS vector store. This is built once per app process using `st.cache_resource`, not rebuilt on every query. The Assistant agent queries this vector store through a custom `@tool`-decorated function (`Customer Support Knowledge Base`) to retrieve the most relevant chunks before answering.

Two sample files (`data/faq.txt`, `data/policies.txt`) are included so the app works out of the box — replace or add to them with your own company documents. If you edit files in `data/` while the app is running, use the **"🔄 Reload knowledge base"** button in the sidebar (or restart the app) to pick up the changes — the vector store is cached across the whole app session, not rebuilt per query.

### Suggested questions

A row of clickable example questions appears above the input box, covering both knowledge-base topics (password reset, refund policy) and web-search topics (latest Python version, today's tech news) — useful for quickly demonstrating both agents' behavior.

## Setup

1. Clone the repo and create a virtual environment:
   ```
   python -m venv venv
   venv\Scripts\Activate.ps1   # Windows PowerShell
   source venv/bin/activate    # macOS/Linux
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Copy `.env.example` to `.env` and add your keys:
   ```
   OPENAI_API_KEY=your-openai-key
   SERPER_API_KEY=your-serper-key
   ```

4. Run the app:
   ```
   streamlit run app.py
   ```

5. Open the app at `http://localhost:8501`, click a suggested question or type your own, and click **Submit**.

## Output

Each run overwrites `answers.txt` (saved next to `app.py`, regardless of which folder you launched the app from) with the query and both agents' answers, and displays both answers in the Streamlit UI.

## Tech stack

Python · CrewAI · crewai-tools (SerperDevTool, FileWriterTool) · LangChain (document loaders, text splitter, OpenAI embeddings, FAISS vector store) · Streamlit · OpenAI

## Notes

- API keys are read from environment variables only — never hard-coded or committed (`.env` is git-ignored).
- All application code lives in a single file: `app.py`. Knowledge base documents live in `data/`.
- The app checks for required API keys on startup, and handles empty queries, crew execution failures, missing/empty knowledge base folders, and missing task output gracefully instead of crashing.
- `data/` and `answers.txt` are resolved as absolute paths relative to `app.py`'s own location, so the app behaves consistently no matter which directory you run `streamlit run` from.
- To apply a light UI theme, place a `.streamlit/config.toml` file next to `app.py` (see the file included in this project).