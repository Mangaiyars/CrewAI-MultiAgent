# Multi-Agent Customer Support (CrewAI + Streamlit)

A customer support system built with three CrewAI agents that run **sequentially**, wrapped in a Streamlit UI.

## How it works

1. **Assistant** — answers the user's query directly from its own knowledge.
2. **Web Search Assistant** — searches the web (via Serper) and answers based on the results.
3. **Entry Agent** — writes the query and both answers to `answers.txt`, and returns both answers to be shown in the UI.

The three agents run in order using CrewAI's `Process.sequential`. The Assistant and Web Search Assistant each return a structured `AnswerOutput` (a Pydantic model with a single `answer` field) instead of free-form text, so the Entry Agent gets a reliable, well-defined value to copy into the file — not text it has to re-interpret.

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

5. Open the app at `http://localhost:8501`, enter a query, and click **Submit**.

## Output

Each run appends/overwrites `answers.txt` with the query and both agents' answers, and displays both answers in the Streamlit UI.

## Tech stack

Python · CrewAI · crewai-tools (SerperDevTool, FileWriterTool) · Streamlit · OpenAI

## Notes

- API keys are read from environment variables only — never hard-coded or committed (`.env` is git-ignored).
- All application code lives in a single file: `app.py`.
- The app checks for required API keys on startup, and handles empty queries, crew execution failures, and missing task output gracefully instead of crashing.
