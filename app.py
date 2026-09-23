import os

import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool
from crewai_tools import SerperDevTool, FileWriterTool
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Resolve paths relative to this script's location, not the current working
# directory, so the app behaves the same no matter where `streamlit run` is
# launched from.
APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(APP_DIR, "data")
ANSWERS_FILE = os.path.join(APP_DIR, "answers.txt")

SUGGESTED_QUESTIONS = [
    "How do I reset my password?",
    "What is your refund policy?",
    "What is the latest version of Python?",
    "What are today's top tech news headlines?",
]


class AnswerOutput(BaseModel):
    """Structured output so downstream agents get a clean field to read,
    instead of having to re-parse free-form text."""
    answer: str


load_dotenv()  # Load OPENAI_API_KEY and SERPER_API_KEY from .env

# ---------------- Environment check ----------------
# Fail fast with a clear message instead of a confusing crash mid-run.
missing_keys = [
    key for key in ("OPENAI_API_KEY", "SERPER_API_KEY") if not os.getenv(key)
]

st.set_page_config(page_title="Multi-Agent Customer Support", page_icon="🛟")
st.title("🛟 Multi-Agent Customer Support")
st.write("Ask a question and three agents will work together to answer it.")

if missing_keys:
    st.error(
        "Missing required environment variable(s): "
        f"{', '.join(missing_keys)}. Add them to your .env file and restart the app."
    )
    st.stop()

llm = LLM(model="gpt-4o-mini")
search_tool = SerperDevTool()
file_writer_tool = FileWriterTool()


# ---------------- Knowledge base (RAG) ----------------
# Builds a FAISS vector store from the .txt files in ./data so the Assistant
# can answer from the company's own knowledge base (RAG) instead of only
# generic model knowledge. Cached with st.cache_resource so it's built once
# per app process, not rebuilt on every query. NOTE: this cache is shared
# across all users of the running app - if you edit files in data/, use the
# "Reload knowledge base" button below (or restart the app) to pick them up.
@st.cache_resource(show_spinner="Loading knowledge base...")
def build_vector_db():
    if not os.path.isdir(DATA_DIR):
        raise FileNotFoundError(
            "No 'data' folder found. Add your .txt knowledge base files to a "
            "'data' folder next to app.py."
        )

    loader = DirectoryLoader(
        DATA_DIR,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"autodetect_encoding": True},
    )
    docs = loader.load()

    if not docs:
        raise ValueError(
            "No .txt files found in the 'data' folder. Add at least one "
            "knowledge base document."
        )

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return FAISS.from_documents(chunks, embeddings)


try:
    vector_db = build_vector_db()
except Exception as e:
    st.error(f"Could not load the knowledge base: {e}")
    st.stop()

with st.sidebar:
    st.caption("Knowledge base")
    if st.button("🔄 Reload knowledge base"):
        build_vector_db.clear()  # clear this function's cached result
        st.rerun()


@tool("Customer Support Knowledge Base")
def knowledge_base_tool(query: str) -> str:
    """Search company FAQs, policies, pricing, refund rules, product
    documentation, and other customer support knowledge base content."""
    results = vector_db.similarity_search(query, k=4)
    return "\n\n".join(doc.page_content for doc in results)


# ---------------- Agents ----------------

assistant = Agent(
    role="Assistant",
    goal="Answer the user's query directly using the company knowledge base",
    backstory=(
        "You are a helpful customer support assistant. You answer questions "
        "using only the company's own knowledge base (FAQs, policies, product "
        "docs) via the Customer Support Knowledge Base tool - never the open "
        "internet. If the knowledge base doesn't cover it, say so honestly."
    ),
    tools=[knowledge_base_tool],
    llm=llm,
    verbose=True,
    allow_delegation=False,
)

web_search_assistant = Agent(
    role="Web Search Assistant",
    goal="Search the web for the query and answer it using the search results",
    backstory=(
        "You are a research-minded support assistant. You always search the "
        "web first, then base your answer on what you find."
    ),
    llm=llm,
    tools=[search_tool],
    verbose=True,
)

entry_agent = Agent(
    role="Entry Agent",
    goal="Save the query and both answers to a text file and present them to the user",
    backstory=(
        "You are responsible for record-keeping. You save every query and its "
        "two answers to a file, then give the user a clean, labeled summary."
    ),
    llm=llm,
    tools=[file_writer_tool],
    verbose=True,
)

# ---------------- Tasks ----------------

task_assistant = Task(
    description=(
        "Search the company knowledge base for this query using the Customer "
        "Support Knowledge Base tool and answer it based only on what you find "
        "there: {query}\n\n"
        "Do not hallucinate. If the knowledge base doesn't cover it, say so "
        "honestly and suggest contacting the support team."
    ),
    expected_output=(
        "A clear, friendly, professional answer based only on the knowledge "
        "base, in the 'answer' field."
    ),
    agent=assistant,
    output_pydantic=AnswerOutput,
)

task_web_search = Task(
    description="Search the web for this query and answer it based on what you find: {query}",
    expected_output="A clear answer to the query, based on web search results, in the 'answer' field.",
    agent=web_search_assistant,
    output_pydantic=AnswerOutput,
)

task_entry = Task(
    description=(
        "The original query was: {query}\n\n"
        "You will also receive two structured outputs: the Assistant's 'answer' "
        "field (from the knowledge base) and the Web Search Assistant's 'answer' "
        f"field. Use the file writer tool to save all three into a file named "
        f"'answers.txt' in the directory '{APP_DIR}' (overwrite it each run) in "
        "this exact format:\n\n"
        "Query: {query}\n"
        "Assistant Answer: <assistant's answer field>\n"
        "Web Search Answer: <web search assistant's answer field>\n\n"
        "Copy the two answer fields exactly as given to you - do not summarize or "
        "reword them. Then return both answers clearly labeled so they can be shown "
        "to the user."
    ),
    expected_output="Confirmation the file was saved, plus both answers clearly labeled.",
    agent=entry_agent,
    context=[task_assistant, task_web_search],
)

crew = Crew(
    agents=[assistant, web_search_assistant, entry_agent],
    tasks=[task_assistant, task_web_search, task_entry],
    process=Process.sequential,
    verbose=True,
)

# ---------------- Streamlit UI ----------------

if "query_input" not in st.session_state:
    st.session_state.query_input = ""

st.write("**Try one of these, or type your own:**")
suggestion_cols = st.columns(len(SUGGESTED_QUESTIONS))
for col, suggestion in zip(suggestion_cols, SUGGESTED_QUESTIONS):
    if col.button(suggestion):
        st.session_state.query_input = suggestion
        st.rerun()

query = st.text_input("Enter your query or task:", key="query_input")

if st.button("Submit"):
    if not query.strip():
        st.warning("Please enter a query before submitting.")
        st.stop()

    try:
        with st.spinner("Agents are working on your query..."):
            crew.kickoff(inputs={"query": query})
    except Exception as e:
        st.error(f"Something went wrong while running the agents: {e}")
        st.stop()

    # Guard against a task not having produced output (e.g. it was skipped
    # or the crew stopped early) so the UI doesn't crash with an AttributeError.
    # Prefer the structured .pydantic.answer field; fall back to .raw if the
    # model didn't return valid structured output for some reason.
    def get_answer(task):
        if not task.output:
            return None
        if task.output.pydantic:
            return task.output.pydantic.answer
        return task.output.raw

    assistant_answer = get_answer(task_assistant)
    web_search_answer = get_answer(task_web_search)

    if assistant_answer:
        st.subheader("Assistant Answer (Knowledge Base)")
        st.write(assistant_answer)
    else:
        st.warning("The Assistant did not return an answer.")

    if web_search_answer:
        st.subheader("Web Search Answer")
        st.write(web_search_answer)
    else:
        st.warning("The Web Search Assistant did not return an answer.")

    if os.path.exists(ANSWERS_FILE):
        st.success("Both answers were saved to answers.txt")
    else:
        st.warning("The Entry Agent finished, but answers.txt was not found.")