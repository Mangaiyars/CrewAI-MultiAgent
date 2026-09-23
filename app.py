import os

import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import SerperDevTool, FileWriterTool


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

st.set_page_config(page_title="CrewAI - Multi-Agent Support", page_icon="🛟")
st.title("🛟 CrewAI - Multi-Agent Support")
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

# ---------------- Agents ----------------

assistant = Agent(
    role="Assistant",
    goal="Answer the user's query directly using your own knowledge",
    backstory=(
        "You are a helpful customer support assistant. You answer questions "
        "using only what you already know, without searching the internet."
    ),
    llm=llm,
    verbose=True,
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
    description="Answer this user query directly, using only your own knowledge: {query}",
    expected_output="A clear, direct answer to the query, in the 'answer' field.",
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
        "field and the Web Search Assistant's 'answer' field. Use the file writer "
        "tool to save all three into a file named 'answers.txt' (overwrite it each "
        "run) in this exact format:\n\n"
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

query = st.text_input("Enter your query or task:")

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
        st.subheader("Assistant Answer")
        st.write(assistant_answer)
    else:
        st.warning("The Assistant did not return an answer.")

    if web_search_answer:
        st.subheader("Web Search Answer")
        st.write(web_search_answer)
    else:
        st.warning("The Web Search Assistant did not return an answer.")

    if os.path.exists("answers.txt"):
        st.success("Both answers were saved to answers.txt")
    else:
        st.warning("The Entry Agent finished, but answers.txt was not found.")