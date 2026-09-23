import streamlit as st
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import SerperDevTool, FileWriterTool

load_dotenv()  # Load OPENAI_API_KEY and SERPER_API_KEY from .env

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
    expected_output="A clear, direct answer to the query.",
    agent=assistant,
)

task_web_search = Task(
    description="Search the web for this query and answer it based on what you find: {query}",
    expected_output="A clear answer to the query, based on web search results.",
    agent=web_search_assistant,
)

task_entry = Task(
    description=(
        "You will receive the original query, the Assistant's answer, and the "
        "Web Search Assistant's answer. Use the file writer tool to save all "
        "three into a file named 'answers.txt' (overwrite it each run) in this "
        "format:\n\n"
        "Query: <query>\n"
        "Assistant Answer: <answer 1>\n"
        "Web Search Answer: <answer 2>\n\n"
        "Then return both answers clearly labeled so they can be shown to the user."
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

st.set_page_config(page_title="Multi-Agent Customer Support", page_icon="🛟")
st.title("🛟 Multi-Agent Customer Support")
st.write("Ask a question and three agents will work together to answer it.")

query = st.text_input("Enter your query or task:")

if st.button("Submit") and query.strip():
    with st.spinner("Agents are working on your query..."):
        crew.kickoff(inputs={"query": query})

    st.subheader("Assistant Answer")
    st.write(task_assistant.output.raw)

    st.subheader("Web Search Answer")
    st.write(task_web_search.output.raw)

    st.success("Both answers were saved to answers.txt")
