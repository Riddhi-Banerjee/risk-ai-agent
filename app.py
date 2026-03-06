import streamlit as st
import pandas as pd
import os
from crewai import Agent, Task, Crew, Process
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool

# --- PAGE CONFIG ---
st.set_page_config(page_title="Risk Intelligence Command", layout="wide", page_icon="🛡️")
st.title("🛡️ Project Risk Intelligence (Multi-Agent)")

# --- API KEY CONFIG ---
# Automatically reads from Streamlit Cloud Secrets
api_key = st.secrets.get("GOOGLE_API_KEY") or st.sidebar.text_input("Enter Gemini API Key", type="password")

if not api_key:
    st.warning("Please provide a Google API Key in the sidebar or app secrets.")
    st.stop()

os.environ["GOOGLE_API_KEY"] = api_key
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1)

# --- CUSTOM TOOLS FOR AGENTS ---
@tool("read_project_data")
def read_project_data(project_id: str):
    """Searches project_risk_raw_dataset.csv for a specific project ID."""
    df = pd.read_csv('project_risk_raw_dataset.csv')
    result = df[df['Project_ID'] == project_id]
    return result.to_string() if not result.empty else "Project not found."

@tool("read_financial_data")
def read_financial_data(project_id: str):
    """Searches transaction.csv for all transactions related to a project ID."""
    df = pd.read_csv('transaction.csv')
    result = df[df['Project_ID'] == project_id]
    return result.to_string() if not result.empty else "No transactions found."

@tool("read_market_trends")
def read_market_trends(query: str):
    """Reads the latest market trends from market_trends.csv."""
    df = pd.read_csv('market_trends.csv')
    return df.tail(20).to_string() # Returns latest 20 entries

# --- AGENT DEFINITIONS (From your Blueprint) ---
market_analyst = Agent(
    role='Market Analysis Agent',
    goal='Identify external economic risks affecting the project.',
    backstory='Expert economist specializing in inflation and industry volatility.',
    tools=[read_market_trends],
    llm=llm
)

risk_scorer = Agent(
    role='Risk Scoring Agent',
    goal='Identify financial risks and payment delays.',
    backstory='Financial auditor focused on overdue payments and budget utilization.',
    tools=[read_financial_data],
    llm=llm
)

status_tracker = Agent(
    role='Project Status Tracking Agent',
    goal='Monitor internal project health and schedule delays.',
    backstory='Senior PM auditor tracking complexity and team size risks.',
    tools=[read_project_data],
    llm=llm
)

manager = Agent(
    role='Project Risk Manager',
    goal='Synthesize all findings into a final executive mitigation plan.',
    backstory='Chief Risk Officer responsible for final strategic decisions.',
    llm=llm
)

# --- UI & EXECUTION ---
target_id = st.text_input("Analyze Project ID:", "PROJ_0001")

if st.button("🚀 Run Agentic Audit"):
    # Define Tasks
    t1 = Task(description=f"Analyze market trends for {target_id}", agent=market_analyst, expected_output="Economic risk summary.")
    t2 = Task(description=f"Check all overdue payments for {target_id}", agent=risk_scorer, expected_output="Financial risk report.")
    t3 = Task(description=f"Review complexity and status for {target_id}", agent=status_tracker, expected_output="Internal health report.")
    t4 = Task(description="Create a 3-step mitigation strategy using all reports.", agent=manager, expected_output="Final Strategy Document.")

    # Form the Crew
    risk_crew = Crew(
        agents=[market_analyst, risk_scorer, status_tracker, manager],
        tasks=[t1, t2, t3, t4],
        process=Process.sequential,
        verbose=True
    )

    with st.status("Agents are collaborating...", expanded=True):
        result = risk_crew.kickoff()
    
    st.markdown("### 📊 Final Intelligence Report")
    st.markdown(result.raw)
