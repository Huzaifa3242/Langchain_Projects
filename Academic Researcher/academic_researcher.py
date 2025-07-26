import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain_community.tools import WikipediaQueryRun, ArxivQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper

load_dotenv()

st.set_page_config(page_title="AI Academic Assistant", page_icon="🤖", layout="centered")

# Custom CSS styling
st.markdown("""
    <style>
        .main {
            background-color: #f2f4f8;
            font-family: 'Segoe UI', sans-serif;
        }
        .stButton>button {
            color: white;
            background-color: #4A90E2;
            padding: 10px 20px;
            border-radius: 8px;
            margin-top: 10px;
        }
        .stTextInput>div>input {
            border-radius: 8px;
            border: 1px solid #ccc;
        }
        .css-1aumxhk {
            background-color: #ffffff;
            border-radius: 12px;
            padding: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("📚 AI Academic Assistant")

# TOOLS 
wiki_api = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=1000)
wiki_tool = WikipediaQueryRun(api_wrapper=wiki_api)

arxiv_api = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=1000)
arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_api)

tools = [wiki_tool, arxiv_tool]

# MEMORY 
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# --- LLM ---
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# --- AGENT ---
if "agent_executor" not in st.session_state:
    st.session_state.agent_executor = initialize_agent(
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        tools=tools,
        llm=llm,
        memory=st.session_state.memory,
        verbose=True,
        handle_parsing_errors=True
    )

#  Display Previous Chat from LangChain Memory 
for msg in st.session_state.memory.chat_memory.messages:
    if msg.type == "human":
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif msg.type == "ai":
        with st.chat_message("assistant"):
            st.markdown(msg.content)

# Chat Input 
user_input = st.chat_input("Ask me anything...")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        response = st.session_state.agent_executor.run(user_input)
        st.markdown(response)
