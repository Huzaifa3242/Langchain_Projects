import streamlit as st
from langchain.agents import create_react_agent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain_community.tools import WikipediaQueryRun, ArxivQueryRun, DuckDuckGoSearchResults
from langchain_community.utilities import WikipediaAPIWrapper,ArxivAPIWrapper,GoogleScholarAPIWrapper

from dotenv import load_dotenv
load_dotenv()

# Set page config
st.set_page_config(page_title="AI Academic Assistant", page_icon="🤖", layout="wide")

# Custom CSS for beautiful styling
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

st.title("📚 AI Academic Assistant Chatbot")

# Sidebar for mode selection
st.sidebar.header("Assistant Settings")
mode = st.sidebar.selectbox("Choose Mode", ["Scholar", "Code", "News"])

# Initialize LLM and Prompt
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0
)
prompt = hub.pull("hwchase17/react")

# Define tools per mode
def get_tools(mode):
    wiki_api=WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=1000)
    wiki_tool = WikipediaQueryRun(api_wrapper=wiki_api)
    arxiv_api=ArxivAPIWrapper(top_k_results=1,doc_content_chars_max=1000)
    arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_api)

    ddg = DuckDuckGoSearchResults()
    # pytool = PythonREPLTool()

    if mode == "Scholar":
        return [wiki_tool, arxiv_tool]
    elif mode == "Code":
        return ["HEllo world"]
    elif mode == "News":
        return [ddg]
    else:
        return [wiki_tool]

# Memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Store chat history in session
if "messages" not in st.session_state:
    st.session_state.messages = []

# Get tools for selected mode
tools = get_tools(mode)
agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=False)

# Chat interface
user_input = st.chat_input("Ask me anything...")

if user_input:
    st.session_state.messages.append(("user", user_input))
    with st.spinner("Thinking..."):
        result = executor.invoke({"input": user_input})
        st.session_state.messages.append(("assistant", result["output"]))

# Display chat messages
for role, msg in st.session_state.messages:
    with st.chat_message(role):
        st.markdown(msg)
