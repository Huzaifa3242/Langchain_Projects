from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings,HuggingFaceEndpoint
from langchain_community.tools import WikipediaQueryRun,ArxivQueryRun,google_scholar
from langchain_community.utilities import WikipediaAPIWrapper,ArxivAPIWrapper,GoogleScholarAPIWrapper
from langchain.agents import create_openai_tools_agent,AgentExecutor,create_react_agent
from langchain import hub
from langchain.memory import ConversationBufferMemory
memory = ConversationBufferMemory(return_messages=True)

from dotenv import load_dotenv
load_dotenv()

# Tool 1
wiki_api=WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=1000)
wiki_tool = WikipediaQueryRun(api_wrapper=wiki_api)

# Tool 2
arxiv_api=ArxivAPIWrapper(top_k_results=1,doc_content_chars_max=1000)
arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_api)

tools = [wiki_tool,arxiv_tool]

prompt = hub.pull('hwchase17/react')
# print(prompt.messages)
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0
)

# Agent
agent=create_react_agent(
    llm=llm,
    prompt=prompt,
    tools=tools
)

executer=AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    memory=memory
)

response=executer.invoke({"input":"What is machine learning ?"})

print(response)
print(response['output'])