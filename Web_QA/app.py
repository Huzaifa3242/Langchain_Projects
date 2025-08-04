import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from dotenv import load_dotenv

load_dotenv()

# Embedding and LLM models
embed_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Streamlit app title
st.title("WEB QA Bot")

# Session state setup
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "url_processed" not in st.session_state:
    st.session_state.url_processed = None

# Input fields
url = st.text_input("Enter the URL:")
query = st.text_input("Ask your question:")
# Prompt template
prompt = ChatPromptTemplate.from_template("""
Answer the question using the context below. If the context is insufficient or irrelevant, just say i dont know and do not hallucinate
<context>
{context}
</context>
Question: {input}
""")

# Load & Process URL 
if url and url != st.session_state.url_processed:
    with st.spinner("Loading and processing URL..."):
        loader = WebBaseLoader(url)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = splitter.split_documents(docs)
        vector_store = FAISS.from_documents(texts, embed_model)
        st.session_state.vector_store = vector_store
        st.session_state.url_processed = url
    st.success("URL content loaded!")



if st.session_state.vector_store and query:
    retriever = st.session_state.vector_store.as_retriever()
    document_chain = create_stuff_documents_chain(llm, prompt)
    chain = create_retrieval_chain(retriever, document_chain)
    
    result = chain.invoke({"input": query})
    st.write("**Answer:**", result['answer'])
