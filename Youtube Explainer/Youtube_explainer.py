import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv

load_dotenv()

def video_id(url):
    if "youtu.be" in url:
        return url.split('/')[-1]
    elif "watch?v=" in url:
        return url.split("watch?v=")[1].split('&')[0]
    else:
        return None

def join_docs(query):
    return "\n\n".join(doc.page_content for doc in query)

# Embed & LLM setup
embed_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)

llm = HuggingFaceEndpoint(
    repo_id='meta-llama/Llama-3.1-8B-Instruct',
    task='text-generation'
)
model = ChatHuggingFace(llm=llm)

# Streamlit UI
st.title("🎥 YouTube Explainer")

video_url = st.text_input("Enter YouTube video URL:")

if video_url:
    question = st.text_input("Ask a question about the video:")

    if question:
        try:
            # Get transcript
            id = video_id(video_url)
            transcript_list = YouTubeTranscriptApi.get_transcript(id, languages=['en'])
            transcript = " ".join(chunk["text"] for chunk in transcript_list)

            # Chunk + Embed + Retrieve
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = splitter.create_documents([transcript])
            vector_store = FAISS.from_documents(chunks, embed_model)
            retriever = vector_store.as_retriever(search_kwargs={"k": 4})
            context = join_docs(retriever.get_relevant_documents(question))

            # Prompt and Chain without memory
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful assistant. Use the video context to answer the user's question."),
                ("human", "{input}")
            ])
            chain = LLMChain(llm=model, prompt=prompt)

            full_input = f"Video Context:\n{context}\n\nQuestion:\n{question}"
            answer = chain.run(input=full_input)

            # Display answer
            st.markdown(f"**You:** {question}")
            st.markdown(f"**Bot:** {answer}")

        except TranscriptsDisabled:
            st.error("Transcript is disabled for this video.")
        except Exception as e:
            st.error(f"Error: {str(e)}")
