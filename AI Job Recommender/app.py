import streamlit as st
from src.helper import load_cv, get_llm_resp, get_all_jobs_list
from src.prompt import PROMPT
from src.pydantic_model import output
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()
st.markdown(
    """
    <style>
    .job-title {
        font-size: 26px;
        font-weight: bold;
        color: #11111; /* Purple for title */
        margin-bottom: 2px;
    }
    .company-name {
        font-size: 20px;
        font-style: italic;
        color: #11111; /* Subtle gray for company */
        margin-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.set_page_config(page_title="AI Job Recommender")
st.title("📑 AI Job Recommender")

file = st.file_uploader("Upload your resume (PDF)", type=["pdf"])

llm = ChatGroq(model="openai/gpt-oss-20b")

# Initialize session state variables
if "cv_analysis" not in st.session_state:
    st.session_state.cv_analysis = None
if "job_recommendations" not in st.session_state:
    st.session_state.job_recommendations = None

# Process CV only once
if file and st.session_state.cv_analysis is None:
    with st.spinner("Analyzing your CV..."):
        doc = load_cv(file)
        response = get_llm_resp(llm=llm, prompt=PROMPT, model=output, doc=doc)
        st.session_state.cv_analysis = {
            "score": response.score,
            "summary": response.summary,
            "improvement": response.improvement,
            "keywords": response.keywords
        }

# Display CV analysis if available
if st.session_state.cv_analysis:
    st.subheader("CV Analysis Results")

    st.metric("Score", f"{st.session_state.cv_analysis['score']} / 10")

    st.write("### Summary")
    st.markdown(st.session_state.cv_analysis['summary'], unsafe_allow_html=True)

    st.write("### Suggested Improvements")
    st.markdown(st.session_state.cv_analysis['improvement'], unsafe_allow_html=True)

    # Fetch jobs button
    if st.button("🔎 Get Job Recommendations"):
        with st.spinner("Fetching job recommendations..."):
            st.session_state.job_recommendations = get_all_jobs_list(
                st.session_state.cv_analysis['keywords']
            )

# Display job recommendations if available
if st.session_state.job_recommendations:
    st.markdown("---")
    st.header("💼 Top LinkedIn Jobs")

if st.session_state.job_recommendations:
    for job in st.session_state.job_recommendations:
        st.markdown(f"<div class='job-title'>{job.get('title')}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='company-name'>{job.get('companyName')}</div>", unsafe_allow_html=True)
        st.markdown(f"- 📍 {job.get('location')}")
        st.markdown(f"- 🔗 [View Job]({job.get('jobUrl')})")
        st.markdown(f"- Posted {job.get('postedTime')}")
        with st.expander("See Job Description"):
            st.markdown(job.get('description'))
        st.markdown("---")

    else:
        st.warning("No LinkedIn jobs found.")
