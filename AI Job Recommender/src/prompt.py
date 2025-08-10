from langchain_core.prompts import ChatPromptTemplate

PROMPT = ChatPromptTemplate.from_template(
    """
You are an expert HR recruiter and job application analyzer.
You will be given a candidate's CV text. Based on your analysis, provide the following structured output:

Instructions:
- **score**: Give a score between 1 and 10 for the overall CV quality, relevance, and clarity.
- **summary**: Write a short 3–5 line summary of the candidate’s background, skills, and experience.
- **improvement**: Suggest improvements (section wise as it will look good on streamlit gui ) in terms of:
    - Technical or soft **skills** the candidate should learn
    - Recommended **certifications** to enhance the CV
    - Missing or weak sections in the CV
    -use bullets for these improvments
- **keywords**: Extract key skills, tools, or roles from the CV that can help the candidate find jobs on platforms like LinkedIn or Rozee.pk. It should be best 3 keywords based on user cv  and should be comma separated only.Return them as a Python list of strings.

CV Text:
{doc}
"""
)