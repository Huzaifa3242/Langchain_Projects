from langchain_community.document_loaders import PyPDFLoader
from apify_client import ApifyClient
import os 
import tempfile
api_token = os.getenv("APIFY_API_TOKEN")
client = ApifyClient(api_token)

# def load_cv(path):
#     loader = PyPDFLoader(path)
#     doc = loader.load()
#     return doc[0].page_content
import tempfile
from langchain_community.document_loaders import PyPDFLoader

def load_cv(uploaded_file):
    if uploaded_file is not None:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name
        
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        return docs
    return None

def get_llm_resp(llm,model,prompt,doc):
    structured_llm = llm.with_structured_output(model)
    chain = prompt | structured_llm
    response = chain.invoke({"doc":doc})
    return response
        

def fetch_linkedin_jobs(search_query,location = "Pakistan",rows =3):
    input={
        "title":search_query,
        "location":location,
        "rows":rows,
        "proxy":{
            "useApifyProxy":True,
            "ApifyProxyGroups":["Residential"]
        }

    }
    run = client.actor("bebity/linkedin-jobs-scraper").call(run_input = input)
    jobs = list(client.dataset(run["defaultDatasetId"]).iterate_items())
    return jobs

def get_all_jobs_list(keywords):
    all_jobs = []
    for kw in keywords:
        jobs = fetch_linkedin_jobs(kw) 
        all_jobs.extend(jobs)
    return all_jobs


