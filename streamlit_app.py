import os
import requests
import numpy as np
from urllib.error import HTTPError

from bs4 import BeautifulSoup
import urllib.request
from inscriptis import get_text

import langchain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.summarize import load_summarize_chain
from langchain.utilities import TextRequestsWrapper
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import LLMChain, HypotheticalDocumentEmbedder
from langchain.vectorstores import Chroma

import streamlit as st
import numpy as np

st.title('Apply to 200 jobs with personalised emails with a single click!')

openai_api_key = st.text_input("OpenAI API Key", type="password")
google_programmable_se_key = st.text_input("Google Programmable SE Key", type="password")

def get_news(topic):

    url = "https://www.googleapis.com/customsearch/v1"

    web_news = requests.get(
        url,
        params={
            "key": google_programmable_se_key,
            "cx": "6034082d68314406a", # News
            "q": f"{topic}",
            "y": 1
        }
    ).json()["items"]

    print(len(web_news))

    web_news_text = ""
    for result in web_news:
        title = result["title"]
        snippet = result["snippet"]
        tmp = f"""TITLE: {title} SNIPPET: {snippet}. """
        web_news_text += tmp

    return web_news_text

with st.form("my_form"):

    job_manager_name = st.text_input("Manager name:", "Deepak Cheenath")
    job_company = st.text_input("Company name:", "Quizizz")
    job_sector = st.text_input("Company sector:", "Education Technology")
    job_company_description = st.text_area("Company description:", "Quizizz is a education tech startup that helps teachers engage with kids in classroom better. They have more than 70m MAU, all built by a small team of people based out of Bangalore.")
    job_company_title = st.text_input("Job Title:", "Senior Software Engineer")
    job_company_website = st.text_input("Company website:", "http://www.quizizz.com")
    job_is_a_brand = 0
    job_company_news = ""

    candidate_name = st.text_input("Your name:", "Ranjeet Agarwal")
    candidate_current_role = st.text_input("Your current company role", "Software Engineer")
    candidate_current_company = st.text_input("Your current company", "Microsoft")
    candidate_current_experience = st.text_area("Your experience details", "I have been working at Microsoft since 6 years and I helped scale the back-end technology for education product at the company. Prior to this, I worked at Google where I spent 2 years working on Google search. Before that I graduated from IIT Bombay with a degree in computer science")

    email_prompt = """
    Write a personalized outreach email from {candidate_name}, working as {candidate_current_role} at {candidate_current_company} to {job_manager_name} working at {job_company} for a {job_company_title} role.
    You will be given with some information about the {candidate_name} and the {job_company}.
    
    The goal of the email is to get a POSITIVE RESPONSE from {job_manager_name}.

    Here are some information about {job_sector}:
    - Sector: {job_sector}
    - Description: {job_company_description}

    The candidate current experience as described by him is: {candidate_current_experience}

    """

    if job_is_a_brand:
        email_prompt+= """\n    Do not miss to add that one reason you are excited about the company is the brand of {job_company}."""
    else:
        email_prompt+= """\n    Do not miss to add that  one reason you are excited about the company is the opportunity and responsibility you would get in a small team."""

    email_prompt += """\n\n    In case its useful, judiciously use to show excited about {job_company} that {job_company} has been in news recently in these situations: {job_company_news}."""
    email_prompt += """
    Follow this format for the email:
    - It should have a short and creative summary.
    - The email should be roughly 100 words and it should have a FORMAL tone.
    - If any relevant news is used as a hook make sure to consider the tone of the news.
    - No need to use over the top flattery or unnecessary appreciation.
    - End the email with a call-to-action such as asking them give a suitable time to discuss more.
    """

    PROMPT = PromptTemplate(
    template=email_prompt,
    input_variables=[
        "candidate_name",
        "candidate_current_role",
        "candidate_current_company",
        "candidate_current_experience",
        "job_manager_name",
        "job_company",
        "job_company_title",
        "job_sector",
        "job_company_description",
        "job_company_news"
        ],
    )

    submitted = st.form_submit_button("Submit")
    if not (openai_api_key and google_programmable_se_key):
        st.info("Please add your keys to continue.")
    elif submitted:
        job_company_news = get_news(job_company)
	
        llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", openai_api_key=openai_api_key)

        llm_chain = LLMChain(prompt=PROMPT, llm=llm, verbose=True)
        output = llm_chain(
        {
            "candidate_name": candidate_name,
            "candidate_current_role": candidate_current_role,
            "candidate_current_company": candidate_current_company,
            "candidate_current_experience": candidate_current_experience,
            "job_manager_name": job_manager_name,
            "job_company": job_company,
            "job_company_title": job_company_title,
            "job_sector": job_sector,
            "job_company_description": job_company_description,
            "job_company_news": job_company_news
        },
        callbacks=[]
        )

        print(output["text"], sep="\n")
        st.info(output["text"])
        print("******\n\n\n\n")
        print(job_company_news)

