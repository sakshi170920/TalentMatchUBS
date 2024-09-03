from flask import Flask
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from numpy.linalg import norm
import pandas as pd
import numpy as np
import requests
import PyPDF2
import re
import plotly.graph_objects as go
import nltk
import heapq
nltk.download('punkt')
nltk.download('punkt_tab')
app = Flask(__name__)


@app.route("/")
def hello():
    resume = get_resume()
    input_CV = preprocess_text(resume)
    jd = get_job_description()
    # jds = ["skill","work","manner","okay","good","india","egypt","java","python","civil"]
    jds = [jd]

    # Apply to CV and JD
    N = 5
    top_N_jobs = get_top_jobs(resume=resume,jds=jds,N=N)
    for job in top_N_jobs:
        print("score : "+str(round(job[1],2)))
        print(job[0])
    return "Welcome to machine learning model APIs!"

def preprocess_text(text):
    # Convert the text to lowercase
    text = text.lower()
    
    # Remove punctuation from the text
    text = re.sub('[^a-z]', ' ', text)
    
    # Remove numerical values from the text
    text = re.sub(r'\d+', '', text)
    
    # Remove extra whitespaces
    text = ' '.join(text.split())
    return text

def get_resume() :
    resume = ""
    pdf = PyPDF2.PdfReader('./CV/Akshay_Srimatrix.pdf')
    for i in range(len(pdf.pages)):
        pageObj = pdf.pages[i]
        resume += pageObj.extract_text()
    return resume    

def get_top_jobs(resume,jds,N):
    similarity_scores = []
    for jd in jds:
        similarity_score = get_similarity_score(resume=resume,job_description=jd)
        similarity_scores.append([jd,similarity_score])
    return heapq.nlargest(N, similarity_scores, key = lambda x: x[1])
    
def get_similarity_score(resume,job_description):
    model = Doc2Vec.load('cv_job_maching.model')

    v1 = model.infer_vector(resume.split())
    v2 = model.infer_vector(job_description.split())

    similarity = 100*(np.dot(np.array(v1), np.array(v2))) / (norm(np.array(v1)) * norm(np.array(v2)))
    return similarity

def get_job_description() :
    jd = """
    India
Information Technology (IT)
Group Functions
Job Reference #

296283BR
City

Pune
Job Type

Full Time
Your role

Do you want to design and build next generation business applications using the latest technologies? Are you confident at iteratively refining user requirements and removing any ambiguity? Do you like to be challenged and encouraged to learn and grow professionally?
Your role :
Do you want to design and build next generation business applications using the latest technologies? Are you confident at iteratively refining user requirements and removing any ambiguity? Do you like to be challenged and encouraged to learn and grow professionally?

We’re looking for senior software engineer to:
provide technology solutions that will solve business problems and strengthen our position as digital leaders in financial services
as a PL/SQL Developer he will be responsible for properly maintaining source code sets and required necessary technical documentation in repository tools like GIT.
design, plan and deliver sustainable solutions using modern programming languages
conduct code reviews and test software as needed, along with participating in application architecture and design and other phases of SDLC
see that proper operational controls and procedures are implemented to process move from test to production
Your team

An Oracle PL/SQL Data Engineer will be a key member of the NCL Market reg porting responsible for database engineering deliveries for regulatory reporting. Data Engineer must have Oracle / PL SQL as primary skillset , should lead tech deliveries with team of engineers. Data Engineer should be working in POD in various capacities including working with Product owners to understand the functional requirements, agree the design with architects and deliver the technical solution.
Your expertise

Oracle PL/SQL , GIT, Unix
Microsoft Azure, PySpark, Cloud data engineering domain experience.
involve in analysis, design, development, testing, compiling, debugging, and maintaining the data using Oracle PL/SQL
strong experience in complex SQL Queries using hints, indexes Joins, DDL, DML, TCL, Types, Object, Collection Development
experience in working complex ETL / Transaction processing/ DWH applications.
experience in writing procedures, packages, functions, exceptions
experience in Analytical functions, Collections Bulk Data operation
experience in development tools such as PL/SQL developer, Toad
good exposure to Oracle Performance Tuning concepts and should be able to tune the SQL queries as well as PL/SQL programs using hints, indexes, partitions, and other performance tuning techniques.
excellent communication and analytical skills, Quick learner, looking for new challenges.
About us

UBS is the world’s largest and the only truly global wealth manager. We operate through four business divisions: Global Wealth Management, Personal & Corporate Banking, Asset Management and the Investment Bank. Our global reach and the breadth of our expertise set us apart from our competitors..

We have a presence in all major financial centers in more than 50 countries.
How we hire

We may request you to complete one or more assessments during the application process. Learn more
Join us

At UBS, we embrace flexible ways of working when the role permits. We offer different working arrangements like part-time, job-sharing and hybrid (office and home) working. Our purpose-led culture and global infrastructure help us connect, collaborate, and work together in agile ways to meet all our business needs.

From gaining new experiences in different roles to acquiring fresh knowledge and skills, we know that great work is never done alone. We know that it's our people, with their unique backgrounds, skills, experience levels and interests, who drive our ongoing success. Together we’re more than ourselves. Ready to be part of #teamUBS and make an impact?
Contact Details

UBS Business Solutions SA
UBS Recruiting
Disclaimer / Policy Statements

UBS is an Equal Opportunity Employer. We respect and seek to empower each individual and support the diverse cultures, perspectives, skills and experiences within our workforce.
    """
    return jd
if __name__ == '__main__':
    app.run(debug=True)