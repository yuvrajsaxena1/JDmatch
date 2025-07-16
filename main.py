import streamlit as st
import os
from dotenv import load_dotenv
import openai
from PyPDF2 import PdfReader
import json
import re
import pandas as pd
from typing import Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# Load environment variables
load_dotenv()

# === OpenAI GPT Call ===
def get_openai_output(prompt):
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return response.choices[0].message.content.strip()

# === PDF Reader & Cleaner ===
def read_pdf(uploaded_file):
    pdf_reader = PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

def clean_text(text):
    text = re.sub(r'\n+', ' ', text)  # Convert multiple newlines to space
    text = re.sub(r'\s{2,}', ' ', text)  # Remove extra spaces
    text = re.sub(r'(?<!\w)\s+|\s+(?!\w)', '', text)  # Remove isolated spaces
    return text.strip()

# === JSON Extractor ===
def extract_json_block(text):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return match.group(0)
    return None

# === Prompt Generator ===
def generate_prompt(cleaned_text: str, job_description: str) -> str:
    return f"""
You are ResumeScanner, an expert in evaluating resumes for job relevance using ATS principles.

🎯 Task:
Analyze the following resume against the provided Job Description and return a JSON object ONLY in the exact format below:

{{
  "name": "Candidate's full name from the resume",
  "ats_score": Score out of 100 (integer only),
  "keywords_matched": [
    "✅ Keyword 1",
    "✅ Keyword 2",
    "... (list ALL keywords/skills from JD that are found in the resume)"
  ],
  "q&a": [
    "Q: A deep, resume-specific question about a technology the candidate has used (e.g., Laravel, Node.js, APIs, databases)",
    "Q: Ask the candidate to explain a project they’ve mentioned — including goals, their role, and challenges",
    "Q: Ask why they chose a specific technology (e.g., MySQL vs MongoDB, PHP vs Node.js) in a past project",
    "Q: Ask about how they ensured performance, security, or scalability in a past solution",
    "Q: Ask about teamwork, leadership, or decision-making in a real technical context"
  ]
}}

📋 Job Description:
{job_description.strip()}

📄 Resume:
{cleaned_text}

Return only the JSON object. Do not include explanations or commentary.
"""

# === Resume Processing Worker ===
def process_resume_worker(args: Tuple[str, str, str]) -> Tuple[Optional[dict], Optional[Tuple[str, Exception, str]]]:
    filename, cleaned_text, job_description = args
    prompt = generate_prompt(cleaned_text, job_description)
    try:
        response = get_openai_output(prompt)
        json_data = extract_json_block(response)
        candidate_info = json.loads(json_data)
        candidate_info["pdf_filename"] = filename
        return candidate_info, None
    except Exception as e:
        return None, (filename, e, locals().get('response', ''))

# === Streamlit UI ===
st.set_page_config(page_title="ResumeATS Candidate Evaluator", layout="wide")

st.markdown("""
    <h1 style='text-align: center; font-size: 42px;'>🧠 ResumeATS Candidate Evaluator</h1>
    <p style='text-align: center; font-size: 18px;'>Upload resumes, paste JD, and get smart candidate cards</p>
""", unsafe_allow_html=True)

# === JD and Upload side-by-side ===
col1, col2 = st.columns(2)
with col1:
    job_description = st.text_area("📋 Paste the Job Description (JD)")
with col2:
    uploaded_files = st.file_uploader("📄 Upload resumes (PDF)", type=["pdf"], accept_multiple_files=True)

# Cache resume content and results
if "resume_texts" not in st.session_state:
    st.session_state.resume_texts = {}
if "candidates_raw" not in st.session_state:
    st.session_state.candidates_raw = []

if st.button("🔍 Analyze Resumes"):
    if not uploaded_files:
        st.error("Please upload at least one resume.")
    elif not job_description.strip():
        st.error("Please paste a job description.")
    else:
        st.session_state.resume_texts = {}
        st.session_state.candidates_raw = []
        st.info("⏳ Processing resumes this may take a while....")

        seen_filenames = set()
        duplicate_filenames = set()
        unique_files = []
        for file in uploaded_files:
            if file.name in seen_filenames:
                duplicate_filenames.add(file.name)
                continue
            seen_filenames.add(file.name)
            unique_files.append(file)

        if duplicate_filenames:
            st.warning(f"⚠️ Duplicate resumes detected and ignored: {', '.join(duplicate_filenames)}")

        # Prepare all resumes and prompts first
        resume_data = []
        for file in unique_files:
            file.seek(0)
            filename = file.name
            raw_text = read_pdf(file)
            cleaned_text = clean_text(raw_text)
            st.session_state.resume_texts[filename] = cleaned_text
            resume_data.append((filename, cleaned_text, job_description))

        st.info(f"⏳ Sending {len(resume_data)} resumes for evaluation in parallel...")
        with ThreadPoolExecutor(max_workers=min(5, len(resume_data))) as executor:
            futures = [executor.submit(process_resume_worker, args) for args in resume_data]
            for future in as_completed(futures):
                candidate_info, error = future.result()
                if candidate_info:
                    st.session_state.candidates_raw.append(candidate_info)
                elif error:
                    filename, e, response = error
                    st.error(f"❌ Error for {filename}: {e}")
                    st.code(response)

if st.button("♻️ Re-evaluate with Updated JD"):
    if not st.session_state.resume_texts:
        st.warning("No resumes to re-evaluate. Please run analysis first.")
    elif not job_description.strip():
        st.warning("Please enter the updated Job Description.")
    else:
        st.session_state.candidates_raw = []
        resume_data = [(filename, cleaned_text, job_description) for filename, cleaned_text in st.session_state.resume_texts.items()]
        st.info(f"⏳ Re-evaluating {len(resume_data)} resumes in parallel...")
        with ThreadPoolExecutor(max_workers=min(5, len(resume_data))) as executor:
            futures = [executor.submit(process_resume_worker, args) for args in resume_data]
            for future in as_completed(futures):
                candidate_info, error = future.result()
                if candidate_info:
                    st.session_state.candidates_raw.append(candidate_info)
                elif error:
                    filename, e, response = error
                    st.error(f"❌ Error for {filename}: {e}")
                    st.code(response)

if st.session_state.candidates_raw:
    candidates = st.session_state.candidates_raw
    st.success("✅ Candidate evaluation complete!")

    filter_col, display_col = st.columns([2, 5])

    with filter_col:
        st.subheader("🔎 Filters")
        search_name = st.text_input("Candidate name contains:")
        min_score_input = st.text_input("Minimum ATS Score", value="0")
        try:
            min_score = int(min_score_input)
        except:
            min_score = 0

    filtered = [
        c for c in candidates
        if search_name.lower() in c["name"].lower() and c["ats_score"] >= min_score
    ]

    with display_col:
        if not filtered:
            st.warning("❌ Candidate not found matching your filters.")
        else:
            st.subheader("🏆 Top-N Display")
            if len(filtered) > 1:
                top_n = st.slider("Show Top N Candidates", 1, len(filtered), min(5, len(filtered)))
            else:
                top_n = 1
                st.info("Only 1 candidate matched your filters.")

            sorted_candidates = sorted(filtered, key=lambda x: x["ats_score"], reverse=True)[:top_n]

            st.subheader(f"📋 Showing Top {top_n} Candidates")
            for idx, candidate in enumerate(sorted_candidates, 1):
                with st.container():
                    st.markdown(f"### {idx}. 🧑 {candidate['name']}")
                    st.markdown(f"**ATS Score:** `{candidate['ats_score']}`")
                    st.markdown("**✅ Keywords Matched:**")
                    st.markdown("<ul>" + "".join(f"<li>{k}</li>" for k in candidate["keywords_matched"]) + "</ul>", unsafe_allow_html=True)
                    st.markdown("**🗣️ Suggested Questions:**")
                    st.markdown("<ul>" + "".join(f"<li>{q}</li>" for q in candidate["q&a"]) + "</ul>", unsafe_allow_html=True)
                    st.markdown(f"📎 **PDF Filename:** `{candidate['pdf_filename']}`")
                    st.markdown("---")
