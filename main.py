from flask import Flask, request, render_template
import os
import PyPDF2
import docx2txt
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"pdf", "docx", "txt"}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Utility functions for file handling and text processing

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def clean_text(text):
    """Basic NLP preprocessing"""
    text = text.lower()
    text = re.sub(r"\W+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


#Use PyPDF2 for PDFs, docx2txt for Word documents, and simple file reading for text files to extract text content from resumes and job descriptions.

def extract_text_pdf(file_path):
    text = ""
    with open(file_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text


def extract_text_docx(file_path):
    return docx2txt.process(file_path)


def extract_text_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def extract_text(file_path):
    if file_path.endswith(".pdf"):
        return extract_text_pdf(file_path)
    elif file_path.endswith(".docx"):
        return extract_text_docx(file_path)
    elif file_path.endswith(".txt"):
        return extract_text_txt(file_path)
    return ""


#I chex's would use this logic for resume matching but you can also explore other methods like word embeddings or transformer-based models for better accuracy

def match_resumes(job_description, resumes):
    documents = [job_description] + resumes

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(documents)

    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0]

    ranked = sorted(
        enumerate(similarity),
        key=lambda x: x[1],
        reverse=True
    )

    return ranked


#Routes you can add more routes if you want to add more features in the future

@app.route("/")
def home():
    return render_template("app.html")


@app.route("/upload", methods=["POST"])
def upload():
    jd = request.form.get("resumeText", "")
    resume_files = request.files.getlist("resumeFile")

    if not jd and not resume_files:
        return render_template(
            "app.html",
            message="Upload resumes or enter job description"
        )

    jd = clean_text(jd)

    resumes = []
    filenames = []

    for file in resume_files:

        if file and allowed_file(file.filename):

            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)

            text = extract_text(filepath)
            text = clean_text(text)

            resumes.append(text)
            filenames.append(file.filename)

    if not resumes:
        return render_template(
            "app.html",
            message="No valid resumes uploaded"
        )

    ranked = match_resumes(jd, resumes)

    top_results = []

    for index, score in ranked[:3]:
        top_results.append({
            "name": filenames[index],
            "score": round(score * 100, 2)
        })

    return render_template(
        "app.html",
        results=top_results,
        message="Top Matching Resumes"
    )




if __name__ == "__main__":

    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)

    app.run(debug=True)