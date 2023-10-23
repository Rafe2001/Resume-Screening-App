from flask import Flask, render_template, request
import joblib
import re
import nltk
import PyPDF2

nltk.download("punkt")
nltk.download("stopwords")

# Load the TF-IDF vectorizer and the trained classifier model
tfidf = joblib.load("model/tfidf.joblib")
model = joblib.load("model/clf.joblib")

def preprocess_text(text):
    text = re.sub(r"http\S+\s", " ", text)
    text = re.sub(r"RT|cc", " ", text)
    text = re.sub(r"@\S+", " ", text)
    text = re.sub(r"#\S+\s", " ", text)
    text = re.sub(r'[%s]' % re.escape('''|"#$%'()"+,-./:;<=>?@[\]^_'{|}~*'''), " ", text)
    text = re.sub(r"[^\x00-\x7f]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text

# Function to convert PDF to text
def pdf_to_text(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    num_pages = len(pdf_reader.pages)
    text = ""
    for page_num in range(num_pages):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()
    return text

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/classify", methods=["POST"])
def classify():
    if request.method == "POST":
        try:
            uploaded_file = request.files["file"]
            if uploaded_file.filename != "":
                # Check the file type and convert to text if it's a PDF
                if uploaded_file.filename.endswith(".pdf"):
                    resume_text = pdf_to_text(uploaded_file)
                else:
                    resume_text = uploaded_file.read().decode("utf-8")
                
                cleaned_resume = preprocess_text(resume_text)
                cleaned_resume = tfidf.transform([cleaned_resume])
                prediction_id = model.predict(cleaned_resume)[0]

                categories = {
            6: 'Data Science',
            12: 'HR',
            0: 'Advocate',
            1: 'Arts',
            24: 'Web Designing',
            16: 'Mechanical Engineer',
            22: 'Sales',
            14: 'Health and Fitness',
            5: 'Civil Engineer',
            15: 'Java Developer',
            4: 'Business Analyst',
            21: 'SAP Developer',
            2: 'Automation Testing',
            11: 'Electrical Engineering',
            18: 'Operations Manager',
            20: 'Python Developer',
            8: 'DevOps Engineer',
            17: 'Network Security Engineer',
            19: 'PMO',
            7: 'Database',
            13: 'Hadoop',
            10: 'ETL Developer',
            9: 'DotNet Developer',
            3: 'Blockchain',
            23: 'Testing',
                }
                
                category_name = categories.get(prediction_id, "Unknown")
                return render_template("result.html", category=category_name)
            else:
                return render_template("index.html", error="No file selected.")
        except UnicodeDecodeError:
            return render_template("index.html", error="Error processing the file.")
    else:
        return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
