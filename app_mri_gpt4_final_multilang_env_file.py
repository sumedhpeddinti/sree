import io
import os
import datetime
import streamlit as st
from PIL import Image
from matplotlib import pyplot as plt
from fpdf import FPDF
import openai
from dotenv import load_dotenv

# ---------------- Load API Key from .env or Streamlit secrets ----------------
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", "")
if not openai.api_key:
    st.error("OpenAI API key not found. Please set OPENAI_API_KEY in .env locally or in Streamlit secrets.")
    st.stop()

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="GenAI Medical Assistant for Cancer Prediction", layout="wide")
st.title("🩺 GenAI Medical Assistant for Cancer Prediction")

st.markdown("""
Upload MRI images and the doctor's report (TXT file).
Enter patient info for personalized analysis.
Select patient symptoms and preferred output language.
You can download a PDF summary for submission.
""")

# ---------------- Inputs ----------------
with st.expander("Step 1: Upload MRI and Doctor Report"):
    uploaded_files = st.file_uploader(
        "Upload MRI image(s) (PNG/JPG)", type=["png", "jpg", "jpeg"], accept_multiple_files=True
    )
    
    report_file = st.file_uploader(
        "Upload Doctor's Report (TXT file)", type=["txt"]
    )
    
    report_text = ""
    if report_file is not None:
        report_text = report_file.read().decode("utf-8")
        st.text_area("Doctor's Report Preview", report_text, height=180)

with st.expander("Step 2: Enter Patient Info"):
    patient_age = st.number_input("Patient Age", min_value=0, max_value=120, value=30)
    patient_gender = st.selectbox("Patient Gender", ["Male", "Female", "Other"])

with st.expander("Step 3: Select Patient Symptoms"):
    common_symptoms = [
        "Headache", "Seizures", "Memory Loss", "Vision Problems", "Nausea", "Dizziness",
        "Weakness in Limbs", "Speech Difficulty", "Balance Issues", "Confusion",
        "Hearing Loss", "Facial Weakness", "Personality Changes", "Severe Fatigue",
        "Unexplained Weight Loss", "Persistent Pain", "Night Sweats",
        "Vomiting (Morning)", "Difficulty Walking", "Neck Pain",
        "Numbness or Tingling", "Loss of Coordination", "Difficulty Concentrating",
        "Mood Swings", "Loss of Appetite",
        "Back Pain", "Limb Numbness", "Difficulty Swallowing",
        "Hormonal Imbalance", "Irregular Menstrual Cycle", "Urinary Problems"
    ]
    selected_symptoms = st.multiselect("Select Symptoms", common_symptoms)

target_language = st.selectbox("Select Language for Output", ["English", "Hindi", "Spanish", "French"])
run_button = st.button("Generate Diagnosis")

# ---------------- Helper Functions ----------------
def dummy_mri_description(images):
    interpretations = []
    for idx, img in enumerate(images):
        width, height = img.size
        prompt = (
            f"You are a highly experienced radiologist and cancer specialist reviewing an MRI scan "
            f"(Image {idx+1}, {width}x{height} pixels). "
            "Provide a professional-style radiology report summary in 5-6 sentences, "
            "describing any abnormal areas, lesions, masses, or other notable features you observe and also give is there any chances of cancer or not if yes tell which cancer. "
            "Avoid giving real medical diagnosis or treatment; focus purely on describing MRI findings."
        )
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an experienced radiologist and cancer specialist."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        interpretations.append(response.choices[0].message.content.strip())
    return "\n\n".join(interpretations)


def compute_risk_score(report, mri_text):
    text = (report + " " + mri_text).lower()

    high_risk_keywords = [
        "malignant", "malignancy", "metastasis", "metastatic", "carcinoma",
        "glioblastoma", "astrocytoma", "invasive", "necrotic core",
        "heterogeneous enhancement", "irregular mass", "rim-enhancing lesion",
        "diffusion restriction", "hypercellularity", "neoplasm",
        "abnormal enhancement pattern"
    ]

    medium_risk_keywords = [
        "lesion", "mass", "cystic", "nodule", "abnormal signal",
        "focal area", "hyperintensity", "hypointensity",
        "calcification", "swelling", "inflammation",
        "midline shift", "compression", "edema"
    ]

    low_risk_keywords = [
        "mild", "small", "minimal enhancement", "benign", "no definite mass",
        "scattered changes", "non-specific", "normal variant"
    ]

    score = 0
    for kw in high_risk_keywords:
        if kw in text:
            score += 3
    for kw in medium_risk_keywords:
        if kw in text:
            score += 2
    for kw in low_risk_keywords:
        if kw in text:
            score += 1

    if score >= 10:
        return "High Risk"
    elif score >= 4:
        return "Medium Risk"
    else:
        return "Low Risk"


def init_session():
    if "history" not in st.session_state:
        st.session_state.history = []
    if "qa_context" not in st.session_state:
        st.session_state.qa_context = ""
    if "treatment_advice" not in st.session_state:
        st.session_state.treatment_advice = ""


def generate_pdf(filename, images, patient_info, diagnosis, risk, treatment):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "MRI + Doctor Report Diagnosis", ln=True, align="C")
    pdf.set_font("Arial", '', 12)
    pdf.ln(5)
    pdf.cell(0, 10, f"Patient Age: {patient_info['age']}, Gender: {patient_info['gender']}", ln=True)
    pdf.cell(0, 10, f"Symptoms: {', '.join(patient_info['symptoms'])}", ln=True)
    pdf.ln(5)
    pdf.multi_cell(0, 8, f"Diagnosis:\n{diagnosis}")
    pdf.cell(0, 10, f"Risk Level: {risk}", ln=True)
    pdf.ln(5)
    pdf.multi_cell(0, 8, f"Suggested Treatment / Referral:\n{treatment}")
    pdf.ln(5)
    for img in images:
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        pdf.image(img_bytes, w=80)
        pdf.ln(5)
    pdf.output(filename)


def translate_text(text, language):
    if language == "English":
        return text
    prompt = f"Translate the following medical text into {language}, maintaining professional medical terminology:\n{text}"
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a professional medical translator."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    return response.choices[0].message.content.strip()

# ---------------- Main Flow ----------------
init_session()

if run_button:
    if not uploaded_files:
        st.error("Please upload at least one MRI image.")
        st.stop()
    if not report_text.strip():
        st.error("Please upload the doctor's report (TXT file).")
        st.stop()

    images = [Image.open(io.BytesIO(f.read())).convert("RGB") for f in uploaded_files]
    mri_text = dummy_mri_description(images)

    st.subheader("🔹 MRI Previews")
    cols = st.columns(len(images))
    for i, img in enumerate(images):
        cols[i].image(img, use_column_width=True, caption=f"MRI {i+1}")

    combined_context = (
        f"MRI Description:\n{mri_text}\n\n"
        f"Doctor Report:\n{report_text}\n\n"
        f"Patient Info:\n- Age: {patient_age}\n- Gender: {patient_gender}\n- Symptoms: {', '.join(selected_symptoms)}"
    )

    prompt = (
        "You are a medical assistant. Read the following context (MRI + doctor's report + patient info) "
        "and generate a unified diagnosis with reasoning:\n\n"
        f"{combined_context}\n\n"
        "Provide:\n- Unified Diagnosis (one line)\n- Short Explanation (2-3 sentences)\n- Key Findings (bullet points)\n- Suggested Next Steps"
    )

    with st.spinner("Generating combined diagnosis..."):
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful medical assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
    output = response.choices[0].message.content.strip()
    output_translated = translate_text(output, target_language)

    st.subheader(f"📌 Combined Diagnosis ({target_language})")
    st.write(output_translated)

    risk = compute_risk_score(report_text, mri_text)
    st.subheader("⚠️ Risk Assessment")
    st.info(f"Risk Level: *{risk}*")

    st.session_state.qa_context = combined_context + "\n\nPrevious Diagnosis:\n" + output
    st.session_state.history.append({
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "diagnosis": output_translated,
        "risk": risk
    })

    st.success("Diagnosis generated successfully.")

    st.subheader("🩺 Suggested Treatment / Referral Advice")
    with st.expander("View Suggested Treatments / Referrals"):
        treatment_prompt = (
            f"Provide suggested next steps, treatment options, or referral advice based on:\n\n"
            f"{st.session_state.qa_context}\n\n"
            "Answer concisely in bullet points."
        )
        with st.spinner("Generating suggestions..."):
            treatment_response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful medical assistant."},
                    {"role": "user", "content": treatment_prompt}
                ],
                temperature=0
            )
        treatment_text = treatment_response.choices[0].message.content.strip()
        treatment_translated = translate_text(treatment_text, target_language)
        st.write(treatment_translated)
        st.session_state.treatment_advice = treatment_translated

    st.subheader("📊 Risk Level History Chart")
    if st.session_state.history:
        counts = {"High Risk":0, "Medium Risk":0, "Low Risk":0}
        for entry in st.session_state.history:
            counts[entry['risk']] += 1
        fig, ax = plt.subplots()
        ax.bar(counts.keys(), counts.values(), color=['red','orange','green'])
        ax.set_ylabel("Number of Diagnoses")
        ax.set_title("Risk Level Distribution")
        st.pyplot(fig)

    st.subheader("📄 Export PDF Summary")
    if st.button("Generate PDF"):
        filename = "diagnosis_summary.pdf"
        patient_info = {"age": patient_age, "gender": patient_gender, "symptoms": selected_symptoms}
        generate_pdf(filename, images, patient_info, output_translated, risk, st.session_state.treatment_advice)
        with open(filename, "rb") as f:
            st.download_button("📥 Download PDF", f, file_name=filename, mime="application/pdf")

st.subheader("💬 Ask Follow-Up Questions")
question = st.text_input("Ask a question regarding the diagnosis")
ask_button = st.button("Ask")

if ask_button:
    if not question.strip():
        st.error("Please type a question.")
        st.stop()
    if not st.session_state.qa_context:
        st.error("Generate a diagnosis first.")
        st.stop()

    qa_prompt = (
        f"Answer the following question based on this context:\n\n"
        f"{st.session_state.qa_context}\n\n"
        f"Question: {question}\nAnswer concisely and accurately."
    )

    with st.spinner("Generating answer..."):
        qa_response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful medical assistant."},
                {"role": "user", "content": qa_prompt}
            ],
            temperature=0
        )
    answer_translated = translate_text(qa_response.choices[0].message.content.strip(), target_language)
    st.subheader(f"📝 Answer ({target_language})")
    st.write(answer_translated)

st.subheader("📂 Diagnosis History")
for entry in reversed(st.session_state.history):
    st.markdown(f"*{entry['timestamp']}* - Risk: {entry['risk']}")
    st.write(entry['diagnosis'])
