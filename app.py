import streamlit as st
import fitz  # PyMuPDF
from transformers import pipeline

# Load free HuggingFace pipelines
@st.cache_resource
def load_pipelines():
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    qna = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")
    generator = pipeline("text2text-generation", model="google/flan-t5-base")
    return summarizer, qna, generator

summarizer, qna, generator = load_pipelines()

# Extract text from uploaded PDF
def extract_text_from_pdf(uploaded_file):
    text = ""
    pdf = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    for page in pdf:
        text += page.get_text()
    return text

# Generate challenge questions using FLAN-T5
def generate_questions(summary_text):
    prompt = f"Generate 3 high-quality comprehension questions based on this summary:\n\n{summary_text}"
    output = generator(prompt, max_length=256, do_sample=False)[0]["generated_text"]
    # Split questions if separated by newlines or numbers
    return [q.strip(" .") for q in output.strip().split('\n') if q.strip()]

# Streamlit App UI
st.title("📚 Smart Research Assistant (No API Needed)")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    with st.spinner("Extracting text..."):
        full_text = extract_text_from_pdf(uploaded_file)

    with st.spinner("Summarizing document..."):
        summary = summarizer(full_text[:1024], max_length=150, min_length=30, do_sample=False)[0]['summary_text']
        st.subheader("📄 Summary (≤150 words):")
        st.write(summary)

    # Interaction Mode
    mode = st.radio("Choose Mode:", ["Ask Anything", "Challenge Me"])

    if mode == "Ask Anything":
        question = st.text_input("Ask your question about the document:")
        if question:
            with st.spinner("Thinking..."):
                answer = qna(question=question, context=full_text[:2000])
            st.subheader("📌 Answer:")
            st.write(answer['answer'])
            st.caption("Answer confidence score: " + str(round(answer['score'] * 100, 2)) + "%")

    elif mode == "Challenge Me":
        if st.button("💡 Generate Challenge Questions"):
            with st.spinner("Generating questions..."):
                questions = generate_questions(summary)
            if questions:
                st.subheader("🧠 Answer these questions:")
                for i, q in enumerate(questions, 1):
                    st.markdown(f"*Q{i}.* {q}")
            else:
                st.warning("Could not generate questions. Try uploading a different PDF or retry.")