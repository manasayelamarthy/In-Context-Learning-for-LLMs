import sys
import torch

torch.classes = None

import streamlit as st
import os
import tempfile
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from data_extraction import doc_retrevier
from dotenv import load_dotenv
load_dotenv()

from streamlit_webrtc import webrtc_streamer, AudioProcessorBase
import speech_recognition as sr
import av
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered, save_output

st.set_page_config(page_title="Multi-Docs ChatBot", layout="wide")
st.title("Multi-Docs ChatBot using LLaMA2")

# Sidebar upload
st.sidebar.header("Document Processing")
uploaded_file = st.sidebar.file_uploader("Upload a document", type=["md", "pdf", "txt"])

# LLM setup
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# Prompt template
PROMPT_TEMPLATE = """
You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer
the question. If you don't know the answer, say that you
don't know. DON'T MAKE UP ANYTHING.

{context}

---

Answer the question based on the above context: {question}
"""

# Process uploaded file
retriever = None

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}", mode='wb') as tmp_file:
        tmp_file.write(uploaded_file.read())
        file_path = tmp_file.name

    output_dir = os.path.join(tempfile.gettempdir(), "vector_store")

    with st.spinner("Processing document..."):
        # Check the file extension and handle accordingly
        if file_path.endswith(".pdf"):
            try:
                converter = PdfConverter(
                    artifact_dict=create_model_dict(),
                )
                rendered = converter(file_path)  # Use the actual file path
                save_output(rendered, output_dir=output_dir, fname_base=os.path.splitext(os.path.basename(file_path))[0])
                
                retriever = doc_retrevier(file_path, output_dir)
                st.sidebar.success("PDF processed and indexed.")

            except Exception as e:
                st.error(f"Error processing PDF: {e}")


        else:
            # If not PDF, use doc_retrevier as usual
            retriever = doc_retrevier(file_path, output_dir)
            st.sidebar.success("Document processed and indexed.")

# Voice input processor
class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.text_output = ""

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        audio = frame.to_ndarray().flatten().astype("float32").tobytes()
        with sr.AudioFile(sr.io.BytesIO(audio)) as source:
            try:
                audio_data = self.recognizer.record(source)
                text = self.recognizer.recognize_google(audio_data)
                self.text_output = text
            except sr.UnknownValueError:
                self.text_output = ""
        return frame

# Voice input interface
st.markdown("### Voice Input (optional)")
voice_ctx = webrtc_streamer(key="voice", audio_processor_factory=AudioProcessor, media_stream_constraints={"audio": True, "video": False})
user_input = ""

if voice_ctx.audio_processor:
    user_input = voice_ctx.audio_processor.text_output
    if user_input:
        st.info(f"Recognized: {user_input}")

# Text input fallback
user_text = st.text_input("Or type your question here:")
if user_text:
    user_input = user_text

# Ask question
if user_input and retriever:
    with st.spinner("Generating answer..."):
        relevant_chunks = retriever.invoke(user_input)
        context_text = "\n\n---\n\n".join([doc.page_content for doc in relevant_chunks])

        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=user_input)

        response = llm.invoke(prompt)
        st.write("### Response:")
        st.write(response.content)
elif not retriever:
    st.info("Please upload a document to start.")
