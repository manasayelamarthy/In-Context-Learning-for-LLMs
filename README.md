# In-Context-Learning LLMs

![User Flow Diagram](LLMs-and-In-Context-learning.jpg)

# Multi-Docs ChatBot using Gemini 2.0 Flash (RAG)

## About the Project
This project is a Retrieval-Augmented Generation (RAG) application built using LangChain, Streamlit, and Google's Gemini 2.0 Flash model. It allows users to upload a document (PDF, Markdown, or TXT), extracts its content, and provides intelligent, context-based answers using Gemini LLM. It supports both text and voice input for seamless interaction.

## Deployed Application
The app is deployed with Streamlit and accessible at:

👉 [**Live Demo**](https://your-deployment-link.com)  
*(Replace this link with your deployed URL)*

### Screenshot of the Webpage
![Webpage Screenshot](your-app-screenshot.png)  
*(Replace with actual screenshot file if added to repo)*

## Dataset Details
There is no preloaded dataset in this project.

- Users upload their own documents at runtime.
- Supported file types: `.pdf`, `.md`, `.txt`
- Documents are converted (PDF to Markdown when necessary), embedded using vector stores, and queried via LangChain retrievers.

## Model Details
The project uses the following core components:

- **LLM:** Gemini 2.0 Flash (via `langchain_google_genai`)
- **Prompt Format:**
  ```text
  You are an assistant for question-answering tasks.
  Use the following pieces of retrieved context to answer
  the question. If you don't know the answer, say that you
  don't know. DON'T MAKE UP ANYTHING.

  {context}

  ---

  Answer the question based on the above context: {question}
Voice Input: Streamlit WebRTC and speech_recognition

PDF to Markdown Conversion: Marker library

Vector Embedding + Retrieval: LangChain

Deployment Details
Frontend & Backend: Streamlit

Document Processing: Marker (for PDF conversion)

Voice Input: Streamlit WebRTC + Google SpeechRecognition

RAG Pipeline: LangChain retriever + Gemini Flash

Instructions to Reproduce
1. Clone the repository
bash
Copy
Edit
git clone https://github.com/manasayelamarthy/In-Context-Learning-for-LLMs
cd In-Context-Learning-for-LLMs
2. Set up a virtual environment
bash
Copy
Edit
python -m venv venv
source venv/bin/activate  # On Windows, use venv\Scripts\activate
3. Install dependencies
bash
Copy
Edit
pip install -r requirements.txt
4. Configure your API key
Create a .env file in the root directory and add:

env
Copy
Edit
GOOGLE_API_KEY=your_google_api_key_here
5. Run the application
bash
Copy
Edit
streamlit run app.py
Features
Upload and query any document using natural language

Works with PDFs, Markdown, and plain text

Voice-to-text support for question input

Handles document chunking, vectorization, and retrieval automatically

Limitations
Only one document at a time is processed per session

Requires internet access for Gemini API and voice recognition

Voice recognition accuracy may vary with background noise

License
This project is for educational/research purposes.

Acknowledgments
LangChain

Google Generative AI (Gemini)

Streamlit
