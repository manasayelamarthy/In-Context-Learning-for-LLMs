# In-Context-Learning LLMs

## About the Project
This project is a Retrieval-Augmented Generation (RAG) application that allows users to upload documents (PDF, Markdown, or TXT), extracts the content, and answers user queries based on the uploaded document. It uses **Gemini 2.0 Flash** as the language model, powered by **LangChain**, and the interface is built using **Streamlit**. Voice input is also supported using WebRTC and Google Speech Recognition.


![User Flow Diagram](LLMs-and-In-Context-learning.jpg)

## Deployed Application
The frontend and backend are built using **Streamlit** and deployed at:

It was a local host link 

👉 [**Live Demo**](https://localhost:8501/) 


## Dataset Details
There is no fixed dataset used in this project. Instead:
- Users upload their own documents at runtime.
- Supported file types:
  - `.pdf`
  - `.md` (Markdown)
  - `.txt`
- PDF files are converted to Markdown using the **Marker** library.
- Uploaded content is embedded using **LangChain's** vector store and used for context retrieval during question answering.

## Model Details
The application uses the following architecture and tools:

- **LLM:** [Gemini 2.0 Flash](https://ai.google.dev/)
  - Accessed via `langchain_google_genai`
- **RAG Pipeline:** LangChain retriever + Chat prompt + Gemini model
- **Voice Input:** `streamlit_webrtc` and `speech_recognition`
- **PDF to Markdown Conversion:** Marker library
- **Prompt Template:**
  ```text
  You are an assistant for question-answering tasks.
  Use the following pieces of retrieved context to answer
  the question. If you don't know the answer, say that you
  don't know. DON'T MAKE UP ANYTHING.

  {context}

  ---

  Answer the question based on the above context: {question}


## Deployment Details

The application components are as follows:

- **Frontend & Backend:** Streamlit  
- **Document Embedding & Retrieval:** LangChain  
- **Language Model:** Gemini 2.0 Flash (Google Generative AI)  
- **Document Conversion:** Marker (for PDF to Markdown)  
- **Voice Recognition:** Streamlit WebRTC + Google SpeechRecognition  

---

## Instructions to Reproduce

### 1. Clone the Repository
```sh
   git clone https://github.com/manasayelamarthy/In-Context-Learning-for-LLMs
   cd In-Context-Learning-for-LLMs
   ```

## 2. Set Up a Virtual Environment
``` sh
     python -m venv venv
     source venv/bin/activate  # On Windows, use venv\Scripts\activate

```
##3. Install Dependencies
```sh
   pip install -r requirements.txt
```
##4. Configure Your API Key
Create a .env file in the root directory and add:
```
   GOOGLE_API_KEY=your_google_api_key_here
```
## 5. Run the Application
```
   streamlit run app.py
```
## License
This project is intended for educational and research purposes.

## Acknowledgments
LangChain

Google Generative AI (Gemini)

Streamlit

