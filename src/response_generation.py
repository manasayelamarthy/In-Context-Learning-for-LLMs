from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from data_extraction import doc_retrevier

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash"
)


def generate_responce(user_input, llm, retreiver):
    relevant_chunks = retreiver.invoke("user_input")

    PROMPT_TEMPLATE = """
    You are an assistant for question-answering tasks.
    Use the following pieces of retrieved context to answer
    the question. If you don't know the answer, say that you
    don't know. DON'T MAKE UP ANYTHING.

    {context}

    ---

    Answer the question based on the above context: {question}
    """


    context_text = "\n\n---\n\n".join([doc.page_content for doc in relevant_chunks])

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, 
                                question=user_input)
    
    response = llm.invoke(prompt)

    return response.content

if __name__== "__main__":
    retrevier = doc_retrevier("../data/hyperbolic image segmentation/hyperbolic image segmentation.md", '../data/vector_base')
    response = generate_responce("what is the title of the document", llm, retrevier)
    print(response)