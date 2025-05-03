from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain.embeddings.base import Embeddings
import uuid
import subprocess
import os
import logging

logging.basicConfig(level=logging.INFO)

class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2", device: str = "cpu"):
        self.transformer = SentenceTransformer(model_name, device=device)
        
    def embed_documents(self, texts: list[str]) -> list[list[float]]: 
        embeddings = self.transformer.encode(
            texts,
            batch_size=100,
            device="cpu",
            show_progress_bar=False
        )
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> list[float]:
        embedding = self.transformer.encode(
            text,
            batch_size=1,
            device="cpu",
            show_progress_bar=False
        )
        return embedding.tolist()

def doc_retrevier(file_path, output_dir):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        logging.info(f"Created output directory: {output_dir}")

    if file_path.endswith('.md'):
        try:
            with open(file_path, "r", encoding='utf-8') as file:
                context = file.read()
                chunks = []

                current_chunk = context.split("/")[0]
                for line in context.split("/"):
                    if line.startswith("#"):
                        chunks.append(current_chunk)
                        current_chunk = line
                    else:
                        current_chunk += f"\n{line}"
        except Exception as e:
            logging.error(f"Error reading file {file_path}: {str(e)}")
            raise

    embedding_function = SentenceTransformerEmbeddings(device="cpu")
    
    if not chunks:
        logging.warning("No content chunks were created")
        return None

    ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk)) for chunk in chunks]
    
    unique_ids = set()
    unique_chunks = []
    
    for chunk, id in zip(chunks, ids):     
        if id not in unique_ids:       
            unique_ids.add(id)
            unique_chunks.append(chunk)

    documents = [Document(page_content=chunk) for chunk in unique_chunks]

    try:
        vectorstore = Chroma.from_documents(
            documents=documents, 
            ids=list(unique_ids),
            embedding=embedding_function,
            persist_directory=output_dir
        )

        vectorstore.persist()
        retriever = vectorstore.as_retriever(search_type="similarity")
        return retriever
    except Exception as e:
        logging.error(f"Error creating vector store: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        input_pdf = os.path.join("..", "data", "hyperbolic image segmentation.pdf")
        output_dir = os.path.join("..", "data")
        subprocess.run(["marker_single", input_pdf, "--output_dir", output_dir], check=True)
        logging.info("PDF processing completed successfully")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error processing PDF: {str(e)}")
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")