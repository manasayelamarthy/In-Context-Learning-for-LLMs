from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain.embeddings.base import Embeddings
import uuid
import subprocess


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
    if file_path.endswith('.md'):
        with open(file_path, "r") as file:
            context = file.read()
            chunks = []

            current_chunk = context.split("/")[0]
            for line in context.split("/"):
                if line.startswith("#"):
                    chunks.append(current_chunk)
                    current_chunk = line
                else:
                    current_chunk += f"\n{line}"

    embedding_function = SentenceTransformerEmbeddings(device="cpu")
    
    ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk)) for chunk in chunks]
    
    unique_ids = set()
    unique_chunks = []
    
    for chunk, id in zip(chunks, ids):     
        if id not in unique_ids:       
            unique_ids.add(id)
            unique_chunks.append(chunk)

    documents = [Document(page_content=chunk) for chunk in unique_chunks]

    vectorstore = Chroma.from_documents(
        documents=documents, 
        ids=list(unique_ids),
        embedding=embedding_function,
        persist_directory=output_dir
    )

    vectorstore.persist()

    retriever = vectorstore.as_retriever(search_type="similarity")

    return retriever
 

if __name__ == "__main__":
    subprocess.run(["marker_single", "../data/hyperbolic image segmentation.pdf" , "--output_dir", "../data"])