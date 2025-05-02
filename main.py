import subprocess
from main import get_input

from sentence_transformers import SentenceTransformer
model_name = "sentence-transformers/all-mpnet-base-v2"

user_input = get_input()
if user_input == "" :
    file = user_input


file_path = f"../data/{file}.pdf"
subprocess.run(["marker_single", file_path , "--output_dir", "../data"])


def convert_chunks(output_file_path): 
    if output_file_path.endswith('.md'):
        with open(output_file_path, "r") as file:
            context = file.read()
            chunks = []

            for line in context.split("#"):
                 chunks.append()

            with open("chunks.txt", "w") as text:
                for chunk in chunks:
                    text.write(chunk.strip() + "\n")

            return chunks

def convert_embeddings(text_batch):
        transformer = SentenceTransformer(model_name, device="cuda")
        embeddings = transformer.encode(
            text_batch,
            batch_size=100,  
            device="cuda",
        ).tolist()

        return  embeddings





   

