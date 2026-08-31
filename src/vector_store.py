import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer


def load_jobs(file_path):
    return pd.read_csv(file_path)


def create_embeddings(jobs, model):
    job_texts = jobs["combined_text"].tolist()
    return model.encode(job_texts)


def create_collection():
    client = chromadb.PersistentClient(path="chroma_db")

    collection = client.get_or_create_collection(
        name="jobs"
    )

    return collection


def add_jobs_to_collection(collection, jobs, embeddings):
    ids = jobs["job_id"].astype(str).tolist()

    documents = jobs["combined_text"].tolist()

    metadatas = jobs[
        ["job_title", "company", "location", "experience"]
    ].to_dict(orient="records")

    collection.add(
        ids=ids,
        embeddings=embeddings.tolist(),
        documents=documents,
        metadatas=metadatas
    )


if __name__ == "__main__":

    model = SentenceTransformer("all-MiniLM-L6-v2")

    jobs = load_jobs(
        "data/processed/cleaned_jobs.csv"
    )

    embeddings = create_embeddings(
        jobs,
        model
    )

    collection = create_collection()

    print("Number of jobs:", len(jobs))
    print("Embedding shape:", embeddings.shape)
    print("Collection name:", collection.name)

    add_jobs_to_collection(
        collection,
        jobs,
        embeddings
    )

    print("Jobs added to Chroma successfully.")

    print(
        "Number of records in Chroma:",
        collection.count()
    )