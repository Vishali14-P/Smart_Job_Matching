import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer


# -----------------------------
# Load Jobs
# -----------------------------

def load_jobs(file_path):
    return pd.read_csv(file_path)


# -----------------------------
# Create Embeddings
# -----------------------------

def create_embeddings(jobs, model):

    job_texts = jobs["combined_text"].tolist()

    return model.encode(job_texts)


# -----------------------------
# Create ChromaDB Collection
# -----------------------------

def create_collection():

    client = chromadb.PersistentClient(
        path="chroma_db"
    )

    collection = client.get_or_create_collection(
        name="jobs"
    )

    return collection


# -----------------------------
# Add Jobs to ChromaDB
# -----------------------------

def add_jobs_to_collection(
    collection,
    jobs,
    embeddings
):

    ids = jobs[
        "job_id"
    ].astype(str).tolist()

    documents = jobs[
        "combined_text"
    ].tolist()

    metadatas = jobs[
        [
            "job_title",
            "company",
            "location",
            "experience",
            "skills"
        ]
    ].to_dict(
        orient="records"
    )


    # ChromaDB has a maximum batch size.
    # We split 7,277 jobs into smaller batches.

    batch_size = 5000


    for start in range(
        0,
        len(ids),
        batch_size
    ):

        end = start + batch_size


        collection.add(

            ids=ids[start:end],

            embeddings=embeddings[
                start:end
            ].tolist(),

            documents=documents[
                start:end
            ],

            metadatas=metadatas[
                start:end
            ]

        )


        print(
            f"Added jobs {start + 1} "
            f"to {min(end, len(ids))}"
        )


# -----------------------------
# Main Program
# -----------------------------

if __name__ == "__main__":


    # Load AI model

    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


    # Load cleaned jobs

    jobs = load_jobs(
        "data/processed/cleaned_jobs.csv"
    )


    # Create job embeddings

    embeddings = create_embeddings(
        jobs,
        model
    )


    # Create ChromaDB collection

    collection = create_collection()


    print(
        "Number of jobs:",
        len(jobs)
    )

    print(
        "Embedding shape:",
        embeddings.shape
    )

    print(
        "Collection name:",
        collection.name
    )


    # Add jobs in batches

    add_jobs_to_collection(
        collection,
        jobs,
        embeddings
    )


    print(
        "Jobs added to Chroma successfully."
    )


    print(
        "Number of records in Chroma:",
        collection.count()
    )