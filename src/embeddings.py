import pandas as pd
from sentence_transformers import SentenceTransformer


def load_jobs(file_path):
    jobs = pd.read_csv(file_path)
    return jobs


def create_embeddings(jobs, model):
    job_texts = jobs["combined_text"].tolist()
    embeddings = model.encode(job_texts)
    return embeddings


if __name__ == "__main__":
    model = SentenceTransformer("all-MiniLM-L6-v2")

    jobs = load_jobs("data/processed/cleaned_jobs.csv")

    embeddings = create_embeddings(jobs, model)

    print("Number of jobs:", len(jobs))
    print("Embedding shape:", embeddings.shape)