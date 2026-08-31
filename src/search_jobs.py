import chromadb
from sentence_transformers import SentenceTransformer


def create_collection():
    client = chromadb.PersistentClient(path="chroma_db")

    collection = client.get_or_create_collection(
        name="jobs"
    )

    return collection


def search_jobs(collection, model, resume_text, top_k=5):

    resume_embedding = model.encode(resume_text)

    results = collection.query(
        query_embeddings=[resume_embedding.tolist()],
        n_results=top_k
    )

    return results


if __name__ == "__main__":

    model = SentenceTransformer("all-MiniLM-L6-v2")

    collection = create_collection()

    resume_text = """
    Python developer with experience in Python, SQL,
    Django and REST APIs. Interested in backend development
    and machine learning.
    """

    results = search_jobs(
        collection,
        model,
        resume_text,
        top_k=5
    )

    print("\nTop matching jobs:")

    for i in range(5):
        print("\nRank:", i + 1)
        print("Job:", results["metadatas"][0][i]["job_title"])
        print("Company:", results["metadatas"][0][i]["company"])
        print("Location:", results["metadatas"][0][i]["location"])
        print("Experience:", results["metadatas"][0][i]["experience"])
        print("Distance:", results["distances"][0][i])