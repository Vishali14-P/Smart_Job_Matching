import chromadb
from sentence_transformers import SentenceTransformer
from ranking import calculate_similarity, calculate_final_scores
from skill_matching import calculate_skill_match


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
        n_results=top_k,
        include=["metadatas", "distances", "embeddings"]
    )

    return results, resume_embedding


if __name__ == "__main__":

    model = SentenceTransformer("all-MiniLM-L6-v2")

    collection = create_collection()

    resume_text = """
    Python developer with experience in Python, SQL,
    Django and REST API. Interested in backend development
    and machine learning.
    """

    resume_skills = "Python, SQL, Django, REST API, Machine Learning"

    results, resume_embedding = search_jobs(
        collection,
        model,
        resume_text,
        top_k=5
    )

    job_embeddings = results["embeddings"][0]

    similarities = calculate_similarity(
        resume_embedding,
        job_embeddings
    )

    skill_scores = []

    for metadata in results["metadatas"][0]:

        job_skills = metadata["skills"]

        skill_score, matched_skills = calculate_skill_match(
            resume_skills,
            job_skills
        )

        skill_scores.append(skill_score)

    final_scores = calculate_final_scores(
        similarities,
        skill_scores
    )

    print("\nTop matching jobs:")

ranked_jobs = []

for i in range(5):

    ranked_jobs.append({
        "metadata": results["metadatas"][0][i],
        "similarity": similarities[i],
        "skill_score": skill_scores[i],
        "final_score": final_scores[i]
    })


ranked_jobs.sort(
    key=lambda x: x["final_score"],
    reverse=True
)


for i, job in enumerate(ranked_jobs):

    print("\nRank:", i + 1)
    print("Job:", job["metadata"]["job_title"])
    print("Company:", job["metadata"]["company"])
    print("Location:", job["metadata"]["location"])
    print("Experience:", job["metadata"]["experience"])

    print(
        "Semantic Similarity:",
        round(float(job["similarity"]) * 100, 2),
        "%"
    )

    print(
        "Skill Match:",
        round(float(job["skill_score"]) * 100, 2),
        "%"
    )

    print(
        "Final Match Score:",
        round(float(job["final_score"]) * 100, 2),
        "%"
    )