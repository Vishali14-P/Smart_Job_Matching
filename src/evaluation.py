import re
import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from skill_matching import calculate_skill_match


# ---------------------------------------
# Create / Load ChromaDB
# ---------------------------------------
def create_collection():
    client = chromadb.PersistentClient(path="chroma_db")
    collection = client.get_or_create_collection(name="jobs")
    return collection


# ---------------------------------------
# Experience Matching
# ---------------------------------------
def calculate_experience_match(resume_text, job_experience):

    resume_text = resume_text.lower()
    job_experience = str(job_experience).lower()

    fresher_keywords = [
        "student",
        "fresher",
        "final year",
        "undergraduate",
        "b.tech",
        "btech"
    ]

    is_fresher = any(
        keyword in resume_text
        for keyword in fresher_keywords
    )

    if is_fresher:

        if "fresher" in job_experience:
            return 1.0

        numbers = re.findall(
            r"\d+(?:\.\d+)?",
            job_experience
        )

        if numbers:

            minimum_experience = float(numbers[0])

            if minimum_experience == 0:
                return 1.0

            elif minimum_experience == 1:
                return 0.6

            else:
                return 0.0

        return 0.5

    return 0.5


# ---------------------------------------
# Relevant Job Keywords
# ---------------------------------------
def get_relevant_keywords(expected_job):

    expected_job = expected_job.lower()

    if "machine learning" in expected_job:
        return [
            "machine learning",
            "data scientist",
            "deep learning",
            "ai",
            "artificial intelligence"
        ]

    if "sql" in expected_job:
        return [
            "sql",
            "database",
            "data analyst",
            "data scientist",
            "data engineer"
        ]

    if "data engineer" in expected_job:
        return [
            "data engineer",
            "data engineering",
            "etl",
            "data pipeline"
        ]

    if "backend" in expected_job:
        return [
            "backend",
            "back end",
            "server",
            "api",
            "python developer"
        ]

    if "python" in expected_job:
        return [
            "python",
            "django",
            "backend",
            "data engineer",
            "data scientist"
        ]

    return [expected_job]


# ---------------------------------------
# Check Top-5 Relevance
# ---------------------------------------
def check_top5_relevance(top_5, expected_job):

    relevant_keywords = get_relevant_keywords(
        expected_job
    )

    for job in top_5:

        title = job["job_title"].lower()

        for keyword in relevant_keywords:

            if keyword in title:
                return True

    return False


# ---------------------------------------
# Evaluate One Resume
# ---------------------------------------
def evaluate_resume(
    resume_text,
    resume_skills,
    expected_job,
    model,
    collection
):

    resume_embedding = model.encode(resume_text)

    # Retrieve Top 200
    results = collection.query(
        query_embeddings=[
            resume_embedding.tolist()
        ],
        n_results=200,
        include=["metadatas", "embeddings"]
    )

    job_embeddings = results["embeddings"][0]
    metadatas = results["metadatas"][0]

    # Semantic similarity
    similarities = cosine_similarity(
        np.array(resume_embedding).reshape(1, -1),
        np.array(job_embeddings)
    )[0]

    ranked_jobs = []

    # Calculate scores
    for i, metadata in enumerate(metadatas):

        skill_score, matched_skills = calculate_skill_match(
            resume_skills,
            metadata["skills"]
        )

        experience_score = calculate_experience_match(
            resume_text,
            metadata["experience"]
        )

        # Same 60 / 30 / 10 logic as website
        final_score = (
            0.60 * similarities[i]
            + 0.30 * skill_score
            + 0.10 * experience_score
        )

        ranked_jobs.append({
            "job_title": metadata["job_title"],
            "company": metadata["company"],
            "final_score": final_score,
            "semantic_score": similarities[i],
            "skill_score": skill_score,
            "experience_score": experience_score
        })

    # Sort
    ranked_jobs.sort(
        key=lambda x: x["final_score"],
        reverse=True
    )

    # Top 5
    top_5 = ranked_jobs[:5]

    # ---------------------------------------
    # Display
    # ---------------------------------------
    print("\n-----------------------------------")
    print("Expected Job:", expected_job)
    print("-----------------------------------")

    print("\nTop 5 Results:")

    for rank, job in enumerate(top_5, start=1):

        print(
            f"{rank}. {job['job_title']} "
            f"({job['company']})"
        )

        print(
            f"   Final: "
            f"{job['final_score'] * 100:.2f}%"
        )

        print(
            f"   Semantic: "
            f"{job['semantic_score'] * 100:.2f}%"
        )

        print(
            f"   Skills: "
            f"{job['skill_score'] * 100:.2f}%"
        )

        print(
            f"   Experience: "
            f"{job['experience_score'] * 100:.2f}%"
        )

    # ---------------------------------------
    # Top-5 Relevance
    # ---------------------------------------
    top5_correct = check_top5_relevance(
        top_5,
        expected_job
    )

    if top5_correct:
        print("\nTop-5 Relevance: CORRECT")
    else:
        print("\nTop-5 Relevance: INCORRECT")

    return top5_correct


# ---------------------------------------
# Main
# ---------------------------------------
if __name__ == "__main__":

    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    collection = create_collection()

    test_cases = [

        {
            "resume_text": """
            Final year B.Tech student and Python developer
            with skills in Python, SQL, Django and REST API.
            Interested in backend development.
            """,
            "resume_skills":
                "Python, SQL, Django, REST API",
            "expected_job":
                "Python Developer"
        },

        {
            "resume_text": """
            Final year B.Tech student interested in machine
            learning and artificial intelligence. Skilled in
            Python, TensorFlow, machine learning and deep learning.
            """,
            "resume_skills":
                "Python, TensorFlow, Machine Learning, Deep Learning",
            "expected_job":
                "Machine Learning Engineer"
        },

        {
            "resume_text": """
            Final year B.Tech student interested in databases.
            Skilled in SQL, MySQL, database management and
            database development.
            """,
            "resume_skills":
                "SQL, MySQL, Database Management",
            "expected_job":
                "SQL Developer"
        },

        {
            "resume_text": """
            Final year B.Tech student interested in data engineering.
            Skilled in Python, SQL, ETL and data pipelines.
            """,
            "resume_skills":
                "Python, SQL, ETL, Data Pipelines",
            "expected_job":
                "Data Engineer"
        },

        {
            "resume_text": """
            Final year B.Tech student interested in backend development.
            Skilled in Python, Django, REST API and backend development.
            """,
            "resume_skills":
                "Python, Django, REST API",
            "expected_job":
                "Backend Developer"
        }
    ]

    correct_predictions = 0

    print("\n===================================")
    print("      SMARTMATCH AI EVALUATION")
    print("===================================")

    for test_case in test_cases:

        result = evaluate_resume(
            test_case["resume_text"],
            test_case["resume_skills"],
            test_case["expected_job"],
            model,
            collection
        )

        if result:
            correct_predictions += 1

    # ---------------------------------------
    # Final Top-5 Accuracy
    # ---------------------------------------
    total_tests = len(test_cases)

    top5_accuracy = (
        correct_predictions / total_tests
    ) * 100

    print("\n===================================")
    print("       FINAL EVALUATION")
    print("===================================")

    print(
        "Relevant Top-5 Predictions:",
        correct_predictions
    )

    print(
        "Total Test Cases:",
        total_tests
    )

    print(
        "Top-5 Relevance:",
        round(top5_accuracy, 2),
        "%"
    )

    print("===================================")