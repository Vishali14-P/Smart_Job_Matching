import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def calculate_similarity(resume_embedding, job_embeddings):

    resume_embedding = np.array(
        resume_embedding
    ).reshape(1, -1)

    job_embeddings = np.array(
        job_embeddings
    )

    similarities = cosine_similarity(
        resume_embedding,
        job_embeddings
    )[0]

    return similarities


def calculate_match_scores(similarities):

    scores = similarities * 100

    return scores


def calculate_experience_match(
    resume_experience,
    job_experience
):

    resume_experience = resume_experience.lower()
    job_experience = job_experience.lower()


    # Fresher / student resume

    if (
        "fresher" in resume_experience
        or "student" in resume_experience
        or "0 year" in resume_experience
    ):

        if (
            "fresher" in job_experience
            or "0-1" in job_experience
            or "0 - 1" in job_experience
            or "0-2" in job_experience
            or "0 - 2" in job_experience
        ):

            return 1.0

        return 0.0


    # If experience information is unavailable

    if (
        not resume_experience.strip()
        or resume_experience == "not specified"
    ):

        return 0.5


    return 0.5


def calculate_final_scores(
    similarities,
    skill_scores,
    experience_scores
):

    final_scores = (

        0.6 * np.array(similarities)

        + 0.3 * np.array(skill_scores)

        + 0.1 * np.array(experience_scores)

    )

    return final_scores