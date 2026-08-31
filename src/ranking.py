import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def calculate_similarity(resume_embedding, job_embeddings):
    resume_embedding = np.array(resume_embedding).reshape(1, -1)
    job_embeddings = np.array(job_embeddings)

    similarities = cosine_similarity(
        resume_embedding,
        job_embeddings
    )[0]

    return similarities


def calculate_match_scores(similarities):
    scores = similarities * 100
    return scores