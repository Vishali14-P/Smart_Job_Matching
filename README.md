# SmartMatch AI

## AI-Powered Resume & Job Matching

SmartMatch AI is an AI-powered job recommendation system that matches a candidate's resume with relevant job opportunities.

The system combines:

- Semantic similarity
- Skill matching
- Experience suitability

It retrieves relevant jobs from a job dataset and ranks them based on an overall matching score.

---

## Project Objective

The objective of this project is to build a simple and explainable AI-based job matching system that can:

1. Accept a candidate's resume.
2. Extract and clean the resume text.
3. Detect relevant skills.
4. Convert the resume and job descriptions into embeddings.
5. Search for semantically similar jobs.
6. Compare candidate skills with job requirements.
7. Consider the candidate's experience level.
8. Rank the most relevant jobs.
9. Display the Top 5 job recommendations through a web interface.

---

## System Architecture

```text
                 Resume PDF / TXT
                        |
                        v
                Resume Text Extraction
                        |
                        v
                  Text Processing
                        |
             +----------+----------+
             |                     |
             v                     v
       Skill Detection       Resume Embedding
                                   |
                                   v
                            ChromaDB Search
                                   |
                                   v
                          Top 200 Job Candidates
                                   |
                    +--------------+--------------+
                    |              |              |
                    v              v              v
              Semantic Score   Skill Score   Experience Score
                    |              |              |
                    +--------------+--------------+
                                   |
                                   v
                         Weighted Final Score
                         60% + 30% + 10%
                                   |
                                   v
                              Top 5 Jobs
                                   |
                                   v
                            Streamlit Interface