def extract_skills(skill_text):
    skills = skill_text.lower().split(",")

    skills = [
        skill.strip()
        for skill in skills
        if skill.strip()
    ]

    return set(skills)


def calculate_skill_match(resume_skills, job_skills):

    resume_skills = extract_skills(resume_skills)
    job_skills = extract_skills(job_skills)

    if not job_skills:
        return 0, set()

    matched_skills = resume_skills.intersection(job_skills)

    score = len(matched_skills) / len(job_skills)

    return score, matched_skills


if __name__ == "__main__":

    resume_skills = "Python, SQL, Django, REST API, Machine Learning"

    job_skills = "Python, SQL, Django, REST API"

    score, matched = calculate_skill_match(
        resume_skills,
        job_skills
    )

    print("Matched skills:", matched)
    print("Skill Match Score:", round(score * 100, 2), "%")