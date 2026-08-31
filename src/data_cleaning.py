import pandas as pd


def load_data(file_path):
    jobs = pd.read_csv(file_path)
    return jobs


def remove_duplicates(jobs):
    jobs = jobs.drop_duplicates()
    jobs = jobs.drop_duplicates(subset=["job_id"])
    return jobs

def handle_missing_values(jobs):
    text_columns = [
        "job_title",
        "company",
        "skills",
        "description",
        "location",
        "experience"
    ]

    for column in text_columns:
        jobs[column] = jobs[column].fillna("Not specified")

    return jobs

def clean_text(jobs):
    text_columns = [
        "job_title",
        "company",
        "skills",
        "description",
        "location",
        "experience"
    ]

    for column in text_columns:
        jobs[column] = (
            jobs[column]
            .astype(str)
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
        )

    return jobs

def create_combined_text(jobs):
    jobs["combined_text"] = (
        jobs["job_title"] + ". "
        + "Skills: " + jobs["skills"] + ". "
        + "Description: " + jobs["description"]
    )

    return jobs

def save_data(jobs, file_path):
    jobs.to_csv(file_path, index=False)

if __name__ == "__main__":
    jobs = load_data("data/raw/jobs.csv")

    print(jobs.head())

    print("\nDuplicate rows:", jobs.duplicated().sum())
    print("Duplicate job IDs:", jobs["job_id"].duplicated().sum())

    jobs = remove_duplicates(jobs)


    print("\nShape after removing duplicates:", jobs.shape)

    jobs = handle_missing_values(jobs)

    print("\nMissing values after cleaning:")

    print(jobs.isnull().sum())

    jobs = clean_text(jobs)

    print("\nData after text cleaning:")

    print(jobs.head())

    jobs = create_combined_text(jobs)

    print("\nCombined text:")

    print(jobs[["job_title", "combined_text"]].head())

    print("\nCleaned data saved successfully.")

    save_data(jobs, "data/processed/cleaned_jobs.csv")

    print("\nCleaned data saved successfully.")