import pandas as pd
import re


# -----------------------------
# Load Data
# -----------------------------

def load_data(file_path):

    df = pd.read_csv(file_path)

    return df


# -----------------------------
# Remove Duplicates
# -----------------------------

def remove_duplicates(df):

    df = df.drop_duplicates()

    return df


# -----------------------------
# Handle Missing Values
# -----------------------------

def handle_missing_values(df):

    text_columns = [
        "Job Title",
        "Company Name",
        "Location",
        "Experience",
        "Job Description",
        "Skills"
    ]

    for column in text_columns:

        df[column] = df[column].fillna("Not specified")

    return df


# -----------------------------
# Clean Text
# -----------------------------

def clean_text(text):

    text = str(text)

    text = text.lower()

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    text = text.strip()

    return text


# -----------------------------
# Create Project Columns
# -----------------------------

def create_project_columns(df):

    df["job_id"] = range(
        1,
        len(df) + 1
    )

    df["job_title"] = df["Job Title"].apply(
        clean_text
    )

    df["company"] = df["Company Name"].apply(
        clean_text
    )

    df["location"] = df["Location"].apply(
        clean_text
    )

    df["experience"] = df["Experience"].apply(
        clean_text
    )

    df["description"] = df["Job Description"].apply(
        clean_text
    )

    df["skills"] = df["Skills"].apply(
        clean_text
    )

    return df


# -----------------------------
# Create Combined Text
# -----------------------------

def create_combined_text(df):

    df["combined_text"] = (

        df["job_title"]
        + ". Skills: "
        + df["skills"]
        + ". Description: "
        + df["description"]

    )

    return df


# -----------------------------
# Save Data
# -----------------------------

def save_data(
    df,
    output_path
):

    columns = [
        "job_id",
        "job_title",
        "company",
        "skills",
        "description",
        "location",
        "experience",
        "combined_text"
    ]

    df[columns].to_csv(
        output_path,
        index=False
    )


# -----------------------------
# Main
# -----------------------------

if __name__ == "__main__":

    input_file = (
        "data/raw/Data Science_Jobs.csv"
    )

    output_file = (
        "data/processed/cleaned_jobs.csv"
    )


    print("Loading dataset...")

    jobs = load_data(
        input_file
    )

    print(
        "Original rows:",
        len(jobs)
    )


    jobs = remove_duplicates(
        jobs
    )

    print(
        "After removing duplicates:",
        len(jobs)
    )


    jobs = handle_missing_values(
        jobs
    )


    jobs = create_project_columns(
        jobs
    )


    jobs = create_combined_text(
        jobs
    )


    save_data(
        jobs,
        output_file
    )


    print(
        "Cleaned dataset saved successfully."
    )

    print(
        "Final rows:",
        len(jobs)
    )

    print(
        "Final columns:",
        jobs[
            [
                "job_id",
                "job_title",
                "company",
                "skills",
                "description",
                "location",
                "experience"
            ]
        ].columns.tolist()
    )