import streamlit as st
from pypdf import PdfReader
import chromadb
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
import html


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="SmartMatch AI",
    page_icon="💼",
    layout="wide"
)


# =========================================================
# PROFESSIONAL UI STYLING
# =========================================================

st.markdown(
    """
    <style>

    /* =====================================================
       GLOBAL
       ===================================================== */

    .stApp {
        background: #f7f9fc;
    }

    .block-container {
        max-width: 1180px;
        padding-top: 3rem;
        padding-bottom: 3rem;
    }


    /* =====================================================
       HEADER
       ===================================================== */

    .main-title {
        font-size: 44px;
        font-weight: 800;
        color: #172033;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 19px;
        font-weight: 500;
        color: #475569;
        margin-bottom: 8px;
    }

    .description {
        font-size: 15px;
        color: #64748b;
        line-height: 1.6;
        margin-bottom: 30px;
    }


    /* =====================================================
       UPLOAD CARD
       ===================================================== */

    .upload-card {
        background: white;
        padding: 35px;
        border-radius: 20px;
        border: 1px solid #e2e8f0;
        margin-top: 15px;
        margin-bottom: 20px;
        box-shadow: 0 8px 25px rgba(15, 23, 42, 0.06);
        text-align: center;
    }

    .upload-icon {
        font-size: 42px;
        margin-bottom: 10px;
    }

    .upload-title {
        font-size: 25px;
        font-weight: 800;
        color: #172033;
        margin-bottom: 8px;
    }

    .upload-text {
        font-size: 14px;
        color: #64748b;
    }


    /* =====================================================
       PROFILE CARD
       ===================================================== */

    .profile-card {
        background: white;
        padding: 25px;
        border-radius: 18px;
        border: 1px solid #e2e8f0;
        margin-top: 20px;
        margin-bottom: 25px;
        box-shadow: 0 6px 20px rgba(15, 23, 42, 0.05);
    }

    .profile-title {
        font-size: 20px;
        font-weight: 800;
        color: #172033;
        margin-bottom: 5px;
    }

    .profile-subtitle {
        font-size: 14px;
        color: #64748b;
        margin-bottom: 15px;
    }


    /* =====================================================
       SKILL TAGS
       ===================================================== */

    .skill-tag {
        display: inline-block;
        padding: 7px 13px;
        margin: 4px;
        border-radius: 20px;
        background: #eef2ff;
        color: #3730a3;
        font-size: 13px;
        font-weight: 650;
    }

    .matched-skill {
        display: inline-block;
        padding: 7px 12px;
        margin: 3px;
        border-radius: 16px;
        background: #ecfdf5;
        color: #047857;
        font-size: 12px;
        font-weight: 700;
    }


    /* =====================================================
       SECTION HEADERS
       ===================================================== */

    .section-title {
        font-size: 29px;
        font-weight: 800;
        color: #172033;
        margin-top: 35px;
        margin-bottom: 5px;
    }

    .section-description {
        font-size: 14px;
        color: #64748b;
        margin-bottom: 20px;
    }


    /* =====================================================
       JOB CARD
       ===================================================== */

    .job-card {
        background: white;
        padding: 26px;
        border-radius: 19px;
        border: 1px solid #e2e8f0;
        margin-top: 24px;
        margin-bottom: 8px;
        box-shadow: 0 7px 24px rgba(15, 23, 42, 0.06);
    }

    .job-rank {
        font-size: 11px;
        font-weight: 800;
        color: #64748b;
        letter-spacing: 1.2px;
        margin-bottom: 6px;
        text-transform: uppercase;
    }

    .job-title {
        font-size: 25px;
        font-weight: 800;
        color: #172033;
        margin-bottom: 5px;
    }

    .job-company {
        font-size: 15px;
        color: #64748b;
        margin-bottom: 14px;
    }

    .job-info {
        font-size: 14px;
        color: #475569;
    }


    /* =====================================================
       SCORE BOX
       ===================================================== */

    .score-box {
        background: #eef2ff;
        border-radius: 16px;
        padding: 15px;
        text-align: center;
        border: 1px solid #e0e7ff;
    }

    .score-label {
        font-size: 10px;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 800;
    }

    .big-score {
        font-size: 31px;
        font-weight: 850;
        color: #3730a3;
        margin-top: 4px;
    }


    /* =====================================================
       METRIC CARDS
       ===================================================== */

    .metric-card {
        background: #ffffff;
        border-radius: 14px;
        padding: 15px;
        text-align: center;
        border: 1px solid #e2e8f0;
    }

    .metric-label {
        font-size: 11px;
        color: #64748b;
        margin-bottom: 5px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .metric-value {
        font-size: 20px;
        font-weight: 800;
        color: #172033;
    }


    /* =====================================================
       EXPLANATION
       ===================================================== */

    .explanation-card {
        background: #f8fafc;
        border-radius: 14px;
        padding: 17px;
        margin-top: 18px;
        border: 1px solid #e2e8f0;
    }

    .explanation-title {
        font-size: 15px;
        font-weight: 800;
        color: #172033;
        margin-bottom: 10px;
    }

    .explanation-item {
        font-size: 13px;
        color: #475569;
        margin: 7px 0;
    }


    /* =====================================================
       FOOTER
       ===================================================== */

    .footer {
        text-align: center;
        color: #94a3b8;
        margin-top: 55px;
        padding: 25px;
        font-size: 13px;
        line-height: 1.7;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# LOAD AI MODEL
# =========================================================

@st.cache_resource
def load_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


# =========================================================
# CONNECT TO CHROMADB
# =========================================================

@st.cache_resource
def create_collection():

    client = chromadb.PersistentClient(
        path="chroma_db"
    )

    collection = client.get_or_create_collection(
        name="jobs"
    )

    return collection


# =========================================================
# SKILL MATCHING
# =========================================================

def extract_skills(skill_text):

    skills = skill_text.lower().split(",")

    skills = [
        skill.strip()
        for skill in skills
        if skill.strip()
    ]

    return set(skills)


def calculate_skill_match(
    resume_skills,
    job_skills
):

    resume_skills = extract_skills(
        resume_skills
    )

    job_skills = extract_skills(
        job_skills
    )

    if (
        not job_skills
        or "not specified" in job_skills
    ):

        return 0, set()


    matched_skills = (
        resume_skills.intersection(
            job_skills
        )
    )


    score = (
        len(matched_skills)
        / len(job_skills)
    )


    return score, matched_skills


# =========================================================
# EXPERIENCE MATCHING
# =========================================================

def calculate_experience_match(
    resume_text,
    job_experience
):

    resume_text = resume_text.lower()

    job_experience = job_experience.lower()


    # Detect student / fresher
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

        # Specifically marked as fresher
        if "fresher" in job_experience:

            return 1.0


        # Extract experience numbers
        numbers = re.findall(

            r"\d+(?:\.\d+)?",

            job_experience

        )


        if numbers:

            minimum_experience = float(
                numbers[0]
            )


            # 0 years
            if minimum_experience == 0:

                return 1.0


            # Minimum 1 year
            elif minimum_experience == 1:

                return 0.6


            # More than 1 year
            else:

                return 0.0


        # Unknown experience
        return 0.5


    # Non-fresher
    return 0.5


# =========================================================
# MATCH EXPLANATION
# =========================================================

def generate_match_explanation(
    matched_skills,
    experience_score,
    job_experience
):

    reasons = []


    # Explain matched skills
    for skill in sorted(
        matched_skills
    ):

        reasons.append(
            f"✓ {skill.title()} matches your profile"
        )


    # Explain experience
    if experience_score == 1.0:

        reasons.append(
            "✓ Experience requirement is suitable "
            "for a fresher"
        )

    elif experience_score == 0.6:

        reasons.append(
            f"⚠ This role requires {job_experience}; "
            "the candidate is a fresher"
        )

    elif experience_score == 0.0:

        reasons.append(
            f"⚠ This role requires {job_experience}, "
            "which is above the fresher level"
        )


    # No direct skill overlap
    if not matched_skills:

        reasons.append(
            "⚠ No direct skill overlap detected"
        )


    return reasons


# =========================================================
# MAIN HEADER
# =========================================================

st.markdown(
    """
    <div class="main-title">
        💼 SmartMatch AI
    </div>

    <div class="subtitle">
        AI-Powered Resume & Job Matching
    </div>

    <div class="description">
        Discover relevant job opportunities using
        semantic similarity, skill matching,
        and experience suitability.
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# UPLOAD CARD
# =========================================================

st.html(
    """
    <div class="upload-card">

        <div class="upload-icon">
            📄
        </div>

        <div class="upload-title">
            Upload your Resume
        </div>

        <div class="upload-text">
            Upload a PDF or TXT resume to discover
            your most relevant job opportunities.
        </div>

    </div>
    """
)


uploaded_file = st.file_uploader(

    "Choose your resume",

    type=[
        "pdf",
        "txt"
    ],

    label_visibility="collapsed"

)


# =========================================================
# PROCESS RESUME
# =========================================================

if uploaded_file is not None:


    # =====================================================
    # SUCCESS MESSAGE
    # =====================================================

    st.success(
        f"Resume uploaded successfully: "
        f"{uploaded_file.name}"
    )


    # =====================================================
    # EXTRACT RESUME TEXT
    # =====================================================

    if uploaded_file.name.lower().endswith(
        ".pdf"
    ):

        pdf_reader = PdfReader(
            uploaded_file
        )

        resume_text = ""

        for page in pdf_reader.pages:

            text = page.extract_text()

            if text:

                resume_text += (
                    text + "\n"
                )

    else:

        resume_text = (

            uploaded_file
            .read()
            .decode("utf-8")

        )


    # =====================================================
    # EXTRACT RESUME SKILLS
    # =====================================================

    skill_list = [

        "python",
        "java",
        "c",
        "sql",
        "mysql",
        "html",
        "css",
        "javascript",
        "django",
        "rest api",
        "machine learning",
        "deep learning",
        "tensorflow",
        "pandas",
        "numpy",
        "excel",
        "google sheets",
        "git",
        "github",
        "data analysis",
        "data analytics",
        "data pipelines",
        "etl"

    ]


    detected_skills = []

    resume_text_lower = (
        resume_text.lower()
    )


    for skill in skill_list:

        pattern = (
            r"\b"
            + re.escape(skill)
            + r"\b"
        )


        if re.search(
            pattern,
            resume_text_lower
        ):

            detected_skills.append(
                skill
            )


    resume_skills = ", ".join(
        detected_skills
    )


    # =====================================================
    # CANDIDATE PROFILE
    # =====================================================

    skill_tags = ""


    for skill in detected_skills:

        safe_skill = html.escape(
            skill.title()
        )

        skill_tags += (
            f'<span class="skill-tag">'
            f'{safe_skill}'
            f'</span>'
        )


    if not skill_tags:

        skill_tags = (
            '<span class="skill-tag">'
            'No predefined skills detected'
            '</span>'
        )


    st.html(
        f"""
        <div class="profile-card">

            <div class="profile-title">
                👤 Candidate Profile
            </div>

            <div class="profile-subtitle">
                Skills detected from your resume
            </div>

            <div>
                {skill_tags}
            </div>

        </div>
        """
    )


    # =====================================================
    # RESUME PREVIEW
    # =====================================================

    with st.expander(
        "📄 View Extracted Resume Text"
    ):

        st.text_area(

            "Resume Content",

            resume_text,

            height=300,

            label_visibility="collapsed"

        )


    # =====================================================
    # FIND MATCHING JOBS
    # =====================================================

    if st.button(

        "🔍 Find Matching Jobs",

        use_container_width=True

    ):


        with st.spinner(
            "Analyzing your resume and finding the best matches..."
        ):


            # =================================================
            # LOAD MODEL
            # =================================================

            model = load_model()


            # =================================================
            # CONNECT TO CHROMADB
            # =================================================

            collection = (
                create_collection()
            )


            # =================================================
            # CREATE RESUME EMBEDDING
            # =================================================

            resume_embedding = model.encode(
                resume_text
            )


            # =================================================
            # RETRIEVE TOP 200 CANDIDATES
            # =================================================

            results = collection.query(

                query_embeddings=[
                    resume_embedding.tolist()
                ],

                n_results=200,

                include=[
                    "metadatas",
                    "embeddings"
                ]

            )


            # =================================================
            # JOB EMBEDDINGS
            # =================================================

            job_embeddings = (
                results["embeddings"][0]
            )


            # =================================================
            # SEMANTIC SIMILARITY
            # =================================================

            similarities = (

                cosine_similarity(

                    resume_embedding.reshape(
                        1,
                        -1
                    ),

                    job_embeddings

                )[0]

            )


            # =================================================
            # SKILL MATCHING
            # =================================================

            skill_scores = []

            matched_skills_list = []


            for metadata in (
                results["metadatas"][0]
            ):

                job_skills = metadata[
                    "skills"
                ]


                skill_score, matched_skills = (

                    calculate_skill_match(

                        resume_skills,

                        job_skills

                    )

                )


                skill_scores.append(
                    skill_score
                )


                matched_skills_list.append(
                    matched_skills
                )


            # =================================================
            # EXPERIENCE MATCHING
            # =================================================

            experience_scores = []


            for metadata in (
                results["metadatas"][0]
            ):

                job_experience = metadata[
                    "experience"
                ]


                experience_score = (

                    calculate_experience_match(

                        resume_text,

                        job_experience

                    )

                )


                experience_scores.append(
                    experience_score
                )


            # =================================================
            # FINAL MATCH SCORE
            # =================================================

            final_scores = (

                0.6 * np.array(
                    similarities
                )

                + 0.3 * np.array(
                    skill_scores
                )

                + 0.1 * np.array(
                    experience_scores
                )

            )


            # =================================================
            # CREATE RANKED JOB LIST
            # =================================================

            ranked_jobs = []


            for i in range(

                len(
                    results[
                        "metadatas"
                    ][0]
                )

            ):


                explanation = (

                    generate_match_explanation(

                        matched_skills_list[i],

                        experience_scores[i],

                        results[
                            "metadatas"
                        ][0][i][
                            "experience"
                        ]

                    )

                )


                ranked_jobs.append({

                    "job_title":
                        results[
                            "metadatas"
                        ][0][i][
                            "job_title"
                        ],

                    "company":
                        results[
                            "metadatas"
                        ][0][i][
                            "company"
                        ],

                    "location":
                        results[
                            "metadatas"
                        ][0][i][
                            "location"
                        ],

                    "experience":
                        results[
                            "metadatas"
                        ][0][i][
                            "experience"
                        ],

                    "similarity":
                        similarities[i],

                    "skill_score":
                        skill_scores[i],

                    "experience_score":
                        experience_scores[i],

                    "final_score":
                        final_scores[i],

                    "matched_skills":
                        matched_skills_list[i],

                    "explanation":
                        explanation

                })


            # =================================================
            # SORT JOBS
            # =================================================

            ranked_jobs.sort(

                key=lambda x:
                    x["final_score"],

                reverse=True

            )


        # =====================================================
        # RESULTS HEADER
        # =====================================================

        st.markdown(
            """
            <div class="section-title">
                🏆 Top Job Matches
            </div>

            <div class="section-description">
                Jobs ranked using semantic relevance,
                skill overlap, and experience suitability.
            </div>
            """,
            unsafe_allow_html=True
        )


        # =====================================================
        # DISPLAY TOP 5
        # =====================================================

        for rank, job in enumerate(

            ranked_jobs[:5],

            start=1

        ):


            # =================================================
            # PERCENTAGES
            # =================================================

            final_percentage = (
                job["final_score"] * 100
            )

            similarity_percentage = (
                job["similarity"] * 100
            )

            skill_percentage = (
                job["skill_score"] * 100
            )

            experience_percentage = (
                job["experience_score"] * 100
            )


            # =================================================
            # SAFE TEXT
            # =================================================

            job_title = html.escape(
                str(
                    job["job_title"]
                ).title()
            )

            company = html.escape(
                str(
                    job["company"]
                ).title()
            )

            location = html.escape(
                str(
                    job["location"]
                ).title()
            )

            experience = html.escape(
                str(
                    job["experience"]
                )
            )


            # =================================================
            # JOB CARD
            # =================================================

            st.html(
                f"""
                <div class="job-card">

                    <div class="job-rank">
                        MATCH #{rank}
                    </div>

                    <div class="job-title">
                        {job_title}
                    </div>

                    <div class="job-company">
                        🏢 {company}
                    </div>

                    <div class="job-info">
                        📍 {location}
                        &nbsp;&nbsp;&nbsp;&nbsp;
                        💼 {experience}
                    </div>

                </div>
                """
            )


            # =================================================
            # SCORE METRICS
            # =================================================

            col1, col2, col3, col4 = st.columns(

                [1.3, 1, 1, 1]

            )


            # -------------------------------------------------
            # Overall Match
            # -------------------------------------------------

            with col1:

                st.html(
                    f"""
                    <div class="score-box">

                        <div class="score-label">
                            Overall Match
                        </div>

                        <div class="big-score">
                            {final_percentage:.1f}%
                        </div>

                    </div>
                    """
                )


            # -------------------------------------------------
            # Semantic
            # -------------------------------------------------

            with col2:

                st.html(
                    f"""
                    <div class="metric-card">

                        <div class="metric-label">
                            Semantic
                        </div>

                        <div class="metric-value">
                            {similarity_percentage:.1f}%
                        </div>

                    </div>
                    """
                )


            # -------------------------------------------------
            # Skills
            # -------------------------------------------------

            with col3:

                st.html(
                    f"""
                    <div class="metric-card">

                        <div class="metric-label">
                            Skills
                        </div>

                        <div class="metric-value">
                            {skill_percentage:.1f}%
                        </div>

                    </div>
                    """
                )


            # -------------------------------------------------
            # Experience
            # -------------------------------------------------

            with col4:

                st.html(
                    f"""
                    <div class="metric-card">

                        <div class="metric-label">
                            Experience
                        </div>

                        <div class="metric-value">
                            {experience_percentage:.1f}%
                        </div>

                    </div>
                    """
                )


            # =================================================
            # SCORE BREAKDOWN
            # =================================================

            st.write("")


            st.write(
                f"**Semantic Relevance** "
                f"— {similarity_percentage:.1f}%"
            )

            st.progress(
                float(
                    np.clip(
                        job["similarity"],
                        0.0,
                        1.0
                    )
                )
            )


            st.write(
                f"**Skill Match** "
                f"— {skill_percentage:.1f}%"
            )

            st.progress(
                float(
                    np.clip(
                        job["skill_score"],
                        0.0,
                        1.0
                    )
                )
            )


            st.write(
                f"**Experience Match** "
                f"— {experience_percentage:.1f}%"
            )

            st.progress(
                float(
                    np.clip(
                        job["experience_score"],
                        0.0,
                        1.0
                    )
                )
            )


            # =================================================
            # MATCHING SKILLS
            # =================================================

            if job["matched_skills"]:

                matched_skill_tags = ""


                for skill in sorted(
                    job["matched_skills"]
                ):

                    safe_skill = html.escape(
                        skill.title()
                    )

                    matched_skill_tags += (

                        f'<span class="matched-skill">'
                        f'✓ {safe_skill}'
                        f'</span>'

                    )


                st.html(
                    f"""
                    <div class="matched-title">
                        🎯 Matching Skills
                    </div>

                    <div>
                        {matched_skill_tags}
                    </div>
                    """
                )

            else:

                st.caption(
                    "No direct skill overlap detected."
                )


            # =================================================
            # WHY THIS JOB MATCHES
            # =================================================

            explanation_items = ""


            for reason in job["explanation"]:

                safe_reason = html.escape(
                    reason
                )


                explanation_items += (

                    f'<div class="explanation-item">'
                    f'{safe_reason}'
                    f'</div>'

                )


            st.html(
                f"""
                <div class="explanation-card">

                    <div class="explanation-title">
                        💡 Why this job matches
                    </div>

                    {explanation_items}

                </div>
                """
            )


            st.divider()


# =========================================================
# FOOTER
# =========================================================

st.html(
    """
    <div class="footer">

        <strong>SmartMatch AI</strong>
        <br>

        Resume Intelligence & Job Matching

        <br>

        Powered by Sentence Transformers,
        ChromaDB and Streamlit

    </div>
    """
)