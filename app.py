import streamlit as st
import pandas as pd
from pypdf import PdfReader
import chromadb
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
import html


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="SmartMatch AI",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed",
)



# =========================================================
# PREMIUM UI
# =========================================================

st.markdown(r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

.stApp {
    background:
        radial-gradient(circle at 50% -10%, rgba(99,102,241,.10), transparent 32%),
        #f7f8fc;
    color: #0f172a;
    font-family: 'Inter', sans-serif;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { max-width: 1080px; padding: 1rem 1.5rem 4rem; }

/* Navigation */
.nav {
    height: 58px;
    display:flex;
    align-items:center;
    justify-content:space-between;
    border-bottom:1px solid #e8ebf2;
    margin-bottom: 15px;
}
.brand { font-size:18px; font-weight:800; letter-spacing:-.6px; color:#111827; }
.brand-mark { color:#5146e5; }
.nav-note { font-size:11px; color:#94a3b8; font-weight:600; }

/* Home */
.hero { text-align:center; padding:72px 0 32px; }
.eyebrow {
    display:inline-flex; align-items:center; gap:7px;
    padding:7px 12px; border:1px solid #dfe3ff; border-radius:999px;
    background:#f1f3ff; color:#5146e5; font-size:10px; font-weight:800;
    letter-spacing:1px; text-transform:uppercase;
}
.hero h1 {
    margin:18px auto 0; max-width:760px; color:#0b1020;
    font-size:54px; line-height:1.02; letter-spacing:-2.8px; font-weight:800;
}
.hero h1 span { color:#5146e5; }
.hero p { max-width:620px; margin:18px auto 0; color:#64748b; font-size:15px; line-height:1.7; }

.upload-shell {
    max-width:720px; margin:24px auto 0; padding:24px;
    background:rgba(255,255,255,.92); border:1px solid #dfe4ee;
    border-radius:22px; box-shadow:0 18px 55px rgba(15,23,42,.08);
}
.upload-heading { display:flex; align-items:center; gap:12px; text-align:left; margin-bottom:16px; }
.upload-icon {
    width:40px; height:40px; display:flex; align-items:center; justify-content:center;
    border-radius:12px; background:#eef0ff; color:#5146e5; font-size:20px;
}
.upload-heading strong { display:block; font-size:14px; color:#111827; }
.upload-heading span { display:block; margin-top:3px; font-size:11px; color:#94a3b8; }

[data-testid="stFileUploader"] { max-width:720px; margin:0 auto; }
[data-testid="stFileUploaderDropzone"] {
    min-height:135px !important; border:1.5px dashed #bfc6da !important;
    border-radius:16px !important; background:#fbfcff !important;
}
[data-testid="stFileUploaderDropzone"]:hover { border-color:#5146e5 !important; background:#f8f8ff !important; }
[data-testid="stFileUploaderDropzoneInstructions"] > div:first-child { display:none !important; }
[data-testid="stFileUploaderDropzoneInstructions"] { color:#64748b !important; }

.trust-row {
    max-width:720px; margin:24px auto 0; display:flex; justify-content:center; gap:0;
    color:#64748b; border:1px solid #e5e8ef; background:#fff; border-radius:14px; overflow:hidden;
}
.trust-item { flex:1; padding:15px 10px; text-align:center; border-right:1px solid #edf0f5; }
.trust-item:last-child { border-right:0; }
.trust-item strong { display:block; color:#111827; font-size:16px; }
.trust-item span { display:block; margin-top:3px; font-size:10px; color:#94a3b8; }

.steps { margin:62px auto 0; max-width:820px; }
.steps-title { text-align:center; color:#111827; font-size:18px; font-weight:750; }
.steps-sub { text-align:center; color:#94a3b8; font-size:12px; margin-top:5px; }
.step-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:14px; margin-top:20px; }
.step { background:#fff; border:1px solid #e6e9f0; border-radius:16px; padding:20px; text-align:left; }
.step-num { color:#5146e5; font-size:10px; font-weight:800; letter-spacing:1px; }
.step h4 { margin:9px 0 5px; color:#111827; font-size:14px; }
.step p { margin:0; color:#64748b; font-size:11px; line-height:1.6; }

/* Buttons */
.stButton > button {
    border-radius:11px !important; min-height:43px !important; font-weight:700 !important;
    border:1px solid #e1e5ed !important; color:#334155 !important; background:#fff !important;
}
.stButton > button[kind="primary"] {
    color:#fff !important; background:#5146e5 !important; border-color:#5146e5 !important;
    box-shadow:0 8px 20px rgba(81,70,229,.22) !important;
}

/* Results */
.results-top { padding:42px 0 22px; }
.kicker { color:#5146e5; font-size:10px; font-weight:800; letter-spacing:1.4px; }
.results-title { margin-top:8px; font-size:34px; line-height:1.1; letter-spacing:-1.4px; font-weight:800; color:#0b1020; }
.results-sub { margin-top:7px; color:#64748b; font-size:13px; }

.profile-bar {
    display:flex; align-items:center; justify-content:space-between; gap:20px;
    background:#fff; border:1px solid #e2e6ef; border-radius:18px; padding:18px 20px;
    box-shadow:0 8px 28px rgba(15,23,42,.04); margin-bottom:18px;
}
.profile-name { font-size:11px; color:#64748b; font-weight:600; }
.profile-name strong { color:#111827; }
.profile-skills { margin-top:9px; }
.skill { display:inline-block; padding:5px 9px; margin-right:5px; border-radius:999px; background:#f1f3ff; color:#4f46e5; font-size:10px; font-weight:700; }
.profile-summary { text-align:right; min-width:115px; }
.profile-summary strong { display:block; font-size:20px; color:#111827; }
.profile-summary span { font-size:10px; color:#94a3b8; }

.feature-label { margin:30px 0 10px; font-size:10px; font-weight:800; color:#64748b; letter-spacing:1.2px; text-transform:uppercase; }
.featured {
    background:#fff; border:1px solid #d9dcf7; border-radius:22px; padding:26px;
    box-shadow:0 16px 45px rgba(79,70,229,.09); position:relative; overflow:hidden;
}
.featured:before { content:""; position:absolute; left:0; top:0; bottom:0; width:4px; background:#5146e5; }
.feature-head { display:flex; justify-content:space-between; gap:28px; }
.feature-rank { color:#5146e5; font-size:10px; font-weight:800; letter-spacing:1px; }
.feature-title { margin-top:7px; color:#0b1020; font-size:24px; font-weight:800; letter-spacing:-.8px; }
.feature-company { margin-top:5px; color:#64748b; font-size:12px; }
.feature-meta { margin-top:10px; color:#64748b; font-size:11px; }
.score-box { text-align:right; min-width:125px; }
.score-ring {
    width:78px; height:78px; border-radius:50%; margin-left:auto;
    display:flex; align-items:center; justify-content:center;
    background:conic-gradient(#5146e5 var(--score), #e9ebf4 0);
    position:relative;
}
.score-ring:after { content:""; width:62px; height:62px; border-radius:50%; background:#fff; position:absolute; }
.score-ring span { position:relative; z-index:1; color:#5146e5; font-size:16px; font-weight:800; }
.score-label { margin-top:7px; font-size:9px; color:#94a3b8; font-weight:700; letter-spacing:.8px; text-transform:uppercase; }

.match-pills { margin-top:17px; }
.matched { display:inline-block; padding:6px 9px; margin-right:5px; border-radius:999px; background:#ecfdf5; color:#047857; font-size:10px; font-weight:700; }
.no-match { color:#94a3b8; font-size:11px; }

.breakdown { margin-top:20px; padding:17px; background:#f8f9fc; border:1px solid #edf0f5; border-radius:15px; }
.breakdown-title { color:#334155; font-size:10px; font-weight:800; letter-spacing:.6px; }
.metric { margin-top:12px; }
.metric-line { display:flex; justify-content:space-between; color:#64748b; font-size:10px; margin-bottom:5px; }
.metric-line strong { color:#334155; }
.bar { height:5px; border-radius:99px; background:#e5e7ef; overflow:hidden; }
.bar > span { display:block; height:100%; border-radius:99px; background:#6366f1; }
.why { margin-top:16px; padding-top:14px; border-top:1px solid #e8ebf2; }
.why-title { color:#334155; font-size:10px; font-weight:800; }
.why-text { margin-top:6px; color:#64748b; font-size:11px; line-height:1.6; }

.other-title { margin:34px 0 10px; color:#64748b; font-size:10px; font-weight:800; letter-spacing:1.2px; text-transform:uppercase; }
.compact {
    background:#fff; border:1px solid #e3e7ef; border-radius:16px; padding:17px 19px;
    margin:10px 0; box-shadow:0 5px 18px rgba(15,23,42,.035);
}
.compact-grid { display:grid; grid-template-columns:1fr auto; gap:18px; align-items:center; }
.compact-rank { color:#94a3b8; font-size:9px; font-weight:800; letter-spacing:1px; }
.compact-title { margin-top:5px; color:#111827; font-size:15px; font-weight:750; }
.compact-company { margin-top:4px; color:#64748b; font-size:11px; }
.compact-meta { margin-top:7px; color:#94a3b8; font-size:10px; }
.compact-score { text-align:right; color:#5146e5; font-size:21px; font-weight:800; }
.compact-score small { display:block; color:#94a3b8; font-size:8px; letter-spacing:.7px; text-transform:uppercase; }

.footer { text-align:center; padding:50px 0 5px; color:#a1a8b7; font-size:10px; }

@media (max-width: 700px) {
    .hero { padding-top:45px; }
    .hero h1 { font-size:39px; letter-spacing:-1.8px; }
    .step-grid { grid-template-columns:1fr; }
    .trust-row { flex-direction:column; }
    .trust-item { border-right:0; border-bottom:1px solid #edf0f5; }
    .trust-item:last-child { border-bottom:0; }
    .profile-bar { align-items:flex-start; flex-direction:column; }
    .profile-summary { text-align:left; }
    .feature-head { flex-direction:column; }
    .score-box { text-align:left; }
    .score-ring { margin-left:0; }
    .compact-grid { grid-template-columns:1fr auto; }
}
</style>
""", unsafe_allow_html=True)
# =========================================================
# MODEL / DATABASE
# =========================================================

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource
def create_collection():
    """Create/load the ChromaDB job collection for local and cloud deployment."""
    client = chromadb.PersistentClient(path="chroma_db")

    data_path = "data/processed/cleaned_jobs.csv"
    jobs = pd.read_csv(data_path)

    # Reuse the existing database when it already contains the full dataset.
    collection = client.get_or_create_collection(name="jobs")
    if collection.count() == len(jobs):
        return collection

    # If the database is missing or incomplete, rebuild it automatically.
    try:
        client.delete_collection(name="jobs")
    except Exception:
        pass

    collection = client.get_or_create_collection(name="jobs")

    model = load_model()
    job_texts = jobs["combined_text"].fillna("").astype(str).tolist()
    embeddings = model.encode(
        job_texts,
        batch_size=64,
        show_progress_bar=False
    )

    ids = jobs["job_id"].astype(str).tolist()
    documents = job_texts
    metadatas = jobs[
        ["job_title", "company", "location", "experience", "skills"]
    ].fillna("Not specified").astype(str).to_dict(orient="records")

    # Chroma has a maximum batch size, so add the records in batches.
    batch_size = 5000
    for start in range(0, len(ids), batch_size):
        end = start + batch_size
        collection.add(
            ids=ids[start:end],
            embeddings=embeddings[start:end].tolist(),
            documents=documents[start:end],
            metadatas=metadatas[start:end]
        )

    return collection


# =========================================================
# MATCHING FUNCTIONS
# =========================================================

def extract_skills(skill_text):
    skills = skill_text.lower().split(",")
    return {skill.strip() for skill in skills if skill.strip()}


def calculate_skill_match(resume_skills, job_skills):
    resume_skills = extract_skills(resume_skills)
    job_skills = extract_skills(job_skills)

    if not job_skills or "not specified" in job_skills:
        return 0, set()

    matched_skills = resume_skills.intersection(job_skills)
    score = len(matched_skills) / len(job_skills)

    return score, matched_skills


def calculate_experience_match(resume_text, job_experience):
    resume_text = resume_text.lower()
    job_experience = str(job_experience).lower()

    fresher_keywords = [
        "student",
        "fresher",
        "final year",
        "undergraduate",
        "b.tech",
        "btech",
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


def generate_match_explanation(
    matched_skills,
    experience_score,
    job_experience
):
    reasons = []

    for skill in sorted(matched_skills):
        reasons.append(
            f"✓ {skill.title()} matches your profile"
        )

    if experience_score == 1.0:
        reasons.append(
            "✓ Experience requirement is suitable for a fresher"
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

    if not matched_skills:
        reasons.append("⚠ No direct skill overlap detected")

    return reasons


# =========================================================
# RESUME PROCESSING
# =========================================================

def extract_resume(uploaded_file):
    if uploaded_file.name.lower().endswith(".pdf"):
        reader = PdfReader(uploaded_file)
        text = ""

        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

        return text

    return uploaded_file.read().decode("utf-8")


def detect_resume_skills(resume_text):
    skill_list = [
        "python", "java", "c", "sql", "mysql", "html", "css",
        "javascript", "django", "rest api", "machine learning",
        "deep learning", "tensorflow", "pandas", "numpy", "excel",
        "google sheets", "git", "github", "data analysis",
        "data analytics", "data pipelines", "etl"
    ]

    detected = []
    text = resume_text.lower()

    for skill in skill_list:
        pattern = r"\b" + re.escape(skill) + r"\b"

        if re.search(pattern, text):
            detected.append(skill)

    return detected


# =========================================================
# FIND MATCHES
# =========================================================

def find_matches(resume_text, resume_skills):
    model = load_model()
    collection = create_collection()

    resume_embedding = model.encode(resume_text)

    results = collection.query(
        query_embeddings=[resume_embedding.tolist()],
        n_results=200,
        include=["metadatas", "embeddings"]
    )

    job_embeddings = results["embeddings"][0]

    similarities = cosine_similarity(
        resume_embedding.reshape(1, -1),
        job_embeddings
    )[0]

    skill_scores = []
    matched_skills_list = []
    experience_scores = []

    for metadata in results["metadatas"][0]:
        skill_score, matched_skills = calculate_skill_match(
            resume_skills,
            metadata["skills"]
        )

        experience_score = calculate_experience_match(
            resume_text,
            metadata["experience"]
        )

        skill_scores.append(skill_score)
        matched_skills_list.append(matched_skills)
        experience_scores.append(experience_score)

    final_scores = (
        0.6 * np.array(similarities)
        + 0.3 * np.array(skill_scores)
        + 0.1 * np.array(experience_scores)
    )

    ranked_jobs = []

    for i, metadata in enumerate(results["metadatas"][0]):
        ranked_jobs.append({
            "job_title": metadata["job_title"],
            "company": metadata["company"],
            "location": metadata["location"],
            "experience": metadata["experience"],
            "similarity": similarities[i],
            "skill_score": skill_scores[i],
            "experience_score": experience_scores[i],
            "final_score": final_scores[i],
            "matched_skills": matched_skills_list[i],
            "explanation": generate_match_explanation(
                matched_skills_list[i],
                experience_scores[i],
                metadata["experience"]
            ),
        })

    ranked_jobs.sort(
        key=lambda x: x["final_score"],
        reverse=True
    )

    return ranked_jobs[:5]



# =========================================================
# SESSION STATE
# =========================================================

if "page" not in st.session_state:
    st.session_state.page = "home"
if "resume_text" not in st.session_state:
    st.session_state.resume_text = ""
if "resume_name" not in st.session_state:
    st.session_state.resume_name = ""
if "detected_skills" not in st.session_state:
    st.session_state.detected_skills = []
if "matches" not in st.session_state:
    st.session_state.matches = None

# =========================================================
# NAVIGATION
# =========================================================

st.html("""
<div class="nav">
    <div class="brand"><span class="brand-mark">✦</span> SmartMatch AI</div>
    <div class="nav-note">AI-powered career matching</div>
</div>
""")

# =========================================================
# HOME
# =========================================================

if st.session_state.page == "home":
    st.html("""
    <div class="hero">
        <div class="eyebrow">✦ AI-POWERED JOB MATCHING</div>
        <h1>Turn your resume into your <span>next opportunity.</span></h1>
        <p>Upload your resume and let SmartMatch AI rank the most relevant roles using semantic similarity, skills, and experience.</p>
    </div>
    """)

    st.html("""
    <div class="upload-shell">
        <div class="upload-heading">
            <div class="upload-icon">↑</div>
            <div>
                <strong>Upload your resume</strong>
                <span>PDF or TXT · Your resume is processed locally</span>
            </div>
        </div>
    </div>
    """)

    uploaded_file = st.file_uploader(
        "Resume",
        type=["pdf", "txt"],
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        resume_text = extract_resume(uploaded_file)
        detected_skills = detect_resume_skills(resume_text)
        st.session_state.resume_text = resume_text
        st.session_state.resume_name = uploaded_file.name
        st.session_state.detected_skills = detected_skills

        if st.button("Analyze Resume & Find Matches  →", type="primary", use_container_width=True):
            with st.spinner("Analyzing your resume and ranking jobs..."):
                st.session_state.matches = find_matches(
                    resume_text,
                    ", ".join(detected_skills)
                )
            st.session_state.page = "results"
            st.rerun()

    st.html("""
    <div class="trust-row">
        <div class="trust-item"><strong>7,277+</strong><span>JOB RECORDS</span></div>
        <div class="trust-item"><strong>384D</strong><span>SEMANTIC EMBEDDINGS</span></div>
        <div class="trust-item"><strong>TOP 5</strong><span>RANKED MATCHES</span></div>
    </div>

    <div class="steps">
        <div class="steps-title">How SmartMatch works</div>
        <div class="steps-sub">Three simple steps from resume to relevant opportunities.</div>
        <div class="step-grid">
            <div class="step"><div class="step-num">01</div><h4>Upload</h4><p>Provide your resume as a PDF or text file.</p></div>
            <div class="step"><div class="step-num">02</div><h4>Analyze</h4><p>AI understands your profile, skills, and experience.</p></div>
            <div class="step"><div class="step-num">03</div><h4>Match</h4><p>Jobs are ranked using a combined matching score.</p></div>
        </div>
    </div>
    """)

# =========================================================
# RESULTS
# =========================================================

else:
    if st.button("← Upload another resume"):
        st.session_state.page = "home"
        st.session_state.matches = None
        st.rerun()

    matches = st.session_state.matches or []
    skills = st.session_state.detected_skills

    st.html("""
    <div class="results-top">
        <div class="kicker">RESUME ANALYSIS</div>
        <div class="results-title">Your best opportunities.</div>
        <div class="results-sub">Ranked from the job database based on your profile.</div>
    </div>
    """)

    skill_html = "".join(
        f'<span class="skill">{html.escape(skill.title())}</span>' for skill in skills
    )
    if not skill_html:
        skill_html = '<span class="skill">No predefined skills detected</span>'

    st.html(f"""
    <div class="profile-bar">
        <div>
            <div class="profile-name">CANDIDATE PROFILE · <strong>{html.escape(st.session_state.resume_name)}</strong></div>
            <div class="profile-skills">{skill_html}</div>
        </div>
        <div class="profile-summary">
            <strong>{len(matches)}</strong>
            <span>RECOMMENDED ROLES</span>
        </div>
    </div>
    """)

    with st.expander("View extracted resume text"):
        st.text(st.session_state.resume_text)

    if matches:
        # Featured #1
        job = matches[0]
        final_percentage = job["final_score"] * 100
        semantic_percentage = job["similarity"] * 100
        skill_percentage = job["skill_score"] * 100
        experience_percentage = job["experience_score"] * 100
        title = html.escape(str(job["job_title"]).title())
        company = html.escape(str(job["company"]).title())
        location = html.escape(str(job["location"]).title())
        experience = html.escape(str(job["experience"]))
        matched = "".join(f'<span class="matched">✓ {html.escape(skill.title())}</span>' for skill in sorted(job["matched_skills"]))
        if not matched:
            matched = '<span class="no-match">No direct skill overlap detected</span>'
        reasons = " ".join(html.escape(str(x).replace("✓ ", "").replace("⚠ ", "")) for x in job["explanation"])
        ring_score = max(0, min(100, final_percentage))

        st.html(f"""
        <div class="feature-label">TOP MATCH</div>
        <div class="featured">
            <div class="feature-head">
                <div>
                    <div class="feature-rank">#1 · BEST FIT</div>
                    <div class="feature-title">{title}</div>
                    <div class="feature-company">⌂ {company}</div>
                    <div class="feature-meta">{location} &nbsp;·&nbsp; {experience}</div>
                </div>
                <div class="score-box">
                    <div class="score-ring" style="--score:{ring_score}%;"><span>{final_percentage:.1f}%</span></div>
                    <div class="score-label">Overall match</div>
                </div>
            </div>
            <div class="match-pills">{matched}</div>
            <div class="breakdown">
                <div class="breakdown-title">WHY THIS MATCH</div>
                <div class="metric"><div class="metric-line"><span>Semantic relevance</span><strong>{semantic_percentage:.1f}%</strong></div><div class="bar"><span style="width:{semantic_percentage}%;"></span></div></div>
                <div class="metric"><div class="metric-line"><span>Skill match</span><strong>{skill_percentage:.1f}%</strong></div><div class="bar"><span style="width:{skill_percentage}%;"></span></div></div>
                <div class="metric"><div class="metric-line"><span>Experience suitability</span><strong>{experience_percentage:.1f}%</strong></div><div class="bar"><span style="width:{experience_percentage}%;"></span></div></div>
                <div class="why"><div class="why-title">MATCH INSIGHT</div><div class="why-text">{reasons}</div></div>
            </div>
        </div>
        """)

        # Compact #2-#5
        st.html('<div class="other-title">Other relevant opportunities</div>')
        for rank, job in enumerate(matches[1:], start=2):
            final_percentage = job["final_score"] * 100
            title = html.escape(str(job["job_title"]).title())
            company = html.escape(str(job["company"]).title())
            location = html.escape(str(job["location"]).title())
            experience = html.escape(str(job["experience"]))
            matched = "".join(f'<span class="matched">✓ {html.escape(skill.title())}</span>' for skill in sorted(job["matched_skills"]))
            st.html(f"""
            <div class="compact">
                <div class="compact-grid">
                    <div>
                        <div class="compact-rank">MATCH {rank}</div>
                        <div class="compact-title">{title}</div>
                        <div class="compact-company">⌂ {company}</div>
                        <div class="compact-meta">{location} &nbsp;·&nbsp; {experience}</div>
                        <div style="margin-top:8px;">{matched if matched else '<span class="no-match">No direct skill overlap</span>'}</div>
                    </div>
                    <div class="compact-score"><small>overall match</small>{final_percentage:.1f}%</div>
                </div>
            </div>
            """)
    else:
        st.html('<div class="empty">No matching jobs were found.</div>')

    st.html("""
    <div class="footer">
        <strong>SmartMatch AI</strong> · Semantic job matching with Sentence Transformers & ChromaDB
    </div>
    """)
