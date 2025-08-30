import streamlit as st
from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="AI Career Advisor (India)", page_icon="🎓", layout="wide")

# -----------------------------
# Knowledge Base (Careers)
# -----------------------------
CAREERS: Dict[str, Dict] = {
    "Data Scientist": {
        "skills": ["Python", "Machine Learning", "Statistics", "SQL"],
        "description": "Data Scientists analyze data to extract insights and build predictive models.",
        "roadmap": [
            "Learn Python, SQL, and Statistics",
            "Master data visualization (Matplotlib, Power BI, Tableau)",
            "Study Machine Learning (Scikit-learn, TensorFlow, PyTorch)",
            "Work on real-world datasets & Kaggle competitions",
            "Intern as Data Analyst → Move to Data Scientist role"
        ],
        "salary": "₹6–20 LPA in India"
    },
    "Software Engineer": {
        "skills": ["Java", "C++", "Python", "System Design"],
        "description": "Software Engineers design, develop, and maintain applications and systems.",
        "roadmap": [
            "Master programming languages (C++, Java, Python)",
            "Learn DSA & problem-solving (LeetCode, Codeforces)",
            "Understand DBMS, OS, Computer Networks",
            "Practice system design for scalability",
            "Apply for SDE internships and jobs"
        ],
        "salary": "₹4–18 LPA in India"
    },
    "Cloud Engineer": {
        "skills": ["Linux", "Networking", "GCP", "AWS", "Azure", "Docker", "Kubernetes"],
        "description": "Cloud Engineers manage and deploy applications on cloud platforms.",
        "roadmap": [
            "Learn basics of Linux & Networking",
            "Get certified in AWS/Azure/GCP",
            "Master containers (Docker, Kubernetes)",
            "Learn Infrastructure as Code (Terraform)",
            "Work on deploying scalable apps"
        ],
        "salary": "₹5–16 LPA in India"
    },
    "UI/UX Designer": {
        "skills": ["Figma", "Adobe XD", "Creativity", "User Research"],
        "description": "UI/UX Designers create intuitive and visually appealing user interfaces.",
        "roadmap": [
            "Learn Figma/Adobe XD for design",
            "Study design principles & color theory",
            "Understand UX research & user psychology",
            "Build a design portfolio",
            "Apply for UI/UX internships"
        ],
        "salary": "₹3–12 LPA in India"
    },
    "Business Analyst": {
        "skills": ["Excel", "SQL", "Data Visualization", "Problem Solving"],
        "description": "Business Analysts analyze processes and suggest improvements using data insights.",
        "roadmap": [
            "Master Excel & SQL",
            "Learn data visualization (Power BI, Tableau)",
            "Understand business processes",
            "Work on case studies & projects",
            "Apply for BA internships"
        ],
        "salary": "₹4–14 LPA in India"
    },
    "AI Engineer": {
        "skills": ["Python", "Deep Learning", "NLP", "TensorFlow", "PyTorch"],
        "description": "AI Engineers build, train, and deploy intelligent systems and AI models.",
        "roadmap": [
            "Learn Python, Linear Algebra & Probability",
            "Master Deep Learning frameworks (PyTorch/TensorFlow)",
            "Work on NLP & Computer Vision projects",
            "Understand MLOps for deploying AI models",
            "Contribute to open-source AI projects"
        ],
        "salary": "₹6–25 LPA in India"
    },
    "DevOps Engineer": {
        "skills": ["Linux", "AWS", "CI/CD", "Docker", "Kubernetes"],
        "description": "DevOps Engineers automate deployment, integration, and system monitoring.",
        "roadmap": [
            "Learn Linux, Git, and Shell scripting",
            "Understand CI/CD pipelines (Jenkins, GitHub Actions)",
            "Master containers & orchestration (Docker, Kubernetes)",
            "Learn cloud platforms (AWS/Azure/GCP)",
            "Work on real-world DevOps projects"
        ],
        "salary": "₹5–18 LPA in India"
    },
    "Cybersecurity Analyst": {
        "skills": ["Networking", "Ethical Hacking", "Security Tools", "Risk Management"],
        "description": "Cybersecurity Analysts protect systems from cyber threats and vulnerabilities.",
        "roadmap": [
            "Learn Networking & Security fundamentals",
            "Study firewalls, IDS/IPS, and encryption",
            "Get certified (CEH, CompTIA Security+, CISSP)",
            "Practice penetration testing & incident response",
            "Work in SOC (Security Operations Center)"
        ],
        "salary": "₹4–15 LPA in India"
    },
    "Data Engineer": {
        "skills": ["SQL", "Python", "ETL", "Big Data", "Spark"],
        "description": "Data Engineers design and maintain systems that collect and process large datasets.",
        "roadmap": [
            "Learn SQL and Python for data processing",
            "Understand ETL pipelines",
            "Work with Hadoop, Spark, Kafka",
            "Learn data warehousing (Snowflake, Redshift)",
            "Work on real-world data engineering projects"
        ],
        "salary": "₹6–18 LPA in India"
    },
    "Full Stack Developer": {
        "skills": ["JavaScript", "React", "Node.js", "Databases", "APIs"],
        "description": "Full Stack Developers work on both frontend and backend of web applications.",
        "roadmap": [
            "Learn HTML, CSS, JavaScript",
            "Master frontend frameworks (React, Angular, Vue)",
            "Learn backend with Node.js/Django/Flask",
            "Understand REST APIs & Databases",
            "Build and deploy full-stack projects"
        ],
        "salary": "₹4–16 LPA in India"
    },
    "Mobile App Developer": {
        "skills": ["Flutter", "React Native", "Java", "Kotlin", "Swift"],
        "description": "Mobile Developers build apps for Android and iOS platforms.",
        "roadmap": [
            "Learn Java/Kotlin for Android, Swift for iOS",
            "Explore cross-platform frameworks (Flutter, React Native)",
            "Understand UI/UX for mobile apps",
            "Practice app deployment on Play Store/App Store",
            "Work on real-world mobile projects"
        ],
        "salary": "₹3–14 LPA in India"
    },
    "Product Manager": {
        "skills": ["Communication", "Business Strategy", "Agile", "Market Research"],
        "description": "Product Managers lead product development and align business strategy with tech solutions.",
        "roadmap": [
            "Learn basics of business & product development",
            "Understand Agile & Scrum methodologies",
            "Work on product case studies",
            "Develop communication & leadership skills",
            "Apply for Associate PM roles"
        ],
        "salary": "₹8–30 LPA in India"
    },
    "QA Engineer": {
        "skills": ["Manual Testing", "Automation Testing", "Selenium", "JMeter"],
        "description": "QA Engineers ensure the quality of software by testing and reporting bugs.",
        "roadmap": [
            "Learn manual testing & test case writing",
            "Understand SDLC & STLC",
            "Master automation tools (Selenium, Cypress)",
            "Learn performance testing (JMeter, LoadRunner)",
            "Apply for QA roles"
        ],
        "salary": "₹3–12 LPA in India"
    },
    "Game Developer": {
        "skills": ["Unity", "C#", "C++", "Game Design", "3D Modeling"],
        "description": "Game Developers design and develop interactive video games for various platforms.",
        "roadmap": [
            "Learn game engines (Unity, Unreal)",
            "Practice programming with C#/C++",
            "Understand graphics, physics & AI in games",
            "Build indie game projects",
            "Apply to gaming studios"
        ],
        "salary": "₹4–15 LPA in India"
    },
    "Blockchain Developer": {
        "skills": ["Solidity", "Ethereum", "Smart Contracts", "Cryptography"],
        "description": "Blockchain Developers create decentralized applications and smart contracts.",
        "roadmap": [
            "Learn blockchain basics & cryptography",
            "Master Solidity & Ethereum",
            "Understand smart contracts & DApps",
            "Work on blockchain projects",
            "Contribute to Web3 open-source"
        ],
        "salary": "₹6–20 LPA in India"
    },
    "AR/VR Developer": {
        "skills": ["Unity", "C#", "3D Modeling", "ARKit", "ARCore"],
        "description": "AR/VR Developers build immersive augmented and virtual reality applications.",
        "roadmap": [
            "Learn Unity/Unreal for AR/VR",
            "Understand 3D modeling (Blender, Maya)",
            "Practice ARKit (iOS) & ARCore (Android)",
            "Build AR/VR projects",
            "Apply for AR/VR developer jobs"
        ],
        "salary": "₹5–18 LPA in India"
    },
    "IT Support Specialist": {
        "skills": ["Troubleshooting", "Networking", "Windows/Linux", "Customer Support"],
        "description": "IT Support Specialists provide technical assistance and solve hardware/software issues.",
        "roadmap": [
            "Learn computer hardware & OS basics",
            "Understand networking & troubleshooting",
            "Get certified (CompTIA A+, CCNA)",
            "Work in helpdesk or IT support roles",
            "Grow into system admin/network engineer"
        ],
        "salary": "₹2–8 LPA in India"
    },
    "Digital Marketer": {
        "skills": ["SEO", "Google Ads", "Content Marketing", "Analytics"],
        "description": "Digital Marketers promote businesses online using SEO, paid ads, and social media.",
        "roadmap": [
            "Learn SEO & SEM fundamentals",
            "Master social media marketing",
            "Understand Google Analytics & Ads",
            "Work on campaigns & content marketing",
            "Apply for digital marketing jobs"
        ],
        "salary": "₹3–10 LPA in India"
    },
    # From your earlier FastAPI roles:
    "Data Analyst": {
        "skills": ["SQL", "Excel", "Python", "Data Visualization", "Statistics"],
        "description": "Analyze data using SQL, Excel, Python; build dashboards and insights.",
        "roadmap": [
            "Master SQL (joins, window functions)",
            "Learn Excel advanced (pivots, charts, formulas)",
            "Use Python for data wrangling (pandas, numpy)",
            "Learn BI tools (Power BI/Tableau)",
            "Build dashboards and publish portfolio"
        ],
        "salary": "₹3.5–7 LPA in India"
    },
    "Product Analyst": {
        "skills": ["SQL", "A/B Testing", "Product Metrics", "Data Visualization"],
        "description": "Use product metrics and experimentation to inform roadmap decisions.",
        "roadmap": [
            "Learn core product metrics (retention, WAU/MAU, funnels)",
            "Study A/B testing design & analysis",
            "Query data with SQL",
            "Build dashboards and run experiments",
            "Partner with PMs to drive insights"
        ],
        "salary": "₹5–9 LPA in India"
    },
    "Cloud Support Associate": {
        "skills": ["Linux", "Networking", "GCP Core", "Scripting", "Customer Support"],
        "description": "Provide technical support for cloud services; troubleshoot infra issues.",
        "roadmap": [
            "Learn Linux & basic networking",
            "Study GCP/AWS fundamentals (cloud concepts)",
            "Practice scripting (Bash/Python)",
            "Understand ticketing & troubleshooting playbooks",
            "Prepare for Cloud Digital Leader/Cloud Practitioner"
        ],
        "salary": "₹3–6 LPA in India"
    },
    "Junior ML Engineer": {
        "skills": ["Python", "ML Basics", "Data Pipelines", "GCP Vertex AI", "Docker"],
        "description": "Build and deploy ML models; data pipelines; evaluate metrics.",
        "roadmap": [
            "Master Python & ML (scikit-learn, model eval)",
            "Build ETL/data pipelines",
            "Containerize with Docker",
            "Deploy simple models (FastAPI/Vertex AI endpoints)",
            "Track experiments & metrics"
        ],
        "salary": "₹6–12 LPA in India"
    },
}

# Simple skill → free course hints
SKILL_COURSES = {
    "Python": ["Kaggle: Python", "freeCodeCamp: Python for Data"],
    "SQL": ["Mode SQL Tutorial", "Khan Academy: SQL"],
    "Statistics": ["Khan Academy: Statistics", "StatQuest on YouTube"],
    "Machine Learning": ["Google ML Crash Course", "fast.ai Practical Deep Learning"],
    "Data Visualization": ["Power BI Microsoft Learn", "Tableau Free Training"],
    "Excel": ["ExcelJet", "Microsoft Learn: Excel"],
    "Linux": ["Linux Journey", "Ubuntu Tutorials"],
    "Networking": ["Cisco NetAcad Intro", "Professor Messer: Network+"],
    "Docker": ["Docker Getting Started", "Kubernetes Basics"],
    "Kubernetes": ["Kubernetes Docs Basics", "KodeKloud Free K8s Course"],
    "React": ["react.dev Learn", "Scrimba React Free"],
    "Node.js": ["nodejs.dev Learn", "The Odin Project: Node"],
    "Flutter": ["Flutter.dev Docs", "AppBrewery Free Intro"],
    "Java": ["JetBrains Academy Java Basics", "Oracle Java Tutorials"],
    "C++": ["cppreference (Learn)", "freeCodeCamp: C++"],
    "System Design": ["System Design Primer", "Grokking summaries"],
    "NLP": ["Hugging Face Course", "fast.ai NLP"],
    "Deep Learning": ["DeepLearning.AI Short Courses", "fast.ai"],
    "Cybersecurity": ["TryHackMe Free Rooms", "OverTheWire Wargames"],
}

# -----------------------------
# Helpers
# -----------------------------
def normalize(items: List[str]) -> List[str]:
    return [i.strip().lower() for i in items if isinstance(i, str)]

def score_match(user_skills: List[str], career_skills: List[str]) -> float:
    u = set(normalize(user_skills))
    r = set(normalize(career_skills))
    if not r:
        return 0.0
    return len(u.intersection(r)) / len(r)

def recommend_careers(name: str, skills: List[str], interests: List[str]):
    out = []
    for title, info in CAREERS.items():
        skill_score = score_match(skills, info["skills"])
        interest_score = 0.0
        if interests:
            interest_score = score_match(interests, info["skills"]) * 0.5
        final = 0.8 * skill_score + 0.2 * interest_score
        if final > 0:
            out.append({
                "career": title,
                "match": round(final * 100, 1),
                "description": info["description"],
                "roadmap": info["roadmap"],
                "salary": info["salary"],
                "skills_missed": [s for s in info["skills"] if s.lower() not in normalize(skills)]
            })
    out.sort(key=lambda x: x["match"], reverse=True)
    return out[:10]

def course_suggestions(recommendations):
    # Aggregate missing skills → courses
    needed = set()
    for r in recommendations:
        for s in r["skills_missed"]:
            needed.add(s)
    courses = {}
    for s in needed:
        courses[s] = SKILL_COURSES.get(s, ["Search Coursera/Udemy/YouTube for good intros"])
    return courses

# Build a simple semantic search over the career KB for Q&A
def build_kb_texts():
    docs = []
    labels = []
    for title, info in CAREERS.items():
        blob = (
            f"{title}. Description: {info['description']}. "
            f"Skills: {', '.join(info['skills'])}. "
            f"Roadmap: {' -> '.join(info['roadmap'])}. Salary: {info['salary']}."
        )
        docs.append(blob)
        labels.append(title)
    return docs, labels

DOCS, LABELS = build_kb_texts()
VEC = TfidfVectorizer(stop_words="english").fit(DOCS)
DOC_VEC = VEC.transform(DOCS)

def qa_answer(query: str):
    qv = VEC.transform([query])
    sims = cosine_similarity(qv, DOC_VEC)[0]
    idx = sims.argmax()
    title = LABELS[idx]
    info = CAREERS[title]
    answer = f"**{title}**\n\n{info['description']}\n\n**Top Skills:** {', '.join(info['skills'])}\n\n**Roadmap (high level):**\n" + \
             "\n".join([f"- {step}" for step in info["roadmap"]]) + \
             f"\n\n**Typical salary (entry to mid):** {info['salary']}"
    return answer, title, sims[idx]

# -----------------------------
# UI
# -----------------------------
def header():
    st.markdown(
        """
        <style>
        .hero {
            border-radius: 24px;
            padding: 36px 36px;
            background: radial-gradient(1200px 600px at 20% -20%, #1f7ae0 0%, transparent 40%),
                        radial-gradient(1000px 500px at 110% 0%, #9b59b6 0%, transparent 40%),
                        linear-gradient(135deg, #0f172a, #111827);
            color: white;
            box-shadow: 0 20px 50px rgba(0,0,0,0.35);
        }
        .pill {
            display:inline-block; padding:6px 12px; border-radius:999px; background:#1f2937; color:#d1d5db; font-size:12px; margin-right:6px;
        }
        .card {
            border-radius: 18px; padding:18px; background:#0b1220; border:1px solid #1f2937;
        }
        .roadstep {padding:10px 14px; margin:6px 0; border-left:4px solid #2563eb; background:#0f172a; border-radius:8px;}
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="hero">
            <h1 style="margin-bottom:4px;">🎓 AI Career Advisor — India</h1>
            <div style="opacity:0.9; font-size:16px;">
                Map your skills → discover careers → get roadmaps & courses → ask questions.
            </div>
            <div style="margin-top:12px;">
                <span class="pill">Personalized</span>
                <span class="pill">Roadmaps</span>
                <span class="pill">Courses</span>
                <span class="pill">Semantic Q&A</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

def sidebar():
    with st.sidebar:
        st.title("🔐 Quick Sign-in")
        st.session_state.setdefault("user", "")
        st.session_state["user"] = st.text_input("Your name", value=st.session_state["user"])
        st.markdown("---")
        st.caption("Made for hackathons • Streamlit Cloud deploy")

def home_form():
    st.subheader("Tell us about you")
    with st.form("profile_form"):
        name = st.text_input("Name")
        skills = st.text_input("Your skills (comma-separated)", placeholder="Python, SQL, Machine Learning")
        interests = st.text_input("Your interests (optional, comma-separated)", placeholder="data, product, cloud")
        submitted = st.form_submit_button("🔍 Get Career Matches")
    if submitted:
        u_skills = [s.strip() for s in skills.split(",") if s.strip()]
        u_interests = [s.strip() for s in interests.split(",") if s.strip()]
        recs = recommend_careers(name, u_skills, u_interests)
        st.session_state["profile"] = {"name": name, "skills": u_skills, "interests": u_interests}
        st.session_state["recommendations"] = recs
        if recs:
            st.success(f"Found {len(recs)} matches. Jump to the **Career Matches** tab!")
        else:
            st.warning("No matches yet. Try adding a few more skills.")

def matches_view():
    st.subheader("Career Matches")
    recs = st.session_state.get("recommendations", [])
    if not recs:
        st.info("No recommendations yet—fill the form in **Home**.")
        return

    for r in recs:
        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            cols = st.columns([4, 1])
            with cols[0]:
                st.markdown(f"### {r['career']}  —  {r['match']}% match")
                st.write(r["description"])
                st.caption(f"Salary: {r['salary']}")
                if r["skills_missed"]:
                    st.caption("Missing skills: " + ", ".join(r["skills_missed"]))
            with cols[1]:
                if st.button("📍 See Roadmap", key=f"rm_{r['career']}"):
                    st.session_state["selected_career"] = r["career"]
                    st.switch_page = "Roadmaps"
                if st.button("📚 Courses", key=f"cs_{r['career']}"):
                    st.session_state["selected_career"] = r["career"]
                    st.switch_page = "Courses"
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown("")

def roadmap_view():
    st.subheader("Roadmap")
    career = st.session_state.get("selected_career")
    recs = st.session_state.get("recommendations", [])
    if not recs:
        st.info("No recommendations yet—use **Home** first.")
        return
    if not career:
        career = recs[0]["career"]

    info = CAREERS.get(career)
    if not info:
        st.error("Career not found.")
        return

    st.markdown(f"### 🛣 {career}")
    st.write(info["description"])
    st.caption(f"Salary: {info['salary']}")
    st.markdown("#### Steps")
    for i, step in enumerate(info["roadmap"], start=1):
        st.markdown(f'<div class="roadstep"><b>Step {i}.</b> {step}</div>', unsafe_allow_html=True)

def courses_view():
    st.subheader("Courses")
    recs = st.session_state.get("recommendations", [])
    if not recs:
        st.info("No recommendations yet—use **Home** first.")
        return
    # aggregate missing skills → show courses
    suggestions = course_suggestions(recs)
    if not suggestions:
        st.success("You’re already covering most skills! Explore advanced topics.")
        return

    for skill, links in suggestions.items():
        with st.expander(f"📘 {skill} — recommended resources"):
            for link in links:
                st.write(f"- {link}")

def qa_view():
    st.subheader("Ask anything about careers (semantic search)")
    q = st.text_input("Your question", placeholder="e.g., What skills do I need for Data Engineer?")
    if st.button("Ask"):
        if not q.strip():
            st.warning("Type a question first.")
        else:
            answer, title, conf = qa_answer(q.strip())
            st.markdown(answer)
            st.caption(f"Match confidence: {conf:.2f}")

# -----------------------------
# Main
# -----------------------------
def main():
    header()
    sidebar()

    tabs = st.tabs(["🏠 Home", "🎯 Career Matches", "🛣 Roadmaps", "📚 Courses", "💬 Q&A"])

    with tabs[0]:
        home_form()
    with tabs[1]:
        matches_view()
    with tabs[2]:
        roadmap_view()
    with tabs[3]:
        courses_view()
    with tabs[4]:
        qa_view()

if __name__ == "__main__":
    main()
