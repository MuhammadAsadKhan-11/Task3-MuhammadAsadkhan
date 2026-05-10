# 🤖 Tech Stack Recommender
### AI Recommendation Logic — Project 3 | DecodeLabs Industrial Training Kit | Batch 2026

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange?style=flat-square&logo=scikit-learn)
![pandas](https://img.shields.io/badge/pandas-1.3%2B-150458?style=flat-square&logo=pandas)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)

---

## 📌 Table of Contents

- [Overview](#-overview)
- [How It Works](#-how-it-works)
- [Project Structure](#-project-structure)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Usage](#-usage)
- [Dataset](#-dataset)
- [Algorithm Deep Dive](#-algorithm-deep-dive)
- [Pipeline Breakdown](#-pipeline-breakdown)
- [Sample Output](#-sample-output)
- [Extending the Project](#-extending-the-project)
- [Known Limitations](#-known-limitations)
- [Contributing](#-contributing)

---

## 🔍 Overview

The **Tech Stack Recommender** is a content-based filtering recommendation engine that maps a user's raw skills to the most relevant tech career paths. Given at least three skill inputs (e.g., `Python`, `Docker`, `Machine_Learning`), the system computes a ranked list of the top 3 matching job roles from a curated dataset.

This project is the capstone of **Project 3: AI Recommendation Logic** from the DecodeLabs Industrial Training Kit. It demonstrates a shift from passive data classification to **active preference prediction** — the core of how real-world systems like Netflix, Amazon, and LinkedIn recommendations work.

**What this is NOT:**
- It does not use collaborative filtering (no user-vs-user comparisons).
- It does not require historical interaction logs or massive datasets.
- It is not a neural network — it is pure algorithmic similarity logic.

**What this IS:**
- A fully functional **content-based filtering** engine.
- Built on the **Input → Process → Output (IPO)** architecture.
- Uses industry-standard **TF-IDF vectorization** + **Cosine Similarity** scoring.

---

## ⚙️ How It Works

At a high level, the system follows three phases:

```
[ USER SKILLS INPUT ]  →  [ TF-IDF + COSINE SIMILARITY ]  →  [ TOP-3 RANKED ROLES ]
     (User State)               (Similarity Logic)                  (Top-N List)
```

Every job role and the user's skill set are converted into **numerical vectors** in a shared mathematical space. The engine then measures the **angular alignment** (cosine similarity) between the user's vector and each job role vector, sorts by score, and returns the top matches.

---

## 📁 Project Structure

```
tech-stack-recommender/
│
├── tech_recommender.py     # Main recommendation engine script
├── raw_skills.csv          # Dataset: 15 job roles with associated skills
├── README.md               # This file
└── requirements.txt        # Python dependencies
```

---

## ✅ Prerequisites

- Python **3.8 or higher**
- pip (Python package manager)

Verify your Python version:
```bash
python --version
```

---

## 🚀 Installation

**1. Clone the repository**
```bash
git clone https://github.com/your-username/tech-stack-recommender.git
cd tech-stack-recommender
```

**2. (Recommended) Create a virtual environment**
```bash
python -m venv venv

# Activate on Windows:
venv\Scripts\activate

# Activate on macOS/Linux:
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install scikit-learn pandas numpy
```

---

## 🖥️ Usage

Run the recommender from your terminal:

```bash
python tech_recommender.py
```

The program will prompt you to enter your skills one by one. A minimum of **3 skills** is required for accurate matching.

**Example session:**
```
Enter skill 1 (at least 3 more needed): Python
Enter skill 2 (at least 2 more needed): SQL
Enter skill 3 (at least 1 more needed): Machine_Learning
Enter skill 4 (or press Enter to finish): Docker
Enter skill 5 (or press Enter to finish): [Enter]
```

> **Tip:** Use underscores for multi-word skills, e.g., `Machine_Learning`, `Data_Analysis`, `Deep_Learning`.

---

## 📊 Dataset

The dataset (`raw_skills.csv`) contains **15 job roles**, each tagged with a set of relevant skills.

| Job Role | Sample Skills |
|---|---|
| Data Scientist | Python, SQL, Machine_Learning, TensorFlow, Statistics |
| ML Engineer | Python, PyTorch, Deep_Learning, Docker, Kubernetes |
| Backend Developer | Java, Python, SQL, REST, APIs, Docker |
| DevOps Engineer | AWS, Docker, Kubernetes, CI_CD, Automation |
| Frontend Developer | JavaScript, React, HTML, CSS, TypeScript |
| Cloud Architect | AWS, Azure, GCP, Kubernetes, Networking |
| Cybersecurity Engineer | Security, Cryptography, Ethical_Hacking, Linux |
| AI Engineer | Python, NLP, Computer_Vision, TensorFlow, Research |
| Full Stack Developer | JavaScript, React, Node.js, SQL, Docker |
| ... | ... |

**Full list of 15 roles:** Data Scientist, ML Engineer, Backend Developer, Frontend Developer, DevOps Engineer, Cloud Architect, Data Analyst, Cybersecurity Engineer, Mobile Developer, Full Stack Developer, Database Administrator, AI Engineer, Systems Administrator, Blockchain Developer, Game Developer.

### Adding Custom Roles

You can extend the dataset by editing `raw_skills.csv`. Follow the existing format:

```csv
job_role,skills
Your_Role,Skill1 Skill2 Skill3 Skill4 Skill5
```

> **Important:** Skills in a single row must be separated by **spaces**, not commas. The comma is the column separator between `job_role` and `skills`.

---

## 🧮 Algorithm Deep Dive

### Why Content-Based Filtering?

| Approach | Requires | Best For |
|---|---|---|
| **Collaborative Filtering** | Large user interaction history | Netflix, Spotify |
| **Content-Based Filtering** ✅ | Only item attributes | Cold starts, small datasets |

Content-based filtering maps user preferences directly to item properties, independent of other users' behavior. This makes it ideal for this project since we don't need historical data.

---

### Step 1 — TF-IDF Vectorization

Raw text skills cannot be compared mathematically. TF-IDF converts them to weighted numerical vectors.

**Term Frequency (TF):**
```
TF(t, d) = Count of term t in document d / Total terms in document d
```

**Inverse Document Frequency (IDF):**
```
IDF(t) = log( Total Documents / Documents containing term t )
```

**Final Weight:**
```
TF-IDF(t, d) = TF(t, d) × IDF(t)
```

- A skill like `Python` (appears in many roles) gets a **low IDF** → lower weight.
- A skill like `Solidity` (appears in only Blockchain) gets a **high IDF** → higher weight.
- This rewards specificity and penalizes generic terms automatically.

---

### Step 2 — Cosine Similarity

Once both the user profile and all job roles are TF-IDF vectors, we measure how "aligned" they are using the cosine of the angle between them:

```
cos(θ) = (A · B) / (||A|| × ||B||)
```

Where:
- `A · B` = dot product of the two vectors
- `||A||` and `||B||` = magnitudes (lengths) of each vector

| Score | Meaning |
|---|---|
| `1.0` | Perfect alignment — identical skill orientation |
| `0.5–0.9` | Strong match |
| `0.1–0.4` | Partial match |
| `0.0` | No shared skills (orthogonal vectors) |
| `-1.0` | Opposite (not possible with TF-IDF since all values ≥ 0) |

**Why Cosine over Euclidean Distance?**

Euclidean distance is sensitive to vector magnitude. If one job role has 10 skills and another has 3, a user matching 2 skills in each case would unfairly rank the longer-description role as "farther away." Cosine similarity only cares about **direction**, not length — making it magnitude-invariant and far more accurate for text-based matching.

---

## 🔄 Pipeline Breakdown

The system follows a strict 4-step assembly line:

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  1.INGEST   │ →  │  2. SCORE   │ →  │  3. SORT    │ →  │  4. FILTER  │
│             │    │             │    │             │    │             │
│ Load CSV    │    │ Cosine Sim  │    │ Descending  │    │ Top-N List  │
│ Get 3+      │    │ per job     │    │ by score    │    │ (Top 3)     │
│ user skills │    │ role        │    │             │    │             │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

| Step | Function in Code | Purpose |
|---|---|---|
| Ingestion | `load_dataset()` + `get_user_profile()` | Capture data and user state |
| Scoring | `score_recommendations()` | Compute cosine similarity |
| Sorting | `get_top_recommendations()` | Rank by relevance |
| Filtering | `head(top_n)` | Return Top-3, prevent choice overload |

---

## 📋 Sample Output

```
=======================================================
   🚀  TECH STACK RECOMMENDER — DecodeLabs
=======================================================

User Input: ['Python', 'SQL', 'Machine_Learning', 'Docker']

=======================================================
   🎯  TOP 3 RECOMMENDED CAREER PATHS
=======================================================

  🥇 Rank #1: ML Engineer
     Match Score : 0.4211  (42% alignment)
     Your Skills : Python, SQL, Machine_Learning, Docker
     Matched With: Python, Docker
     Role Skills : Python TensorFlow PyTorch Machine Learning Docker Kubernetes

  🥈 Rank #2: Data Scientist
     Match Score : 0.3907  (39% alignment)
     Your Skills : Python, SQL, Machine_Learning, Docker
     Matched With: Python, SQL
     Role Skills : Python SQL Machine Learning Statistics TensorFlow Pandas

  🥉 Rank #3: Backend Developer
     Match Score : 0.3539  (35% alignment)
     Your Skills : Python, SQL, Machine_Learning, Docker
     Matched With: Python, SQL, Docker
     Role Skills : Java Python SQL REST Docker Databases Microservices

=======================================================
✅ Recommendation complete! Build your skills accordingly.
=======================================================
```

---

## 🛠️ Extending the Project

Here are ways to level up this project beyond the base requirements:

### 1. Add More Job Roles
Edit `raw_skills.csv` to include more roles like `Prompt Engineer`, `Data Engineer`, `Robotics Engineer`, etc.

### 2. Add Skill Weighting (User Ratings)
Allow users to rate skills by proficiency (1–5). Multiply the TF-IDF score by proficiency weight before computing cosine similarity.

```python
# Example extension
skills_with_weights = {"Python": 5, "Docker": 3, "SQL": 4}
```

### 3. Show Why a Role Was Recommended
Display the top contributing skills to the similarity score by inspecting TF-IDF feature weights.

### 4. Build a Web Interface
Wrap the engine in a Flask or FastAPI backend with a simple HTML frontend where users select skills via checkboxes.

### 5. Add Hybrid Filtering
Combine content-based results with a "trending roles" fallback for the Cold Start problem — show globally popular roles when the user enters no recognized skills.

### 6. Expand the Dataset
Scrape real job postings from LinkedIn or Indeed to build a larger, real-world `raw_skills.csv` with hundreds of job roles.

---

## ⚠️ Known Limitations

| Limitation | Description | Fix |
|---|---|---|
| **Cold Start** | If the user enters skills not in the vocabulary, their vector is all zeros → similarity = 0 for all roles | Show "trending roles" fallback |
| **Vocabulary Mismatch** | `"Web Design"` ≠ `"Frontend_Development"` — naming must match the dataset | Standardize skill names or use synonym mapping |
| **Small Dataset** | 15 roles is a proof-of-concept — real engines use thousands of items | Extend `raw_skills.csv` |
| **No Learning** | The engine doesn't improve from feedback | Add a feedback loop to re-weight skills |
| **Binary Presence** | TF-IDF doesn't capture skill depth (beginner vs expert) | Integrate proficiency ratings |

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Commit your changes: `git commit -m "Add: your feature description"`
4. Push to the branch: `git push origin feature/your-feature-name`
5. Open a Pull Request

Please ensure your code follows the existing structure and includes comments explaining any new logic.

---

## 📄 License

This project is licensed under the MIT License.

---

## 🏫 About

Built as part of the **DecodeLabs Industrial Training Kit — Batch 2026**.
Project 3: AI Recommendation Logic | Track: Artificial Intelligence

> *"The absolute best way to master Artificial Intelligence is through hands-on practice, not just theory."*
> — DecodeLabs
