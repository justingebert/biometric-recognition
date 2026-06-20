# Biometrics Assignment – Grade 1.0 Checklist
*Advanced Topics – Introduction to Biometrics – SoSe 26 – Knaut*
*Max: 60 pts | Based on tutorial: https://www.youtube.com/watch?v=bK_k7eebGgc*

---
## What you need to build (beyond the base tutorial)
 
### Basic Performance Evaluation *(required for higher grades)*
- [ ] Implement the biometric error metrics from the course textbook chapter "Biometric System Error":
  - [ ] **FNMR** (False Non-Match Rate)
  - [ ] **FMR** (False Match Rate)
  - [ ] **DET curve** (Detection Error Tradeoff)
  - [ ] **Histogram of genuine / impostor attempts**
- [ ] Metrics reference material was shared in the course forum (used for the first presentation)
### Enhanced Training & Testing *(required for grade 1.3–1.0)*
- [ ] Implement a version that can **verify more than one face (5–10 faces)**
- [ ] **Explain in the Jupyter Notebook** what was changed compared to Renotte's original implementation

---

## Foundation (required for any grade ≥ 3.7)

- [ ] Jupyter Notebook runs without errors
- [X] Code hosted on HTW GitLab (submit only the repo link in Moodle)
- [ ] `README.md` includes:
  - [ ] Required Python version
  - [ ] Required dependency versions / installation remarks
  - [ ] Original source code and paper author(s) credited
  - [ ] All other used sources properly referenced
- [ ] *(nice to have / extra points)*: `requirements.txt` for pip included
- [ ] *(nice to have / extra points)*: Docker image provided with setup instructions using pyenv/venv

---

## Grade 2.7–3.3 additions (≥ 60 * 0.6 pts)

- [ ] Installation remarks written from the perspective of another developer (not just yourself)
- [ ] Jupyter Notebook and/or `README.md` includes a **background / purpose section**:
  - [ ] Explains the main idea and purpose of the implementation
  - [ ] Can reference the paper — may also be written directly in the notebook
- [ ] Code runs with minimal or no manual corrections needed (< 0.5h)

---

## Grade 1.7–2.3 additions (≥ 60 * 0.75 pts)

- [ ] **Enhanced performance measurement** is provided and **visualized** in the Jupyter Notebook
  - [ ] (e.g. FAR, FRR, EER, DET curves, d-prime — metrics relevant to biometric evaluation)
- [ ] Code runs with almost no corrections needed; any remaining issues handled live on **21.07.26**
- [ ] `README.md` is a **complete project documentation**
  - [ ] Comparable in quality to: https://github.com/nevoit/Siamese-Neural-Networks-for-One-shot-Image-Recognition
- [ ] Performance measurement results are **interpreted thoroughly** in `README.md`
  - [ ] Not just visualized — explained and analyzed

---

## Grade 1.0–1.3 additions (≥ 60 * 0.9 pts)

- [ ] **Multi-subject evaluation**: complex evaluation with multiple test subjects performed and documented
  - [ ] References "Enhanced Training & Testing" slides
  - [ ] Test subject data stored on HTW server (with consent) or commented in notebook where own data is needed
- [ ] **Proper exception handling** on all I/O operations
- [ ] **Implementation and evaluation presented live** on **21.07.26**

### For a perfect 1.0 (60 * 0.95–1.0 pts) — "Altogether Stunning and Perfect":
- [ ] *(one or more of the following)*:
  - [ ] Actively helped other students solve issues
  - [ ] Found, documented, and fixed errors beyond dependency/version problems
  - [ ] Implemented and documented alternative solutions/models in notebook or `README.md`
  - [ ] Created a new tutorial (filmed, Jupyter Notebook, or similar)
  - [ ] Contributed genuinely new ideas to the project

---

## 📅 Key Dates

| Date | Event |
|------|-------|
| Submission deadline | ~4 weeks from now |
| **21.07.26** | Live presentation and evaluation session |

---

## 💡 Extra Points (any level)

These can boost your score at any tier:
- [ ] Helping/supporting other students in the course forum
- [ ] Theoretical background section in `README.md`
- [ ] Enhanced evaluation beyond what's required at your current tier


## decision:
- ship models using lfs
- ship 