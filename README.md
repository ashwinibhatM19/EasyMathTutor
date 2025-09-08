# 🧮 Easy Math Tutor Pro

## 1. Objective of Project
The objective of this project is to create an **interactive math tutoring platform** that helps students practice problems, understand concepts, and track their progress.  
It solves the **real-world problem** of making math learning engaging and accessible by combining:
- Auto-generated quizzes across difficulty levels.
- Deterministic (SymPy-based) and AI-driven explanations.
- Built-in calculator, text-to-speech support, and notepad for rough work.
- Progress tracking with analytics and downloadable PDF markcards.

This addresses the lack of **personalized, interactive, and explainable math practice tools** for students in school, PU, and engineering levels.

---

## 2. Data Source
- **Source:** Data is not external. The project **generates problems dynamically** using Python functions (`random` and `SymPy`).  
- **Size:** Since questions are generated on the fly, the dataset is virtually infinite.  
- **Features of generated data:**
  - Arithmetic problems (Addition, Subtraction).
  - Algebraic problems (Quadratic equations: sum/product of roots, discriminant).
  - Metadata such as difficulty level, subject, user progress, explanations, and scores are stored in `progress.json`.

---

## 3. Methodologies Used
- **Dynamic Question Generation:** Randomized math problems to ensure variety in practice.  
- **Expression Parsing & Preprocessing:** Custom regex and SymPy parser for user-input normalization (e.g., handling `2x`, `^`, implicit multiplication).  
- **Step-by-Step Explanations:**  
  - Deterministic (using SymPy for symbolic manipulation and solving).  
  - AI-based (optional integration with Ollama Llama3 for conversational teaching style).  
- **Learning Aids:**  
  - Text-to-speech for auditory learners.  
  - Integrated calculator with symbolic and numeric evaluations.  
  - Rough notepad per question.  
- **Performance Tracking:** User progress saved in JSON and visualized with Pandas/Streamlit dashboards.  
- **Feedback & Assessment:** PDF markcards generated using `fpdf2`.

---

## 4. AI / ML Models Used
- **SymPy (Symbolic Mathematics):** Used for deterministic parsing, simplification, solving, factorization, and generating explanations.  
- **Ollama Llama3 (Optional):** Provides AI-driven, human-like explanations of solutions when enabled.  
- **Fallback Safe Evaluator:** For numeric-only inputs if SymPy is unavailable.

---

## 5. Predictions and Findings
### Predictions
- The system predicts **correctness of student answers** against the expected solution.  
- It evaluates:
  - Exact numeric values.
  - Algebraic simplifications and polynomial roots.
- Performance is quantified as **score (%)** per quiz attempt.

### Findings
- Students receive **immediate feedback** on correctness.  
- **Explanations** (step-by-step or AI conversational) help students understand mistakes.  
- **Progress tracking** reveals improvement trends across levels and subjects.  
- **PDF Markcards** allow offline review of performance.  
