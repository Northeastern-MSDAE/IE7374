


# Generative Math Tutor: Curriculum-Aligned Word Problem Generator

This project develops an AI-powered math tutor that generates curriculum-aligned word problems and provides step-by-step solutions using a chain-of-thought reasoning approach. It supports adaptive learning by dynamically tailoring problems to a student's grade level and difficulty preferences.

## Project Goals

- Generate novel, curriculum-aligned math word problems (Grades 3–6)
- Solve each problem using chain-of-thought (CoT) reasoning.
- Tag problems with appropriate grade levels and math topics using NLP
- Provide an interactive CLI/web interface for learners or educators.
- Evaluate the correctness, clarity, and alignment of generated content.

---

## Features

- **LLM-Powered Generation**: Uses GPT-3.5 for problem and solution generation with few-shot CoT prompts.
- **Curriculum Tagging**: Uses `spaCy` + rule-based mapping to tag generated problems with Common Core grade levels and math concepts.
- **Evaluation Framework**:
  - Math correctness via `SymPy`
  - Linguistic coherence (grammar checks)
  - Curriculum alignment QA
- **CLI/Web Interface**: Interactive tool for generating and viewing problems by grade and topic.

---

## Repository Structure

```bash
math-tutor/
│
├── data/
│  ├── raw/            # Sample problems from MAWPS, ASDiv-A
│  └── processed/      # Normalized and tagged data
│
├── prompts/
│  └── prompt_templates.txt # Few-shot examples for GPT-3.5
│
├── src/
│  ├── generator.py     # GPT-based problem + solution generator
│  ├── tagger.py        # NLP + rule-based curriculum tagger
│  ├── evaluator.py     # SymPy-based correctness evaluator
│  ├── interface.py     # CLI or Streamlit app for users
│  └── utils.py         # Shared utilities and helpers
│
├── tests/
│  └── unit/               # Test cases for tagging and evaluation
│
├── notebooks/
│  └── exploration.ipynb   # Early analysis and CoT testing
│
├── requirements.txt       # Python dependencies
└── README.md

```

---

## Milestones

| Milestone                  | Description                         | Due Date |
| -------------------------- | ----------------------------------- | -------- |
| Group Formation & Proposal | Define idea, roles, scope           | May 26   |
| Final Project Proposal     | Architecture + plan                 | Jun 30   |
| Milestone 2                | Data pipeline: tagging, schema, NLP | Jul 14   |
| Milestone 3                | Model generation + evaluation       | Jul 21   |
| Final Submission           | Complete pipeline + report          | Aug 4    |

---

## Setup Instructions

**Clone the repo**

```bash
git clone https://github.com/your-username/math-tutor.git
cd math-tutor
```

**Install dependencies**

```bash
pip install -r requirements.txt
```

**Set up OpenAI credentials**

```bash
export OPENAI_API_KEY=your_key_here
```

**Run the CLI tool**

```bash
python src/interface.py
```

------


## ** Sample Workflow**

```bash
# Generate a problem for Grade 4
python src/generator.py --grade 4 --topic "multiplication"

# Tag the generated problem
python src/tagger.py generated_output.json

# Evaluate solution correctness
python src/evaluator.py generated_output.json
```

------


## **Team Roles**



| **Name** | **Role**                                   |
| -------- | ------------------------------------------ |
| Member 1 | Curriculum Tagging (spaCy + rules), QA     |
| Member 2 | Prompt Design, Chain-of-Thought Generation |
| Member 3 | Evaluation (SymPy, Grammar, Tagging)       |
| Member 4 | UI Development & Integration               |



## **Datasets Used**


- [MAWPS](https://huggingface.co/datasets/MU-NLPC/Calc-mawps)

- [ASDiv-A](https://huggingface.co/datasets/MU-NLPC/Calc-asdiv_a)

- [MathQA](https://huggingface.co/datasets/allenai/math_qa)

  

## **Tools & Libraries**

- [OpenAI GPT-3.5](https://platform.openai.com/)
- [spaCy](https://spacy.io/)
- [SymPy](https://www.sympy.org/)
- [Streamlit](https://streamlit.io/) or CLI for UI

------

## **Evaluation Metrics**

- **Correctness** – Symbolic validation using SymPy
- **Reasoning Clarity** – Chain-of-thought coherence
- **Curriculum Alignment** – Tag accuracy vs. expected grade/topic
- **Grammar/Clarity** – Language checks using heuristics or models



## **License**

This project is for educational use only under university course guidelines.



## **Contact**

If you have questions or suggestions, feel free to contact us via GitHub issues
