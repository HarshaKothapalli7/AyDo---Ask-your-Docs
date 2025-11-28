# AyDo RAG System - Evaluation Module

This directory contains the evaluation infrastructure for measuring and analyzing the performance of the AyDo RAG system.

## Overview

The evaluation module provides:
- **Automated testing** against a predefined test dataset
- **Performance metrics** including latency, retrieval success, and routing accuracy
- **Manual answer quality scoring** framework
- **Detailed reporting** with per-query analysis

## Files Structure

```
evaluation/
├── EVALUATION_README.md        # This file
├── test_dataset.json           # Test questions and expected answers
├── evaluate_rag.py             # Main evaluation script
└── results/
    ├── results_YYYYMMDD_HHMMSS.json          # Raw evaluation data
    ├── evaluation_report_YYYYMMDD_HHMMSS.txt # Human-readable report
    └── figures/                              # Visualization outputs (future)
```

## Setup

### Prerequisites

1. **Backend must be running:**
   ```bash
   cd backend
   uvicorn main:app --reload
   ```

2. **Documents uploaded:**
   - Upload 2-3 PDF documents through the frontend
   - Note the document names for test dataset creation

3. **Python dependencies:**
   All required dependencies are already in `requirements.txt`

## Creating Your Test Dataset

### Step 1: Customize test_dataset.json

The provided `test_dataset.json` contains sample questions about diabetes. You need to replace these with questions from **your actual uploaded documents**.

1. Open `evaluation/test_dataset.json`
2. For each question (IDs 1-13), replace with your own:
   - `question`: Your actual question
   - `expected_answer`: The correct answer from your documents
   - `document_source`: Your actual PDF filename
3. Keep questions 14-15 as out-of-scope (or create similar ones)
4. Update questions 16-18 with current events relevant to your domain

### Step 2: Question Type Guidelines

**Factual (5 questions):**
- Direct lookup questions with specific answers
- Example: "What is X?", "When did Y happen?", "Who is Z?"

**Explanation (5 questions):**
- Questions requiring understanding and explanation
- Example: "How does X work?", "Why does Y occur?", "Explain the process of Z"

**Comparison (3 questions):**
- Questions about differences or relationships
- Example: "What's the difference between X and Y?", "Compare A and B"

**Out-of-scope (2-3 questions):**
- Questions NOT covered in your documents
- Tests fallback behavior and general knowledge
- Example: "What is quantum computing?" (if your docs are about medicine)

**Web-needed (2-3 questions):**
- Questions requiring current/real-time information
- Example: "What happened today in [your domain]?", "Latest news on X?"

### Step 3: Validate Your Dataset

```bash
# Quick validation
python -c "import json; print('Valid JSON' if json.load(open('evaluation/test_dataset.json')) else 'Invalid')"
```

## Running the Evaluation

### Basic Usage

```bash
# From project root
python evaluation/evaluate_rag.py
```

### What It Does

1. **Loads test dataset** from `test_dataset.json`
2. **Checks backend health** at `http://localhost:8000/health`
3. **Runs each query** against the `/chat/` endpoint
4. **Measures metrics:**
   - Response latency (average, median, min, max, P95, std dev)
   - Retrieval success rate (did RAG return chunks?)
   - Routing accuracy (did router choose correct path?)
5. **Generates reports:**
   - JSON file with raw data
   - Text file with human-readable analysis

### Expected Output

```
================================================================================
                     AyDo RAG System Evaluation
================================================================================

ℹ Loading test dataset from: evaluation/test_dataset.json
✓ Loaded 18 test questions
ℹ Checking backend health...
✓ Backend is healthy and ready

================================================================================
                          Running Queries
================================================================================

[1/18] Question ID 1
Type: factual
Q: What is diabetes?
Route: rag | RAG Retrieved: True | Latency: 2.34s
Answer: Diabetes is a chronic medical condition where the body cannot...
✓ Routing correct

[2/18] Question ID 2
...

================================================================================
                       Calculating Metrics
================================================================================

ℹ Latency Metrics:
  Average: 2.45s
  Median: 2.31s
  Min: 1.89s
  Max: 4.12s
  P95: 3.78s
  Std Dev: 0.54s

ℹ Retrieval Metrics:
  Success Rate: 92.31%
  Successful: 12/13

ℹ Routing Metrics:
  Accuracy: 94.44%
  Correct: 17/18

================================================================================
                        Generating Reports
================================================================================

✓ Raw results saved to: evaluation/results/results_20251125_143022.json
✓ Evaluation report saved to: evaluation/results/evaluation_report_20251125_143022.txt
```

## Manual Answer Quality Scoring

The evaluation script cannot automatically judge answer quality - this requires human review.

### Scoring Rubric

- **2 points (CORRECT)**: Answer is accurate, complete, and directly addresses the question
- **1 point (PARTIALLY CORRECT)**: Answer is relevant but incomplete or slightly incorrect
- **0 points (INCORRECT)**: Answer is wrong, irrelevant, or hallucinates information

### How to Score

1. **Review the report:**
   ```bash
   cat evaluation/results/evaluation_report_YYYYMMDD_HHMMSS.txt
   ```

2. **For each query, compare:**
   - Expected Answer (from your test dataset)
   - Actual Answer (from the system)

3. **Assign score** (0, 1, or 2) based on rubric

4. **Update JSON results:**
   ```bash
   # Open the results JSON file
   code evaluation/results/results_YYYYMMDD_HHMMSS.json

   # Find each query result and update:
   "answer_quality_score": 2  # Change from null to your score
   ```

5. **Calculate final percentage:**
   ```python
   # In Python or manually
   total_points = sum_of_all_scores
   max_possible = num_questions * 2
   quality_percentage = (total_points / max_possible) * 100
   ```

### Example Scoring

**Question:** "What is diabetes?"

**Expected:** "Diabetes is a chronic condition where the body cannot properly process blood glucose..."

**Actual:** "Diabetes is a disease affecting blood sugar regulation due to insufficient insulin production or insulin resistance."

**Score:** 2 (Correct - covers key concepts accurately)

---

**Question:** "What are the symptoms of Type 2 diabetes?"

**Expected:** "Increased thirst, frequent urination, increased hunger, fatigue, blurred vision..."

**Actual:** "Common symptoms include feeling thirsty and tired."

**Score:** 1 (Partially Correct - mentions some but incomplete)

---

**Question:** "When was insulin discovered?"

**Expected:** "1921 by Banting and Best"

**Actual:** "Insulin was discovered in the early 1900s by researchers in Canada."

**Score:** 1 (Partially Correct - right timeframe and location, but missing specific year and names)

## Understanding the Metrics

### Response Latency

Measures how fast the system responds to queries.

- **Average**: Typical response time
- **Median**: Middle value (less affected by outliers)
- **P95**: 95% of queries complete within this time
- **Target**: < 3 seconds average for good user experience

### Retrieval Success Rate

Percentage of queries where RAG successfully retrieved relevant chunks.

- **Formula**: `(successful_retrievals / total_attempts) * 100`
- **Target**: > 80% for good RAG coverage
- **Low rate indicates**: Documents may not cover query topics, or chunking strategy needs adjustment

### Routing Accuracy

Percentage of queries where the router made the correct decision.

- **Formula**: `(correct_routes / total_queries) * 100`
- **Target**: > 90% for effective routing
- **Measures**: RAG vs Web vs Direct answer decisions

### Answer Quality

Human-evaluated accuracy and completeness of answers.

- **Formula**: `(total_points / (num_questions * 2)) * 100`
- **Target**: > 70% for acceptable quality, > 85% for excellent
- **Requires**: Manual scoring by domain expert

## Results for Your IEEE Paper

### Section 8: Experimental Setup

Use this format:

```
We evaluated our system using a test dataset of [N] question-answer pairs
derived from [X] documents covering [domain]. The test set included:
- [N] factual lookup questions
- [N] explanation questions
- [N] comparison questions
- [N] out-of-scope questions
- [N] web-required questions

Evaluation was conducted on [date] with the backend running locally.
Metrics measured include response latency, retrieval success rate,
routing accuracy, and answer quality.
```

### Section 9: Results

Include these tables:

**Table 1: Overall Performance Metrics**
| Metric | Value |
|--------|-------|
| Average Latency | X.XXs |
| Retrieval Success Rate | XX.X% |
| Routing Accuracy | XX.X% |
| Answer Quality Score | XX.X% |

**Table 2: Performance by Query Type**
| Query Type | Count | Avg Latency | RAG Success | Routing Accuracy |
|------------|-------|-------------|-------------|------------------|
| Factual | X | X.XXs | XX.X% | XX.X% |
| Explanation | X | X.XXs | XX.X% | XX.X% |
| Comparison | X | X.XXs | XX.X% | XX.X% |
| Out-of-scope | X | X.XXs | XX.X% | XX.X% |
| Web-needed | X | X.XXs | XX.X% | XX.X% |

Copy data directly from `evaluation_report_*.txt`

## Troubleshooting

### Backend Connection Error

```
✗ Cannot connect to backend. Is it running on http://localhost:8000?
```

**Solution:** Start the backend:
```bash
cd backend
uvicorn main:app --reload
```

### No Results for RAG Questions

**Possible causes:**
1. No documents uploaded to the system
2. Questions don't match document content
3. Embedding/retrieval issues

**Solution:** Upload relevant documents and verify questions match content

### All Queries Timing Out

**Possible causes:**
1. LLM API key issues
2. Vector database connection issues
3. Backend overloaded

**Solution:** Check backend logs, verify API keys in `.env`

## Next Steps

1. **Run baseline evaluation** with your customized dataset
2. **Score answer quality** manually
3. **Document results** in your IEEE paper
4. **(Optional)** Run ablation studies (see parent project plan)
5. **(Optional)** Create visualizations (see parent project plan)

## Need Help?

- Check `backend/main.py` for API endpoint details
- Review `backend/agent.py` for routing logic
- Examine trace events in results JSON for debugging

---

