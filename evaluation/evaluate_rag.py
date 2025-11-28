"""
AyDo RAG System Evaluation Script

This script evaluates the AyDo RAG system's performance by:
1. Running test queries against the backend API
2. Measuring response latency
3. Evaluating retrieval success rate
4. Checking routing accuracy
5. Supporting manual answer quality scoring

Usage:
    python evaluation/evaluate_rag.py
"""

import json
import time
import requests
import statistics
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple
import uuid

# Configuration
BACKEND_URL = "http://localhost:8000"
TEST_DATASET_PATH = Path(__file__).parent / "test_dataset.json"
RESULTS_DIR = Path(__file__).parent / "results"
SESSION_ID = f"eval_session_{uuid.uuid4().hex[:8]}"

# Ensure results directory exists
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


class Colors:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_section(title: str):
    """Print a formatted section header"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{title.center(80)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}\n")


def print_success(message: str):
    """Print success message"""
    print(f"{Colors.OKGREEN}✓ {message}{Colors.ENDC}")


def print_error(message: str):
    """Print error message"""
    print(f"{Colors.FAIL}✗ {message}{Colors.ENDC}")


def print_info(message: str):
    """Print info message"""
    print(f"{Colors.OKCYAN}ℹ {message}{Colors.ENDC}")


def print_warning(message: str):
    """Print warning message"""
    print(f"{Colors.WARNING}⚠ {message}{Colors.ENDC}")


def load_test_dataset() -> Dict[str, Any]:
    """
    Load the test dataset from JSON file.

    Returns:
        Dictionary containing test questions and metadata
    """
    print_info(f"Loading test dataset from: {TEST_DATASET_PATH}")

    try:
        with open(TEST_DATASET_PATH, 'r', encoding='utf-8') as f:
            dataset = json.load(f)

        num_questions = len(dataset.get('questions', []))
        print_success(f"Loaded {num_questions} test questions")

        return dataset
    except FileNotFoundError:
        print_error(f"Test dataset not found at: {TEST_DATASET_PATH}")
        raise
    except json.JSONDecodeError as e:
        print_error(f"Invalid JSON in test dataset: {e}")
        raise


def check_backend_health() -> bool:
    """
    Check if the FastAPI backend is running and healthy.

    Returns:
        True if backend is healthy, False otherwise
    """
    print_info("Checking backend health...")

    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        if response.status_code == 200:
            print_success("Backend is healthy and ready")
            return True
        else:
            print_error(f"Backend returned status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print_error("Cannot connect to backend. Is it running on http://localhost:8000?")
        return False
    except requests.exceptions.Timeout:
        print_error("Backend health check timed out")
        return False


def run_query(question: str, enable_web_search: bool = True) -> Tuple[Dict[str, Any], float, bool]:
    """
    Run a single query against the backend API and measure latency.

    Args:
        question: The question to ask
        enable_web_search: Whether to enable web search for this query

    Returns:
        Tuple of (response_data, latency_seconds, success)
    """
    url = f"{BACKEND_URL}/chat/"
    payload = {
        "session_id": SESSION_ID,
        "query": question,
        "enable_web_search": enable_web_search
    }

    start_time = time.time()

    try:
        response = requests.post(url, json=payload, timeout=60)
        latency = time.time() - start_time

        if response.status_code == 200:
            return response.json(), latency, True
        else:
            print_error(f"API returned status code: {response.status_code}")
            return {}, latency, False

    except requests.exceptions.Timeout:
        latency = time.time() - start_time
        print_error("Query timed out")
        return {}, latency, False
    except requests.exceptions.RequestException as e:
        latency = time.time() - start_time
        print_error(f"Request failed: {e}")
        return {}, latency, False


def evaluate_retrieval(trace_events: List[Dict[str, Any]]) -> Tuple[bool, str]:
    """
    Evaluate if RAG retrieval was successful by analyzing trace events.

    Args:
        trace_events: List of trace events from the agent

    Returns:
        Tuple of (success, route_taken)
    """
    rag_retrieved = False
    route_taken = "unknown"

    for event in trace_events:
        node_name = event.get('node_name', '')
        event_type = event.get('event_type', '')

        # Check if router made a decision
        if node_name == 'router' and event_type == 'router_decision':
            details = event.get('details', {})
            route_taken = details.get('decision', details.get('final_decision', 'unknown'))

        # Check if RAG lookup was performed and returned content
        if node_name == 'rag_lookup' and event_type == 'rag_action':
            details = event.get('details', {})
            retrieved_content = details.get('retrieved_content_summary', '')

            # Consider retrieval successful if non-empty content was retrieved
            if retrieved_content and len(retrieved_content.strip()) > 0:
                rag_retrieved = True

    return rag_retrieved, route_taken


def check_routing_accuracy(expected_rag: bool, expected_web: bool, route_taken: str) -> bool:
    """
    Check if the routing decision was correct.

    Args:
        expected_rag: Whether RAG should be used
        expected_web: Whether web search should be used
        route_taken: The actual route taken by the agent

    Returns:
        True if routing was correct, False otherwise
    """
    # For RAG questions, we expect route to start with "rag"
    if expected_rag and not expected_web:
        return route_taken == "rag"

    # For web questions, we expect route to be "web"
    elif expected_web and not expected_rag:
        return route_taken == "web"

    # For out-of-scope, RAG should be tried first (then may fallback)
    # This is considered correct routing
    elif expected_rag and expected_web:
        return route_taken in ["rag", "web"]

    return True


def calculate_percentile(data: List[float], percentile: int) -> float:
    """
    Calculate the nth percentile of a list of numbers.

    Args:
        data: List of numeric values
        percentile: The percentile to calculate (0-100)

    Returns:
        The percentile value
    """
    if not data:
        return 0.0

    sorted_data = sorted(data)
    index = (percentile / 100) * (len(sorted_data) - 1)

    if index.is_integer():
        return sorted_data[int(index)]
    else:
        lower = sorted_data[int(index)]
        upper = sorted_data[int(index) + 1]
        return lower + (upper - lower) * (index - int(index))


def run_evaluation() -> Dict[str, Any]:
    """
    Run the complete evaluation pipeline.

    Returns:
        Dictionary containing all evaluation results
    """
    print_section("AyDo RAG System Evaluation")

    # Load test dataset
    dataset = load_test_dataset()
    questions = dataset.get('questions', [])

    # Check backend health
    if not check_backend_health():
        print_error("Backend is not available. Please start it before running evaluation.")
        return {}

    print_section("Running Queries")

    results = {
        "metadata": {
            "evaluation_date": datetime.now().isoformat(),
            "backend_url": BACKEND_URL,
            "session_id": SESSION_ID,
            "total_questions": len(questions)
        },
        "query_results": [],
        "metrics": {}
    }

    latencies = []
    retrieval_successes = 0
    retrieval_attempts = 0
    routing_correct = 0
    total_queries = len(questions)

    for idx, question_data in enumerate(questions, 1):
        question_id = question_data.get('id')
        question = question_data.get('question')
        query_type = question_data.get('query_type')
        should_use_web = question_data.get('should_use_web', True)

        print(f"\n{Colors.BOLD}[{idx}/{total_queries}] Question ID {question_id}{Colors.ENDC}")
        print(f"Type: {query_type}")
        print(f"Q: {question}")

        # Run the query
        response_data, latency, success = run_query(question, enable_web_search=should_use_web)

        if not success:
            print_error(f"Query failed (latency: {latency:.2f}s)")
            results["query_results"].append({
                "question_id": question_id,
                "question": question,
                "query_type": query_type,
                "success": False,
                "latency": latency,
                "error": "API request failed"
            })
            continue

        # Extract response and trace events
        answer = response_data.get('response', '')
        trace_events = response_data.get('trace_events', [])

        # Evaluate retrieval
        rag_retrieved, route_taken = evaluate_retrieval(trace_events)

        # Check if RAG was attempted for this query type
        should_use_rag = question_data.get('should_use_rag', True)

        if should_use_rag:
            retrieval_attempts += 1
            if rag_retrieved:
                retrieval_successes += 1

        # Check routing accuracy
        expected_rag = question_data.get('should_use_rag', True)
        expected_web = question_data.get('should_use_web', False)
        routing_is_correct = check_routing_accuracy(expected_rag, expected_web, route_taken)

        if routing_is_correct:
            routing_correct += 1

        # Store results
        query_result = {
            "question_id": question_id,
            "question": question,
            "query_type": query_type,
            "expected_answer": question_data.get('expected_answer', ''),
            "actual_answer": answer,
            "latency": round(latency, 3),
            "success": True,
            "route_taken": route_taken,
            "rag_retrieved": rag_retrieved,
            "routing_correct": routing_is_correct,
            "trace_events": trace_events,
            "answer_quality_score": None  # To be filled manually
        }

        results["query_results"].append(query_result)
        latencies.append(latency)

        # Print summary
        print(f"Route: {route_taken} | RAG Retrieved: {rag_retrieved} | Latency: {latency:.2f}s")
        print(f"Answer: {answer[:150]}{'...' if len(answer) > 150 else ''}")

        if routing_is_correct:
            print_success("Routing correct")
        else:
            print_warning(f"Routing incorrect (expected RAG={expected_rag}, Web={expected_web})")

    # Calculate metrics
    print_section("Calculating Metrics")

    if latencies:
        metrics = {
            "latency": {
                "average": round(statistics.mean(latencies), 3),
                "median": round(statistics.median(latencies), 3),
                "min": round(min(latencies), 3),
                "max": round(max(latencies), 3),
                "p95": round(calculate_percentile(latencies, 95), 3),
                "stddev": round(statistics.stdev(latencies), 3) if len(latencies) > 1 else 0
            },
            "retrieval": {
                "success_rate": round((retrieval_successes / retrieval_attempts * 100), 2) if retrieval_attempts > 0 else 0,
                "successful_retrievals": retrieval_successes,
                "total_attempts": retrieval_attempts
            },
            "routing": {
                "accuracy": round((routing_correct / total_queries * 100), 2),
                "correct_routes": routing_correct,
                "total_queries": total_queries
            },
            "answer_quality": {
                "note": "Manual scoring required. Use the scoring rubric in the report.",
                "rubric": {
                    "correct": 2,
                    "partially_correct": 1,
                    "incorrect": 0
                }
            }
        }

        results["metrics"] = metrics

        # Print metrics
        print_info("Latency Metrics:")
        print(f"  Average: {metrics['latency']['average']}s")
        print(f"  Median: {metrics['latency']['median']}s")
        print(f"  Min: {metrics['latency']['min']}s")
        print(f"  Max: {metrics['latency']['max']}s")
        print(f"  P95: {metrics['latency']['p95']}s")
        print(f"  Std Dev: {metrics['latency']['stddev']}s")

        print_info("\nRetrieval Metrics:")
        print(f"  Success Rate: {metrics['retrieval']['success_rate']}%")
        print(f"  Successful: {metrics['retrieval']['successful_retrievals']}/{metrics['retrieval']['total_attempts']}")

        print_info("\nRouting Metrics:")
        print(f"  Accuracy: {metrics['routing']['accuracy']}%")
        print(f"  Correct: {metrics['routing']['correct_routes']}/{metrics['routing']['total_queries']}")

        print_warning("\n⚠ Answer Quality Scoring:")
        print("  Manual scoring required - see evaluation_report.txt for instructions")

    return results


def generate_report(results: Dict[str, Any]):
    """
    Generate human-readable evaluation report.

    Args:
        results: The evaluation results dictionary
    """
    print_section("Generating Reports")

    # Save raw JSON results
    results_json_path = RESULTS_DIR / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_json_path, 'w', encoding='utf-8') as f:
        json.dump(results, indent=2, fp=f)

    print_success(f"Raw results saved to: {results_json_path}")

    # Generate text report
    report_path = RESULTS_DIR / f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("AyDo RAG System - Evaluation Report\n")
        f.write("="*80 + "\n\n")

        # Metadata
        metadata = results.get('metadata', {})
        f.write("EVALUATION METADATA\n")
        f.write("-"*80 + "\n")
        f.write(f"Date: {metadata.get('evaluation_date', 'N/A')}\n")
        f.write(f"Backend URL: {metadata.get('backend_url', 'N/A')}\n")
        f.write(f"Session ID: {metadata.get('session_id', 'N/A')}\n")
        f.write(f"Total Questions: {metadata.get('total_questions', 0)}\n\n")

        # Overall Metrics
        metrics = results.get('metrics', {})
        f.write("\n" + "="*80 + "\n")
        f.write("OVERALL PERFORMANCE METRICS\n")
        f.write("="*80 + "\n\n")

        # Latency
        latency = metrics.get('latency', {})
        f.write("Latency Metrics:\n")
        f.write("-"*80 + "\n")
        f.write(f"  Average Response Time: {latency.get('average', 0)}s\n")
        f.write(f"  Median Response Time: {latency.get('median', 0)}s\n")
        f.write(f"  Min Response Time: {latency.get('min', 0)}s\n")
        f.write(f"  Max Response Time: {latency.get('max', 0)}s\n")
        f.write(f"  P95 Response Time: {latency.get('p95', 0)}s\n")
        f.write(f"  Standard Deviation: {latency.get('stddev', 0)}s\n\n")

        # Retrieval
        retrieval = metrics.get('retrieval', {})
        f.write("Retrieval Metrics:\n")
        f.write("-"*80 + "\n")
        f.write(f"  Success Rate: {retrieval.get('success_rate', 0)}%\n")
        f.write(f"  Successful Retrievals: {retrieval.get('successful_retrievals', 0)}\n")
        f.write(f"  Total Attempts: {retrieval.get('total_attempts', 0)}\n\n")

        # Routing
        routing = metrics.get('routing', {})
        f.write("Routing Metrics:\n")
        f.write("-"*80 + "\n")
        f.write(f"  Accuracy: {routing.get('accuracy', 0)}%\n")
        f.write(f"  Correct Routes: {routing.get('correct_routes', 0)}\n")
        f.write(f"  Total Queries: {routing.get('total_queries', 0)}\n\n")

        # Answer Quality Instructions
        f.write("\n" + "="*80 + "\n")
        f.write("ANSWER QUALITY EVALUATION (MANUAL SCORING REQUIRED)\n")
        f.write("="*80 + "\n\n")
        f.write("SCORING RUBRIC:\n")
        f.write("-"*80 + "\n")
        f.write("  2 points (CORRECT): Answer is accurate, complete, and directly addresses the question\n")
        f.write("  1 point (PARTIALLY CORRECT): Answer is relevant but incomplete or slightly incorrect\n")
        f.write("  0 points (INCORRECT): Answer is wrong, irrelevant, or hallucinates information\n\n")
        f.write("INSTRUCTIONS:\n")
        f.write("  1. Review each query result below\n")
        f.write("  2. Compare 'Actual Answer' with 'Expected Answer'\n")
        f.write("  3. Assign a score (0, 1, or 2) based on the rubric\n")
        f.write("  4. Update the 'answer_quality_score' field in the JSON results file\n")
        f.write("  5. Re-run report generation to see updated quality metrics\n\n")

        # Per-Query Results
        f.write("\n" + "="*80 + "\n")
        f.write("PER-QUERY RESULTS\n")
        f.write("="*80 + "\n\n")

        query_results = results.get('query_results', [])
        for result in query_results:
            f.write("-"*80 + "\n")
            f.write(f"Question ID: {result.get('question_id')}\n")
            f.write(f"Type: {result.get('query_type')}\n")
            f.write(f"Question: {result.get('question')}\n\n")

            f.write(f"Expected Answer:\n{result.get('expected_answer', 'N/A')}\n\n")
            f.write(f"Actual Answer:\n{result.get('actual_answer', 'N/A')}\n\n")

            f.write(f"Latency: {result.get('latency', 0)}s\n")
            f.write(f"Route Taken: {result.get('route_taken', 'unknown')}\n")
            f.write(f"RAG Retrieved: {result.get('rag_retrieved', False)}\n")
            f.write(f"Routing Correct: {result.get('routing_correct', False)}\n")
            f.write(f"Answer Quality Score: {result.get('answer_quality_score', 'NOT SCORED')}\n\n")

        # Summary Statistics by Query Type
        f.write("\n" + "="*80 + "\n")
        f.write("PERFORMANCE BY QUERY TYPE\n")
        f.write("="*80 + "\n\n")

        query_types = {}
        for result in query_results:
            qtype = result.get('query_type', 'unknown')
            if qtype not in query_types:
                query_types[qtype] = {
                    'count': 0,
                    'total_latency': 0,
                    'rag_retrieved': 0,
                    'routing_correct': 0
                }

            query_types[qtype]['count'] += 1
            query_types[qtype]['total_latency'] += result.get('latency', 0)
            query_types[qtype]['rag_retrieved'] += 1 if result.get('rag_retrieved') else 0
            query_types[qtype]['routing_correct'] += 1 if result.get('routing_correct') else 0

        for qtype, stats in query_types.items():
            f.write(f"\n{qtype.upper()}:\n")
            f.write(f"  Count: {stats['count']}\n")
            f.write(f"  Avg Latency: {stats['total_latency'] / stats['count']:.3f}s\n")
            f.write(f"  RAG Success: {stats['rag_retrieved']}/{stats['count']}\n")
            f.write(f"  Routing Accuracy: {stats['routing_correct']}/{stats['count']}\n")

        f.write("\n" + "="*80 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*80 + "\n")

    print_success(f"Evaluation report saved to: {report_path}")

    # Print summary
    print_section("Evaluation Complete!")
    print(f"Results saved to: {results_json_path}")
    print(f" Report saved to: {report_path}")
    print(f"\n{Colors.WARNING}Next Steps:{Colors.ENDC}")
    print(f"  1. Review the report: {report_path}")
    print(f"  2. Manually score answer quality for each query")
    print(f"  3. Update 'answer_quality_score' in: {results_json_path}")
    print(f"  4. Calculate final answer quality percentage")


def main():
    """Main entry point for the evaluation script"""
    try:
        results = run_evaluation()

        if results:
            generate_report(results)
        else:
            print_error("Evaluation failed - no results to report")
            return 1

        return 0

    except KeyboardInterrupt:
        print_warning("\n\nEvaluation interrupted by user")
        return 1
    except Exception as e:
        print_error(f"Evaluation failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
