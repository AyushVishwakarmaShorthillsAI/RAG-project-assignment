import torch
import pytest
import time
import logging
import os
import sys
import numpy as np
import importlib.util
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from evaluate import load

# Add project root to PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'logs'))
os.makedirs(log_dir, exist_ok=True)

# Import from app module
from app.rag_pipeline import RAGPipeline
from app.llm import OllamaLLM
from app.vector_store import FAISSVectorStore

# Log file paths
qa_test_log_path = os.path.join(log_dir, 'qa_testing.log')
main_log_path = os.path.join(log_dir, 'rag_project.log')

# General logger
main_handler = logging.FileHandler(main_log_path)
main_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(main_handler)

# QA interaction logger
qa_test_logger = logging.getLogger('qa_test_interactions')
qa_test_handler = logging.FileHandler(qa_test_log_path)
qa_test_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
qa_test_logger.addHandler(qa_test_handler)
qa_test_logger.setLevel(logging.INFO)

# Fixture to set up RAG pipeline
@pytest.fixture(scope="session")
def rag_pipeline():
    logger.info("Setting up RAG pipeline...")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedding_model = embedding_model.to(device)

    index_path = os.path.join("..", "app", "faiss_index.bin")
    texts_path = os.path.join("..", "app", "texts.pkl")

    if os.path.exists(index_path) and os.path.exists(texts_path):
        vector_store = FAISSVectorStore.load(embedding_model, index_path, texts_path)
        logger.info("Loaded existing FAISS index and texts.")
    else:
        raise FileNotFoundError("FAISS index or texts file not found. Run scraping and indexing first.")

    llm = OllamaLLM()
    pipeline = RAGPipeline(vector_store, llm, embedding_model)
    return pipeline

# Fixture to load test cases
@pytest.fixture(scope="session")
def test_cases():
    logger.info("Loading test cases from all_test_cases.py...")
    spec = importlib.util.spec_from_file_location(
        "all_test_cases",
        os.path.join("..", "test_cases", "all_test_cases.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.test_cases


# Initialize metrics collectors
results = []
pass_count = 0
fail_count = 0
response_times = []
cosine_similarities = []
bleu_scores = []
rouge_scores = []
extracted_scores = []

# Load evaluation metrics
bleu = load("bleu")
rouge = load("rouge")

# File paths for saving results
INTERMEDIATE_RESULTS_FILE = os.path.join("..", "evaluations", "intermediate_test_results.json")
EXTRACTED_SCORES_FILE = os.path.join("..", "evaluations", "extracted_score.json")


# Test function to evaluate each test case
def test_rag_response(rag_pipeline, test_cases):
    global pass_count, fail_count, results, extracted_scores

    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    total_cases = len(test_cases)

    # Initialize intermediate results file
    with open(INTERMEDIATE_RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump([], f)

    for idx, test_case in enumerate(test_cases, 1):
        question = test_case["Question"]
        expected_answer = test_case["Answer"]
        context = test_case.get("Context", "")

        logger.info(f"Testing question: {question}")
        logger.info(f"Progress: Processed {idx}/{total_cases} test cases ({(idx/total_cases)*100:.2f}%)")

        qa_test_logger.info(f"Question: {question}")
        qa_test_logger.info(f"Expected Answer: {expected_answer}")

        start_time = time.time()
        response = rag_pipeline.process(question)
        duration = time.time() - start_time
        response_times.append(duration)
        logger.info(f"Response time: {duration:.2f} seconds")

        logger.info(f"Generated Response: {response}")
        qa_test_logger.info(f"Generated Response: {response}")

        test_result = {
            "question": question,
            "expected_answer": expected_answer,
            "generated_response": response,
            "context": context,
            "response_time": duration
        }

        exact_match = expected_answer.lower() in response.lower()
        test_result["exact_match"] = exact_match

        response_embedding = embedder.encode(response)
        expected_embedding = embedder.encode(expected_answer)
        similarity = cosine_similarity([response_embedding], [expected_embedding])[0][0]
        cosine_similarities.append(similarity)
        test_result["cosine_similarity"] = float(similarity)
        logger.info(f"Cosine similarity: {similarity:.4f}")

        bleu_score = bleu.compute(predictions=[response], references=[[expected_answer]])["bleu"]
        bleu_scores.append(bleu_score)
        test_result["bleu_score"] = float(bleu_score)
        logger.info(f"BLEU score: {bleu_score:.4f}")

        rouge_score = rouge.compute(predictions=[response], references=[expected_answer])["rouge1"]
        rouge_scores.append(rouge_score)
        test_result["rouge1_score"] = float(rouge_score)
        logger.info(f"ROUGE-1 score: {rouge_score:.4f}")

        passed = exact_match or similarity > 0.7
        test_result["passed"] = str(passed).lower()
        if passed:
            pass_count += 1
        else:
            fail_count += 1

        results.append(test_result)
        extracted_scores.append({
            "question": question,
            "expected_answer": expected_answer,
            "generated_response": response,
            "cosine_similarity": float(similarity),
            "bleu_score": float(bleu_score),
            "rouge1_score": float(rouge_score),
            "response_time": duration,
            "passed": passed
        })

        with open(INTERMEDIATE_RESULTS_FILE, "r+", encoding="utf-8") as f:
            try:
                intermediate_data = json.load(f)
            except json.JSONDecodeError:
                intermediate_data = []
            intermediate_data.append(test_result)
            f.seek(0)
            f.truncate()
            json.dump(intermediate_data, f, indent=2)

    evaluation_summary = {
        "total_tests": len(test_cases),
        "pass_count": pass_count,
        "fail_count": fail_count,
        "pass_rate": float(pass_count / len(test_cases)) if test_cases else 0,
        "average_response_time": float(np.mean(response_times)) if response_times else 0,
        "average_cosine_similarity": float(np.mean(cosine_similarities)) if cosine_similarities else 0,
        "average_bleu_score": float(np.mean(bleu_scores)) if bleu_scores else 0,
        "average_rouge1_score": float(np.mean(rouge_scores)) if rouge_scores else 0,
        "detailed_results": results
    }

    eval_summary_path = os.path.join("..", "evaluations", "evaluation_results_2.json")
    with open(eval_summary_path, "w", encoding="utf-8") as f:
        json.dump(evaluation_summary, f, indent=2)
    logger.info(f"Evaluation results saved to {eval_summary_path}")

    with open(EXTRACTED_SCORES_FILE, "w", encoding="utf-8") as f:
        json.dump(extracted_scores, f, indent=2)
    logger.info(f"Extracted scores saved to {EXTRACTED_SCORES_FILE}")

    assert evaluation_summary["pass_rate"] >= 0.8, f"Pass rate too low: {evaluation_summary['pass_rate']:.2%}"


if __name__ == "__main__":
    spec = importlib.util.spec_from_file_location(
        "all_test_cases",
        os.path.join("..", "test_cases", "all_test_cases.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    test_data = module.test_cases

    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedding_model = embedding_model.to(device)

    index_path = os.path.join("..", "app", "faiss_index.bin")
    texts_path = os.path.join("..", "app", "texts.pkl")

    if os.path.exists(index_path) and os.path.exists(texts_path):
        vector_store = FAISSVectorStore.load(embedding_model, index_path, texts_path)
        llm = OllamaLLM()
        pipeline = RAGPipeline(vector_store, llm, embedding_model)
        test_rag_response(pipeline, test_data)
    else:
        logger.error("FAISS index or texts file not found. Run scraping and indexing first.")