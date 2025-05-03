import torch
import pytest
import time
import logging
import os
from main import RAGPipeline, OllamaLLM, FAISSVectorStore, SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from evaluate import load
import numpy as np
import importlib.util
import json

# Setup paths
EVALUATION_DIR = os.path.join(os.path.dirname(__file__), '..', 'evaluations')
LOG_DIR = os.path.join(os.path.dirname(__file__), '..', 'logs')
TEST_CASES_PATH = os.path.join(os.path.dirname(__file__), '..', 'test_cases', 'all_test_cases.py')

os.makedirs(EVALUATION_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

qa_test_logger = logging.getLogger('qa_test_interactions')
qa_test_handler = logging.FileHandler(os.path.join(LOG_DIR, 'qa_testing.log'))
qa_test_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
qa_test_logger.addHandler(qa_test_handler)
qa_test_logger.setLevel(logging.INFO)

@pytest.fixture(scope="session")
def rag_pipeline():
    logger.info("Setting up RAG pipeline...")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedding_model = embedding_model.to(device)

    index_path = "faiss_index.bin"
    texts_path = "texts.pkl"

    if os.path.exists(index_path) and os.path.exists(texts_path):
        vector_store = FAISSVectorStore.load(embedding_model, index_path, texts_path)
        logger.info("Loaded existing FAISS index and texts.")
    else:
        raise FileNotFoundError("FAISS index or texts file not found.")

    llm = OllamaLLM()
    return RAGPipeline(vector_store, llm, embedding_model)

@pytest.fixture(scope="session")
def test_cases():
    logger.info("Loading test cases from all_test_cases.py...")
    spec = importlib.util.spec_from_file_location("all_test_cases", TEST_CASES_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.test_cases

# Metrics setup
results = []
pass_count = 0
fail_count = 0
response_times = []
cosine_similarities = []
bleu_scores = []
rouge_scores = []
extracted_scores = []

bleu = load("bleu")
rouge = load("rouge")

INTERMEDIATE_RESULTS_FILE = os.path.join(EVALUATION_DIR, "intermediate_test_results.json")
EXTRACTED_SCORES_FILE = os.path.join(EVALUATION_DIR, "weird_test_score.json")
EVALUATION_RESULTS_FILE = os.path.join(EVALUATION_DIR, "evaluation_results_weird.json")

def test_rag_response(rag_pipeline, test_cases):
    global pass_count, fail_count, results, extracted_scores

    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    total_cases = len(test_cases)

    with open(INTERMEDIATE_RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump([], f)

    for idx, test_case in enumerate(test_cases, 1):
        question = test_case["Question"]
        expected_answer = test_case["Answer"]
        context = test_case.get("Context", "")

        logger.info(f"Testing question: {question} [{idx}/{total_cases}]")

        qa_test_logger.info(f"Question: {question}")
        qa_test_logger.info(f"Expected Answer: {expected_answer}")

        start_time = time.time()
        response = rag_pipeline.process(question)
        duration = time.time() - start_time
        response_times.append(duration)

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

        bleu_score = bleu.compute(predictions=[response], references=[[expected_answer]])["bleu"]
        bleu_scores.append(bleu_score)
        test_result["bleu_score"] = float(bleu_score)

        rouge_score = rouge.compute(predictions=[response], references=[expected_answer])["rouge1"]
        rouge_scores.append(rouge_score)
        test_result["rouge1_score"] = float(rouge_score)

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
        "total_tests": total_cases,
        "pass_count": pass_count,
        "fail_count": fail_count,
        "pass_rate": float(pass_count / total_cases) if total_cases else 0,
        "average_response_time": float(np.mean(response_times)) if response_times else 0,
        "average_cosine_similarity": float(np.mean(cosine_similarities)) if cosine_similarities else 0,
        "average_bleu_score": float(np.mean(bleu_scores)) if bleu_scores else 0,
        "average_rouge1_score": float(np.mean(rouge_scores)) if rouge_scores else 0,
        "detailed_results": results
    }

    with open(EVALUATION_RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(evaluation_summary, f, indent=2)

    with open(EXTRACTED_SCORES_FILE, "w", encoding="utf-8") as f:
        json.dump(extracted_scores, f, indent=2)

    logger.info(f"Evaluation saved to {EVALUATION_RESULTS_FILE}")
    logger.info(f"Extracted scores saved to {EXTRACTED_SCORES_FILE}")

    assert evaluation_summary["pass_rate"] >= 0.8, f"Pass rate too low: {evaluation_summary['pass_rate']:.2%}"
