import json
import logging
from rag_pipeline import RAGPipeline
from llm import OllamaLLM
from sentence_transformers import SentenceTransformer

logging.basicConfig(filename='rag_project.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_test_cases(file_path: str = "test_cases.json") -> list:
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def run_rag_tests(pipeline: RAGPipeline, test_cases: list, output_file: str = "test_results.json") -> None:
    results = []
    for i, test_case in enumerate(test_cases, 1):
        question = test_case["question"]
        context = test_case["context"]
        logging.info(f"Processing test case {i}/{len(test_cases)}: {question}")
        answer, retrieved_contexts = pipeline.process(question)
        results.append({
            "question": question,
            "answer": answer,
            "context": context,
            "retrieved_contexts": retrieved_contexts
        })
        if i % 100 == 0:
            logging.info(f"Completed {i} test cases")

    # Save results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    logging.info(f"Test results saved to {output_file}")

if __name__ == "__main__":
    # Initialize components
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    llm = OllamaLLM()
    vector_store = None  # Replace with your vector store implementation (e.g., FAISSVectorStore)
    pipeline = RAGPipeline(vector_store, llm, embedding_model)

    # Load and run tests
    test_cases = load_test_cases()
    run_rag_tests(pipeline, test_cases)