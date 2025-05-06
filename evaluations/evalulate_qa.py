import json
import logging
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from evaluate import load  # For BLEU, ROUGE, METEOR, BERTScore
from transformers import pipeline
import os

# === Configure Logging ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === Paths ===
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
EVAL_DIR = os.path.join(PROJECT_ROOT, "evaluations")
os.makedirs(EVAL_DIR, exist_ok=True)

INPUT_FILE = os.path.join(EVAL_DIR, "test_scores.json")
OUTPUT_FILE = os.path.join(EVAL_DIR, "final_all_enhanced_scores.json")
SUMMARY_FILE = os.path.join(EVAL_DIR, "enchanced_final_avg_score.json")

# === Load Metrics Libraries ===
bleu_metric = load("bleu")
rouge_metric = load("rouge")
meteor_metric = load("meteor")
bertscore_metric = load("bertscore", lang="en")

# === Initialize NLI Pipeline ===
nli_pipeline = pipeline("text-classification", model="cross-encoder/nli-deberta-v3-base")

# === Load Embedder for Semantic Similarity ===
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# === Helper Functions ===
def compute_bertscore(reference, prediction):
    """Compute BERTScore F1 score with language specified."""
    result = bertscore_metric.compute(
        predictions=[prediction],
        references=[reference],
        lang="en"
    )
    return result["f1"][0]

def classify_nli(premise_text, hypothesis_text):
    """
    Use NLI model to predict relationship: Does premise_text entail/contradict/neutral to hypothesis_text?
    Returns 'entailment', 'contradiction', or 'neutral'.
    The underlying model (e.g., facebook/bart-large-mnli) is trained to determine
    if the hypothesis is entailed by, contradicts, or is neutral to the premise.
    """
    # The text-classification pipeline with an NLI model takes premise first, then hypothesis.
    # Pass the input as a list containing the tuple: [(premise, hypothesis)]
    # Add padding=True and truncation=True to handle sequences of different lengths.
    results = nli_pipeline([(premise_text, hypothesis_text)], padding=True, truncation=True) # Pass as list containing tuple
    # Example result: [{'label': 'entailment', 'score': 0.99}]
    # Extract label from the first result in the list
    return results[0]["label"]

# === Main Function ===
def enhance_evaluation_results(input_path: str, output_path: str, summary_path: str):
    logger.info(f"Loading test results from {input_path}")
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
    except Exception as e:
        logger.error(f"Failed to read input file: {e}")
        return

    bert_f1_scores = []
    meteor_scores = []
    cosine_similarities = []
    bleu_scores = []
    rouge_scores = []
    contradictions = []
    neutrals = []
    entailments = []

    for idx, item in enumerate(results, 1):
        question = item["question"]
        expected_answer = item["expected_answer"]
        generated_response = item["generated_response"]

        logger.info(f"Evaluating Q{idx}")

        # Compute Cosine Similarity
        response_embedding = embedder.encode(generated_response)
        expected_embedding = embedder.encode(expected_answer)
        similarity = cosine_similarity([response_embedding], [expected_embedding])[0][0]
        item["cosine_similarity"] = float(similarity)
        cosine_similarities.append(similarity)

        # Compute BLEU
        bleu_score = bleu_metric.compute(predictions=[generated_response], references=[[expected_answer]])["bleu"]
        item["bleu_score"] = float(bleu_score)
        bleu_scores.append(bleu_score)

        # Compute ROUGE
        rouge_score = rouge_metric.compute(predictions=[generated_response], references=[[expected_answer]])["rouge1"]
        item["rouge1_score"] = float(rouge_score)
        rouge_scores.append(rouge_score)

        # Compute METEOR
        meteor_score = meteor_metric.compute(predictions=[generated_response], references=[[expected_answer]])["meteor"]
        item["meteor_score"] = float(meteor_score)
        meteor_scores.append(meteor_score)

        # Compute BERTScore F1
        bert_f1 = compute_bertscore(expected_answer, generated_response)
        item["bert_f1_score"] = float(bert_f1)
        bert_f1_scores.append(bert_f1)

        # NLI Classification
        nli_label = classify_nli(generated_response, expected_answer)
        item["nli_label"] = nli_label
        item["contradiction"] = nli_label == "contradiction"
        item["neutral"] = nli_label == "neutral"
        item["entailment"] = nli_label == "entailment"

        contradictions.append(item["contradiction"])
        neutrals.append(item["neutral"])
        entailments.append(item["entailment"])

        logger.info(f"BERTScore F1: {bert_f1:.4f} | Cosine Sim: {similarity:.4f} | NLI: {nli_label}")

    # Save enhanced results
    logger.info(f"Saving enhanced results to {output_path}")
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to write output file: {e}")
        return

    # Generate Summary
    total = len(results)
    summary = {
        "total_questions": total,
        "average_bert_f1_score": float(np.mean(bert_f1_scores)) if bert_f1_scores else 0,
        "average_cosine_similarity": float(np.mean(cosine_similarities)) if cosine_similarities else 0,
        "average_bleu_score": float(np.mean(bleu_scores)) if bleu_scores else 0,
        "average_rouge_score": float(np.mean(rouge_scores)) if rouge_scores else 0,
        "average_meteor_score": float(np.mean(meteor_scores)) if meteor_scores else 0,
        "total_contradictions": sum(contradictions),
        "total_neutrals": sum(neutrals),
        "total_entailments": sum(entailments),
        "contradiction_rate": round(sum(contradictions) / total, 4) if total else 0,
        "entailment_rate": round(sum(entailments) / total, 4) if total else 0
    }

    try:
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to write summary file: {e}")
        return

    logger.info(f"Evaluation complete. Summary saved to {summary_path}")


# === Run Enhancement ===
if __name__ == "__main__":
    enhance_evaluation_results(INPUT_FILE, OUTPUT_FILE, SUMMARY_FILE)
