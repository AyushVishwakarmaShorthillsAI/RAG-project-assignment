import re
import json

# Input and output file paths
LOG_FILE = "terminal_logs.txt"
SCORES_FILE = "extracted_scores.json"
EVALUATION_FILE = "final_evaluation.json"

# Regex patterns to match the required data
QUESTION_PATTERN = re.compile(r"Testing question: (.+)$")
EXPECTED_ANSWER_PATTERN = re.compile(r"Expected Answer: (.+)$")
GENERATED_RESPONSE_PATTERN = re.compile(r"Generated Response: (.+)$")
COSINE_PATTERN = re.compile(r"Cosine similarity: (\d+\.\d+)")
BLEU_PATTERN = re.compile(r"BLEU score: (\d+\.\d+)")
ROUGE_PATTERN = re.compile(r"ROUGE-1 score: (\d+\.\d+)")
RESPONSE_TIME_PATTERN = re.compile(r"Response time: (\d+\.\d+) seconds")

# Lists and counters for evaluation
extracted_scores = []
response_times = []
pass_count = 0
fail_count = 0

# Temporary variables to hold data for the current test case
current_data = {}
current_question = None

# Read the log file
with open(LOG_FILE, "r", encoding="utf-8") as f:
    for line in f:
        # Match the question (start of a new test case)
        question_match = QUESTION_PATTERN.search(line)
        if question_match:
            # If we have a previous test case with complete data, process it
            if all(key in current_data for key in ["cosine_similarity", "bleu_score", "rouge1_score", "response_time", "expected_answer", "generated_response"]):
                # Compute pass/fail
                exact_match = current_data["expected_answer"].lower() in current_data["generated_response"].lower()
                passed = exact_match or current_data["cosine_similarity"] > 0.7
                current_data["passed"] = passed
                
                # Update counters
                if passed:
                    pass_count
                    pass_count += 1
                else:
                    fail_count
                    fail_count += 1
                
                # Add to extracted scores
                extracted_scores.append({
                    "question": current_question,
                    "expected_answer": current_data["expected_answer"],
                    "generated_response": current_data["generated_response"],
                    "cosine_similarity": current_data["cosine_similarity"],
                    "bleu_score": current_data["bleu_score"],
                    "rouge1_score": current_data["rouge1_score"],
                    "response_time": current_data["response_time"],
                    "passed": current_data["passed"]
                })
                
                # Add response time to list
                response_times.append(current_data["response_time"])
            
            # Reset for the new test case
            current_question = question_match.group(1)
            current_data = {}
            continue
        
        # Match expected answer
        expected_match = EXPECTED_ANSWER_PATTERN.search(line)
        if expected_match:
            current_data["expected_answer"] = expected_match.group(1)
            continue
        
        # Match generated response
        generated_match = GENERATED_RESPONSE_PATTERN.search(line)
        if generated_match:
            current_data["generated_response"] = generated_match.group(1)
            continue
        
        # Match cosine similarity
        cosine_match = COSINE_PATTERN.search(line)
        if cosine_match:
            current_data["cosine_similarity"] = float(cosine_match.group(1))
            continue
        
        # Match BLEU score
        bleu_match = BLEU_PATTERN.search(line)
        if bleu_match:
            current_data["bleu_score"] = float(bleu_match.group(1))
            continue
        
        # Match ROUGE score
        rouge_match = ROUGE_PATTERN.search(line)
        if rouge_match:
            current_data["rouge1_score"] = float(rouge_match.group(1))
            continue
        
        # Match response time
        time_match = RESPONSE_TIME_PATTERN.search(line)
        if time_match:
            current_data["response_time"] = float(time_match.group(1))
            continue

# Save the last test case if it has complete data
if all(key in current_data for key in ["cosine_similarity", "bleu_score", "rouge1_score", "response_time", "expected_answer", "generated_response"]):
    # Compute pass/fail
    exact_match = current_data["expected_answer"].lower() in current_data["generated_response"].lower()
    passed = exact_match or current_data["cosine_similarity"] > 0.6
    current_data["passed"] = passed
    
    # Update counters
    if passed:
        pass_count += 1
    else:
        fail_count += 1
    
    # Add to extracted scores
    extracted_scores.append({
        "question": current_question,
        "expected_answer": current_data["expected_answer"],
        "generated_response": current_data["generated_response"],
        "cosine_similarity": current_data["cosine_similarity"],
        "bleu_score": current_data["bleu_score"],
        "rouge1_score": current_data["rouge1_score"],
        "response_time": current_data["response_time"],
        "passed": current_data["passed"]
    })
    
    # Add response time to list
    response_times.append(current_data["response_time"])

# Calculate evaluation metrics
total_tests = len(extracted_scores)
pass_rate = pass_count / total_tests if total_tests > 0 else 0
average_response_time = sum(response_times) / len(response_times) if response_times else 0
average_cosine_similarity = sum(score["cosine_similarity"] for score in extracted_scores) / total_tests if total_tests > 0 else 0
average_bleu_score = sum(score["bleu_score"] for score in extracted_scores) / total_tests if total_tests > 0 else 0
average_rouge1_score = sum(score["rouge1_score"] for score in extracted_scores) / total_tests if total_tests > 0 else 0

# Save extracted scores to a file
with open(SCORES_FILE, "w", encoding="utf-8") as f:
    json.dump(extracted_scores, f, indent=2)

# Save final evaluation to a file
evaluation_summary = {
    "total_tests": total_tests,
    "pass_count": pass_count,
    "fail_count": fail_count,
    "pass_rate": pass_rate,
    "average_response_time": average_response_time,
    "average_cosine_similarity": average_cosine_similarity,
    "average_bleu_score": average_bleu_score,
    "average_rouge1_score": average_rouge1_score
}
with open(EVALUATION_FILE, "w", encoding="utf-8") as f:
    json.dump(evaluation_summary, f, indent=2)

print(f"Extracted scores saved to {SCORES_FILE}")
print(f"Final evaluation saved to {EVALUATION_FILE}")
print(f"Total test cases processed: {total_tests}")
print(f"Pass count: {pass_count}")
print(f"Fail count: {fail_count}")
print(f"Pass rate: {pass_rate:.2%}")
print(f"Average response time: {average_response_time:.2f} seconds")