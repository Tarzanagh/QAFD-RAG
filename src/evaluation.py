"""
Answer evaluation utilities for comparing and scoring RAG responses.

This module provides functions to evaluate multiple answers to the same question
based on criteria like comprehensiveness, diversity, logicality, relevance, and coherence.

This is a standalone module for post-hoc quality assessment, separate from the
core answering pipeline.
"""

import re
import json
from .utils import logger


async def evaluate_multiple_answers(
    query: str,
    answers: list[str],
    use_llm_func: callable,
) -> dict:
    """
    Evaluate multiple answers to the same question based on five criteria.

    Parameters:
    -----------
    query : str
        The original question
    answers : list[str]
        List of answers to evaluate
    use_llm_func : callable
        LLM function to use for evaluation

    Returns:
    --------
    dict
        Evaluation results with scores and rankings for each answer
    """
    if len(answers) < 2:
        logger.warning("Need at least 2 answers for evaluation")
        return {}

    # Create evaluation prompt for multiple answers
    answers_text = ""
    for i, answer in enumerate(answers, 1):
        answers_text += f"Answer {i}: {answer}\n\n"

    prompt = f"""---Role---
You are an expert tasked with evaluating multiple answers to the same question based on five criteria: Comprehensiveness, Diversity, Logicality, Relevance, and Coherence.

---Goal---
You will evaluate {len(answers)} answers to the same question based on five criteria:
- Comprehensiveness: How much detail does the answer provide to cover all aspects and details of the question?
- Diversity: How varied and rich is the answer in providing different perspectives and insights on the question?
- Logicality: How logically does the answer respond to all parts of the question?
- Relevance: How relevant is the answer to the question, staying focused and addressing the intended topic or issue?
- Coherence: How well does the answer maintain internal logical connections between its parts, ensuring a smooth and consistent structure?

Here is the question: {query}

Here are the {len(answers)} answers:
{answers_text}

For each criterion, assign a score from 1 to 10 to each answer, where:
- 1-2: Poor performance
- 3-4: Below average
- 5-6: Average
- 7-8: Good
- 9-10: Excellent

Then provide an overall ranking of the answers from best to worst.

Output your evaluation in the following JSON format:
{{
    "criterion_scores": {{
        "Comprehensiveness": {{
            "Answer 1": [score],
            "Answer 2": [score],
            ...
        }},
        "Diversity": {{
            "Answer 1": [score],
            "Answer 2": [score],
            ...
        }},
        "Logicality": {{
            "Answer 1": [score],
            "Answer 2": [score],
            ...
        }},
        "Relevance": {{
            "Answer 1": [score],
            "Answer 2": [score],
            ...
        }},
        "Coherence": {{
            "Answer 1": [score],
            "Answer 2": [score],
            ...
        }}
    }},
    "overall_scores": {{
        "Answer 1": [total_score],
        "Answer 2": [total_score],
        ...
    }},
    "ranking": ["Answer X", "Answer Y", ...],
    "best_answer": "Answer X",
    "explanations": {{
        "Answer 1": "Brief explanation of strengths and weaknesses",
        "Answer 2": "Brief explanation of strengths and weaknesses",
        ...
    }}
}}"""

    try:
        response = await use_llm_func(prompt, max_tokens=1000)

        # Try to find JSON in the response
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            try:
                evaluation_result = json.loads(json_match.group())
                return evaluation_result
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON from LLM response: {e}")
                logger.error(f"Response: {response}")
                return {}
        else:
            logger.error(f"No JSON found in LLM response: {response}")
            return {}

    except Exception as e:
        logger.error(f"Error during answer evaluation: {e}")
        return {}


async def compare_two_answers(
    query: str,
    answer1: str,
    answer2: str,
    use_llm_func: callable,
) -> dict:
    """
    Compare two answers to the same question based on five criteria.

    Parameters:
    -----------
    query : str
        The original question
    answer1 : str
        First answer to evaluate
    answer2 : str
        Second answer to evaluate
    use_llm_func : callable
        LLM function to use for evaluation

    Returns:
    --------
    dict
        Comparison results with winner for each criterion and overall winner
    """
    prompt = f"""---Role---
You are an expert tasked with evaluating two answers to the same question based on five criteria: Comprehensiveness, Diversity, Logicality, Relevance, and Coherence.

---Goal---
You will evaluate two answers to the same question based on five criteria:
- Comprehensiveness: How much detail does the answer provide to cover all aspects and details of the question?
- Diversity: How varied and rich is the answer in providing different perspectives and insights on the question?
- Logicality: How logically does the answer respond to all parts of the question?
- Relevance: How relevant is the answer to the question, staying focused and addressing the intended topic or issue?
- Coherence: How well does the answer maintain internal logical connections between its parts, ensuring a smooth and consistent structure?

Here is the question: {query}

Here are the two answers:
Answer 1: {answer1}

Answer 2: {answer2}

For each criterion, choose the better answer (either Answer 1 or Answer 2) and explain why. Then, select an overall winner based on these five criteria.

Output your evaluation in the following JSON format:
{{
    "Comprehensiveness": {{ "Winner": "[Answer 1 or Answer 2]", "Explanation": "[Provide explanation here]" }},
    "Diversity": {{ "Winner": "[Answer 1 or Answer 2]", "Explanation": "[Provide explanation here]" }},
    "Logicality": {{ "Winner": "[Answer 1 or Answer 2]", "Explanation": "[Provide explanation here]" }},
    "Relevance": {{ "Winner": "[Answer 1 or Answer 2]", "Explanation": "[Provide explanation here]" }},
    "Coherence": {{ "Winner": "[Answer 1 or Answer 2]", "Explanation": "[Provide explanation here]" }},
    "Overall Winner": {{ "Winner": "[Answer 1 or Answer 2]", "Explanation": "[Summarize why this answer is the overall winner based on the five criteria]" }}
}}"""

    try:
        response = await use_llm_func(prompt, max_tokens=800)

        # Try to find JSON in the response
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            try:
                comparison_result = json.loads(json_match.group())
                return comparison_result
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON from LLM response: {e}")
                logger.error(f"Response: {response}")
                return {}
        else:
            logger.error(f"No JSON found in LLM response: {response}")
            return {}

    except Exception as e:
        logger.error(f"Error during answer comparison: {e}")
        return {}


def calculate_evaluation_metrics(evaluation_result: dict) -> dict:
    """
    Calculate additional metrics from evaluation results.

    Parameters:
    -----------
    evaluation_result : dict
        Result from evaluate_multiple_answers function

    Returns:
    --------
    dict
        Additional metrics including average scores, standard deviations, etc.
    """
    if not evaluation_result or "criterion_scores" not in evaluation_result:
        return {}

    metrics = {}
    criterion_scores = evaluation_result["criterion_scores"]

    # Calculate average scores for each criterion
    for criterion, scores in criterion_scores.items():
        if isinstance(scores, dict):
            values = [v for v in scores.values() if isinstance(v, (int, float))]
            if values:
                metrics[f"{criterion}_average"] = sum(values) / len(values)
                metrics[f"{criterion}_max"] = max(values)
                metrics[f"{criterion}_min"] = min(values)

    # Calculate overall statistics
    if "overall_scores" in evaluation_result:
        overall_scores = evaluation_result["overall_scores"]
        if isinstance(overall_scores, dict):
            values = [v for v in overall_scores.values() if isinstance(v, (int, float))]
            if values:
                metrics["overall_average"] = sum(values) / len(values)
                metrics["overall_max"] = max(values)
                metrics["overall_min"] = min(values)
                metrics["score_range"] = max(values) - min(values)

    return metrics


__all__ = [
    "evaluate_multiple_answers",
    "compare_two_answers",
    "calculate_evaluation_metrics",
]
