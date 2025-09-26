
"""
copy/pasted from: https://github.com/knoveleng/open-rs/
"""

from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify


def accuracy_reward(completion: str, solution: str) -> bool:
    """Reward function that checks if the completion is the same as the ground truth."""

    gold_parsed = parse(
        solution,
        extraction_mode="first_match",
        extraction_config=[LatexExtractionConfig()],
    )
    if len(gold_parsed) != 0:
        # We require the answer to be provided in correct latex (no malformed operators)
        answer_parsed = parse(
            completion,
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        equations=True,
                        boxed="all",
                        units=True,
                    ),
                    # Ensures that boxed is tried first
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                )
            ],
            extraction_mode="first_match",
        )
        # Reward 1 if the content is the same as the ground truth, 0 otherwise
        try:
            return verify(answer_parsed, gold_parsed)
        except Exception as e:
            print(
                f"verify failed: {e}, answer: {answer_parsed}, gold: {gold_parsed}"
            )
            return False

    # If the gold solution is not parseable, we reward 1 to skip this example
    print("Failed to parse gold solution: ", solution)

    return True