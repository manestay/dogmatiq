"""
This script selects the top-k nuggets from a NuggetBank based on various selection methods.
It expects pre-processed NuggetBank files from step3.
"""

import argparse
import random
import sys
from copy import copy
from pathlib import Path

import pandas as pd
from argue_eval.validation.nugget_data import (
    NuggetBank,
    NuggetQuestion,
    load_nugget_bank_json,
    write_nugget_bank_json,
)
from config import MERGED_OUTPUT_DIR

sys.path.append("..")
from metrics.scripts.classification_lib import load_model_package, predict_with_model_package

# TODO: Uncomment the above line when the classification_lib is available

METHODS = ["most_common", "model", "most_common_model", "random"]

parser = argparse.ArgumentParser(description="Select top nuggets based on various methods")
parser.add_argument(
    "-i",
    "--processed_path",
    type=Path,
    default=MERGED_OUTPUT_DIR / "clean_only",
    help="Dir containing processed NuggetBank JSON files from step3, or a single file",
)
parser.add_argument(
    "-o",
    "--output_path",
    type=Path,
    default=None,
    help="Dir to save top-k subset files, or a single file",
)
parser.add_argument(
    "--top_k", type=int, default=20, help="Number of top questions to select (default: 20)"
)
parser.add_argument(
    "-mq",
    "--method",
    type=str,
    choices=METHODS,
    default="most_common",
    help="Method to use for question selection",
)
parser.add_argument(
    "-m",
    "--model_package",
    type=Path,
    default=None,
    help="Model package to use for answer selection (if applicable)",
)
parser.add_argument(
    "--no_delete_subnuggets",
    dest="delete_subnuggets",
    action="store_false",
    help="Do not delete subnuggets from the NuggetBank",
)
parser.add_argument(
    "--max_most_common",
    type=float,
    default=1.0,
    help="For most_common_model method, max fraction from most_common method",
)
parser.add_argument(
    "--seed",
    type=int,
    default=1234,
    help="Random seed for random selection method",
)


def get_questions_with_counts(nugget_bank: NuggetBank):
    """
    Helper function to get questions with their counts - used for code reuse.

    Returns:
        List of (question, paraphrase_count, answer_count) tuples sorted by counts descending
    """
    if nugget_bank.nugget_bank is None:
        return []

    nugget_questions = [nugget for nugget in nugget_bank.nugget_bank.values()]

    questions_with_counts = [
        (
            q,
            q.metadata.get("num_paraphrases_total", 0) if q.metadata else 0,
            q.metadata.get("num_answers", 0) if q.metadata else 0,
        )
        for q in nugget_questions
    ]
    questions_with_counts.sort(key=lambda x: (x[1], x[2]), reverse=True)
    return questions_with_counts


def create_bank_excluding_questions(
    nugget_bank: NuggetBank, excluded_question_ids: set
) -> NuggetBank:
    """
    Helper function to create a new NuggetBank excluding questions with specified IDs.

    Args:
        nugget_bank: The original NuggetBank
        excluded_question_ids: Set of question IDs to exclude

    Returns:
        New NuggetBank with excluded questions removed
    """
    remaining_bank = copy(nugget_bank)
    remaining_bank.nugget_bank = {}

    if nugget_bank.nugget_bank is not None:
        for question, nugget in nugget_bank.nugget_bank.items():
            nugget_question_id = getattr(nugget, "question_id", None)
            if not nugget_question_id or nugget_question_id not in excluded_question_ids:
                remaining_bank.nugget_bank[question] = nugget

    return remaining_bank


def filter_high_count_questions(questions_with_counts, min_paraphrase_count=1, min_answer_count=1):
    """
    Filter questions that have counts greater than the specified minimums.

    Args:
        questions_with_counts: List of (question, paraphrase_count, answer_count) tuples
        min_paraphrase_count: Minimum paraphrase count (exclusive)
        min_answer_count: Minimum answer count (exclusive)

    Returns:
        Filtered list of questions with counts above the thresholds
    """
    return [
        (q, para_count, ans_count)
        for q, para_count, ans_count in questions_with_counts
        if isinstance(para_count, int)
        and isinstance(ans_count, int)
        and para_count > min_paraphrase_count
        and ans_count > min_answer_count
    ]


def convert_nugget_bank_to_dataframe(nugget_bank: NuggetBank):
    """
    Convert a NuggetBank to a DataFrame format expected by the model.

    Args:
        nugget_bank: The NuggetBank to convert

    Returns:
        DataFrame with question_id as index and quality features as columns
    """
    # Convert nugget_bank to DataFrame format expected by the model
    scores_data = []
    if nugget_bank.nugget_bank is not None:
        for question, nugget_question in nugget_bank.nugget_bank.items():
            # Only process NuggetQuestion objects, not NuggetClaim
            if not isinstance(nugget_question, NuggetQuestion):
                continue
            if nugget_question.question_id is None:
                continue
            if nugget_question.metadata is None:
                continue

            quality = nugget_question.metadata.get("quality", {})

            # Dictionary to store this question's scores
            question_scores = {}
            question_scores["question_id"] = nugget_question.question_id

            for key, value in quality.items():
                if key == "vitality":
                    value = 1 if value == "VITAL" else 0
                if isinstance(value, (int, float)):
                    question_scores[key] = value
                elif isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        if key == "general_qualities":
                            key_prefix = "quality"  # shorten
                        else:
                            key_prefix = key
                        if isinstance(subvalue, (int, float)):
                            flat_key = f"{key_prefix}_{subkey}"
                            question_scores[flat_key] = subvalue

            scores_data.append(question_scores)

    # Create DataFrame with questions as rows and features as columns
    data_df = pd.DataFrame(scores_data)
    if not data_df.empty:
        data_df = data_df.set_index("question_id")

    return data_df


def select_nuggets_most_common(nugget_bank: NuggetBank, top_k: int = 10) -> NuggetBank:
    """
    Create a subset with the top-k questions from a NuggetBank
    based on metadata.num_paraphrases_total.

    Args:
        nugget_bank: The NuggetBank to select from
        top_k: Number of top questions to select (default: 10)
    """
    if nugget_bank.nugget_bank is None:
        return nugget_bank

    # Use the shared helper function to get questions with counts
    questions_with_counts = get_questions_with_counts(nugget_bank)

    # Select top-k questions
    top_questions_with_counts = questions_with_counts[:top_k]

    # Print the counts of the top-k questions
    print(f"Top {top_k} questions by (paraphrase count, answer count):")
    for i, (question, paraphrase_count, answer_count) in enumerate(top_questions_with_counts, 1):
        question_text = getattr(question, "question", getattr(question, "claim", str(question)))
        print(f"  {i}. {question_text} | paraphrases: {paraphrase_count}, answers: {answer_count}")

    # Extract just the questions for the bank
    top_questions = [q for q, _, _ in top_questions_with_counts]

    # Create a new NuggetBank with only the top questions
    subset_bank = copy(nugget_bank)
    subset_bank.nugget_bank = {}
    subset_bank.add_nuggets(top_questions)

    return subset_bank


def select_nuggets_with_model(
    nugget_bank: NuggetBank, model_package_path: Path, request_id: str, top_k: int = 10
) -> NuggetBank:
    """
    Process a cleaned NuggetBank using a trained model to select top-k questions.

    Args:
        nugget_bank: The NuggetBank to process
        model_package_path: Path to the saved model package
        request_id: Request ID for group-specific scaling
        top_k: Number of top questions to select
    """
    if nugget_bank.nugget_bank is None:
        return nugget_bank

    qid_question_map = {
        getattr(nugget, "question_id", None): question
        for question, nugget in nugget_bank.nugget_bank.items()
        if getattr(nugget, "question_id", None) is not None
    }

    model_package = load_model_package(model_package_path)

    # Convert cleaned_bank to DataFrame format expected by the model
    data_df = convert_nugget_bank_to_dataframe(nugget_bank)

    # Get model predictions and rankings
    ranked_question_ids = predict_with_model_package(
        model_package,
        data_df,
        request_id=request_id,
    )

    # Select top-k questions based on model ranking
    top_questions_with_scores = [
        (qid, score) for qid, score in ranked_question_ids if qid in data_df.index
    ]
    top_questions_with_scores = top_questions_with_scores[:top_k]
    # Create a new bank with only the top-ranked questions
    top_questions = []
    for qid, score in top_questions_with_scores:
        question = qid_question_map[qid]
        nugget_question = nugget_bank.nugget_bank[question]
        if nugget_question is not None and nugget_question.metadata is not None:
            nugget_question.metadata["model_score"] = score
            top_questions.append(nugget_question)

    processed_bank = copy(nugget_bank)
    processed_bank.nugget_bank = {}
    processed_bank.add_nuggets(top_questions)

    return processed_bank


def select_nuggets_most_common_model(
    nugget_bank: NuggetBank,
    model_package_path: Path,
    request_id: str,
    top_k: int = 10,
    max_most_common: float = 1.0,
) -> NuggetBank:
    """
    Hybrid selection method that prioritizes high-count nuggets from most_common method,
    then fills remaining slots with model-ranked nuggets.

    Args:
        nugget_bank: The NuggetBank to select from
        model_package_path: Path to the saved model package
        request_id: Request ID for group-specific scaling
        top_k: Number of top questions to select
    """
    if nugget_bank.nugget_bank is None:
        return nugget_bank

    # Step 1: Get all nuggets with their counts using existing logic
    questions_with_counts = get_questions_with_counts(nugget_bank)

    # Filter for counts > (1,1) and sort by counts (already sorted from function)
    high_count_questions = filter_high_count_questions(
        questions_with_counts, min_paraphrase_count=1, min_answer_count=1
    )
    if max_most_common < 1.0:
        num_high_count = len(high_count_questions)
        num_to_use = int(top_k * max_most_common)
        if num_to_use < num_high_count:
            print(f"  Applying max_most_common, selected {num_to_use}/{num_high_count}")
            high_count_questions = high_count_questions[:num_to_use]

    # Track selected question IDs to avoid duplicates
    selected_question_ids = set()
    selected_nuggets = []

    # Add high-count questions first (up to top_k)
    for q, para_count, ans_count in high_count_questions[:top_k]:
        question_id = getattr(q, "question_id", None)
        if question_id:
            selected_question_ids.add(question_id)
        selected_nuggets.append(q)

    high_count_selected = len(selected_nuggets)

    # Step 2: If we need more nuggets, use model ranking to fill remaining slots
    remaining_slots = top_k - high_count_selected
    if remaining_slots > 0:
        # Create a temporary bank with all remaining nuggets
        remaining_bank = create_bank_excluding_questions(nugget_bank, selected_question_ids)

        if remaining_bank.nugget_bank:
            # Use model selection on remaining nuggets
            model_selected_bank = select_nuggets_with_model(
                remaining_bank, model_package_path, request_id, remaining_slots
            )

            # Add model-selected nuggets to our selection
            if model_selected_bank.nugget_bank:
                selected_nuggets.extend(model_selected_bank.nugget_bank.values())

    # Step 3: Create final bank with selected nuggets
    result_bank = copy(nugget_bank)
    result_bank.nugget_bank = {}
    result_bank.add_nuggets(selected_nuggets)

    # Print summary
    total_selected = len(result_bank.nugget_bank)
    model_selected = total_selected - high_count_selected

    print(f"  - High-count nuggets: {high_count_selected}")
    if model_selected > 0:
        print(f"  - Model-selected nuggets: {model_selected}")

    return result_bank


def select_nuggets_random(nugget_bank: NuggetBank, top_k: int = 10, seed: int = 42) -> NuggetBank:
    """
    Select nuggets randomly from a NuggetBank. Used as a baseline method.

    Args:
        nugget_bank: The NuggetBank to select from
        top_k: Number of nuggets to select randomly
        seed: Random seed for reproducibility (default: 42)

    Returns:
        NuggetBank with randomly selected nuggets
    """
    if nugget_bank.nugget_bank is None:
        return nugget_bank

    # Set the random seed for reproducibility
    random.seed(seed)

    # Get all nugget questions
    all_nuggets = list(nugget_bank.nugget_bank.values())

    # Randomly sample top_k nuggets
    selected_nuggets = random.sample(all_nuggets, top_k)

    print(f"Selected {len(selected_nuggets)} nuggets randomly (seed: {seed})")

    # Create a new NuggetBank with only the selected nuggets
    result_bank = copy(nugget_bank)
    result_bank.nugget_bank = {}
    result_bank.add_nuggets(selected_nuggets)

    return result_bank


def main():
    args = parser.parse_args()
    print(args)

    assert args.output_path.suffix == "", "Error: Output path must be a directory"
    args.output_path.mkdir(parents=True, exist_ok=True)

    # Generate list of input files
    if args.processed_path.is_file():
        input_files = [args.processed_path]
        assert args.output_path != args.processed_path.parent
    else:
        BASIC_METHODS = set(["most_common", "random"])
        pattern = "*_bank.json" if args.method in BASIC_METHODS else "*_with_scores.json"
        input_files = list(args.processed_path.glob(pattern))
        assert args.output_path != args.processed_path

    if not input_files:
        print(f"No nugget bank files found in {args.processed_path}")
        return

    if (
        args.method == "model" or args.method == "most_common_model"
    ) and args.model_package is None:
        print(f"Error: --model_package must be specified when using method {args.method}")
        return

    # Process each nugget bank file
    for input_file in input_files:
        print(f"--- Processing {input_file} ---")

        # Load the nugget bank (already processed from step3)
        nugget_bank = load_nugget_bank_json(input_file)
        num_nuggets_orig = len(nugget_bank.nugget_bank) if nugget_bank.nugget_bank else 0
        print(f"Loaded: {num_nuggets_orig} nuggets")

        # Check if we have fewer nuggets than requested - if so, just use all nuggets
        if num_nuggets_orig <= args.top_k:
            print(
                f"Only {num_nuggets_orig} nuggets available (requested {args.top_k}), using all nuggets"
            )
            processed_bank = nugget_bank
        else:
            # Apply the selected method to choose top_k nuggets
            if args.method == "most_common":
                # Select top-k based on paraphrase counts
                processed_bank = select_nuggets_most_common(nugget_bank, args.top_k)
            elif args.method == "model":
                request_id = input_file.stem.split("_")[0]
                processed_bank = select_nuggets_with_model(
                    nugget_bank, args.model_package, request_id, args.top_k
                )
            elif args.method == "most_common_model":
                request_id = input_file.stem.split("_")[0]
                processed_bank = select_nuggets_most_common_model(
                    nugget_bank, args.model_package, request_id, args.top_k, args.max_most_common
                )
            elif args.method == "random":
                # Select top-k randomly with seed
                processed_bank = select_nuggets_random(nugget_bank, args.top_k, args.seed)

        print(
            f"Selected: {len(processed_bank.nugget_bank) if processed_bank.nugget_bank else 0} nuggets"
        )

        if args.delete_subnuggets:
            # Delete subnuggets from the processed bank
            if processed_bank.nugget_bank is not None:
                for nugget in processed_bank.nugget_bank.values():
                    if hasattr(nugget, "sub_nuggets"):
                        nugget.sub_nuggets = None
            # print("Deleted subnuggets from the processed bank")

        # Save the processed bank
        output_path = args.output_path / input_file.name
        args.output_path.mkdir(parents=True, exist_ok=True)
        write_nugget_bank_json(processed_bank, output_path)
        print(f"Saved processed NuggetBank to {output_path}")


if __name__ == "__main__":
    main()
