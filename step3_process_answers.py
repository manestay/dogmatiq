"""
This script processes and cleans answers in NuggetBank files.
It applies cleaning and deduplication steps to answers and questions.
"""

import argparse
import asyncio
import json
import sys
from copy import copy
from pathlib import Path

from argue_eval.utils import batch_model_responses
from argue_eval.validation.nugget_data import (
    AggregatorType,
    NuggetBank,
    NuggetQuestion,
    Reference,
    load_nugget_bank_json,
    write_nugget_bank_json,
)
from config import MERGED_OUTPUT_DIR, PROVIDER, STOP_SEQUENCES
from prompts import (
    AGGREGATE_ANSWERS_SYSTEM_MULTI,
    AGGREGATE_ANSWERS_SYSTEM_SINGLE,
    AGGREGATE_ANSWERS_USER,
)

sys.path.append(".")
sys.path.append("..")
from lib import json_loads, refresh_metadata_nums

parser = argparse.ArgumentParser(description="Process and clean answers in NuggetBank files")
parser.add_argument(
    "-i",
    "--merged_path",
    type=Path,
    default=MERGED_OUTPUT_DIR,
    help="Path to a dir containing NuggetBank JSON files from step2, or a single file",
)
parser.add_argument(
    "-o",
    "--output_path",
    type=Path,
    default=None,
    help="Directory to save processed files, or a single file",
)
parser.add_argument(
    "--mode",
    choices=["clean_only", "clean_agg"],
    default="clean_agg",
    help="Processing mode: clean_only (just clean) or clean_agg (clean and aggregate)",
)

UNANSWERED_PREFIX = {
    "unknown",
    "no info",
    "no details",
    "not ",  # with trailing space to avoid partial matches
    "none ",  #
}
UNANSWERED_EXACT = {"none"}


def merge_references(
    refs1: list[Reference] | None, refs2: list[Reference] | None
) -> list[Reference] | None:
    """
    Merge two lists of references, avoiding duplicates based on doc_id.
    """
    if refs1 is None:
        return refs2
    if refs2 is None:
        return refs1

    seen_ids = set()
    refs_merged = []

    for ref in refs1:
        if ref.doc_id not in seen_ids:
            seen_ids.add(ref.doc_id)
            refs_merged.append(ref)

    for ref in refs2:
        if ref.doc_id not in seen_ids:
            seen_ids.add(ref.doc_id)
            refs_merged.append(ref)

    return refs_merged


def clean_answers(nugget_question: NuggetQuestion) -> NuggetQuestion:
    """Clean and deduplicate answers for a NuggetQuestion."""
    if not nugget_question.answers:
        return nugget_question

    num_answers = len(nugget_question.answers)
    num_cleaned = 0
    subset_answers = {}
    seen_answers = {}  # map uncased to cased text

    for text, obj in nugget_question.answers.items():
        text_uncased = text.lower()
        if (
            any(text_uncased.startswith(x) for x in UNANSWERED_PREFIX)
            or text_uncased in UNANSWERED_EXACT
        ):
            num_cleaned += 1
            continue
        if text_uncased in seen_answers:
            text_cased = seen_answers[text_uncased]
            ans = subset_answers[text_cased]
            ans.references = merge_references(ans.references, obj.references)
            # print(f"Duplicate answer found: '{text}' merged with '{text_cased}'")
            continue
        seen_answers[text_uncased] = text
        subset_answers[text] = obj
    if num_cleaned > 0:
        print(
            f"Dropped {num_cleaned}/{num_answers} unanswered answers from: {nugget_question.question}"
        )

    new_question = copy(nugget_question)
    new_question.answers = subset_answers
    refresh_metadata_nums(new_question)
    return new_question


def clean_nugget_bank(nugget_bank: NuggetBank) -> tuple[NuggetBank, NuggetBank | None]:
    """Clean a NuggetBank by processing answers and refreshing metadata."""
    if nugget_bank.nugget_bank is None:
        return nugget_bank, None

    cleaned_nuggets = []
    unanswered_nuggets = []
    for nugget in nugget_bank.nugget_bank.values():
        if isinstance(nugget, NuggetQuestion):
            cleaned_nugget = clean_answers(nugget)

            # Only keep nuggets with answers
            if cleaned_nugget.answers and len(cleaned_nugget.answers) > 0:
                cleaned_nuggets.append(cleaned_nugget)
            else:
                unanswered_nuggets.append(cleaned_nugget)
        # Note: NuggetClaim handling could be added here if needed

    # Create a new bank with cleaned nuggets
    cleaned_bank = copy(nugget_bank)
    cleaned_bank.nugget_bank = {}
    cleaned_bank.add_nuggets(cleaned_nuggets)

    unanswered_bank = None
    if unanswered_nuggets:
        print(f"{len(unanswered_nuggets)} nuggets removed, due to null answers.")
        unanswered_bank = copy(nugget_bank)
        unanswered_bank.nugget_bank = {}
        unanswered_bank.add_nuggets(unanswered_nuggets)
    return cleaned_bank, unanswered_bank


async def aggregate_nugget_bank(nugget_bank: NuggetBank) -> NuggetBank:
    """Determine the aggregator for each answer set in the NuggetBank."""
    system_prompts = []
    user_prompts = []
    nugget_list = list(nugget_bank.nugget_bank.values())

    for nugget in nugget_list:
        question_text = nugget.question
        answers = list(nugget.answers.keys())
        if len(answers) > 1:
            answer_d = json.dumps({f"{i + 1}": ans for i, ans in enumerate(answers)})
            user_prompt = AGGREGATE_ANSWERS_USER.format(question=question_text, answers=answer_d)
            system_prompts.append(AGGREGATE_ANSWERS_SYSTEM_MULTI)
        else:
            user_prompt = AGGREGATE_ANSWERS_USER.format(question=question_text, answers=answers[0])
            system_prompts.append(AGGREGATE_ANSWERS_SYSTEM_SINGLE)
        user_prompts.append(user_prompt)

    responses = await batch_model_responses(
        system_prompts,
        user_prompts,
        provider=PROVIDER,
        stop=STOP_SEQUENCES,
        max_tokens=None,
        modify=False,
    )

    aggregated_nuggets = []
    removed_nuggets = 0
    for response, nugget_old in zip(responses, nugget_list):
        if not isinstance(nugget_old, NuggetQuestion):
            # Skip non-question nuggets
            continue

        # Check if nugget has answers
        if not nugget_old.answers:
            continue

        num_answers = len(nugget_old.answers)
        is_multi = num_answers > 1

        if is_multi:
            response = json_loads(response)
            answers_judged, agg = response
            idx_answers_to_remove = []
            for idx, label in answers_judged.items():
                if label.strip().upper() == "NO":
                    idx_answers_to_remove.append(idx)
            agg_nugget = aggregate_answers(nugget_old, idx_answers_to_remove, agg)
        else:
            response_upper = response.strip().upper()
            if response_upper.startswith("NO"):
                answer = list(nugget_old.answers.keys())[0]
                print(f"Nugget '{nugget_old.question}' has invalid answer '{answer}', skipping")
                removed_nuggets += 1
                continue  # Skip this nugget entirely
            agg_nugget = set_aggregator(nugget_old, "OR")

        if agg_nugget.answers and len(agg_nugget.answers) > 0:
            aggregated_nuggets.append(agg_nugget)
        else:
            print(
                f"Nugget '{nugget_old.question}' has 0/{num_answers} answers after aggregation, skipping"
            )
            removed_nuggets += 1

            continue

    # Create a new bank with aggregated nuggets
    aggregated_bank = copy(nugget_bank)
    aggregated_bank.nugget_bank = {}
    aggregated_bank.add_nuggets(aggregated_nuggets)

    if removed_nuggets > 0:
        print(f"{removed_nuggets} nuggets removed during aggregation, due to no valid answers.")

    return aggregated_bank


def set_aggregator(nugget_question: NuggetQuestion, aggregator: str):
    new_question = copy(nugget_question)
    new_question.aggregator_type = AggregatorType(aggregator.upper())
    return new_question


def aggregate_answers(
    nugget_question: NuggetQuestion, idx_answers_to_remove: list[str], aggregator: str
) -> NuggetQuestion:
    """Aggregate answers for a NuggetQuestion by removing specified answers and setting aggregator."""

    if not nugget_question.answers:
        return set_aggregator(nugget_question, aggregator)

    # Convert to list for indexing
    answers_list = list(nugget_question.answers.items())

    # Remove answers by index (convert string indices to int)
    indices_to_remove = set()
    for idx in idx_answers_to_remove:
        try:
            # Convert 1-based index to 0-based
            zero_based_idx = int(idx) - 1
            if 0 <= zero_based_idx < len(answers_list):
                indices_to_remove.add(zero_based_idx)
        except (ValueError, TypeError):
            print(f"Warning: Invalid index '{idx}' in {nugget_question.question}, skipping")
            continue

    # Keep only non-removed answers
    filtered_answers = {}
    num_removed = 0
    for i, (text, obj) in enumerate(answers_list):
        if i not in indices_to_remove:
            filtered_answers[text] = obj
        else:
            num_removed += 1

    if num_removed > 0:
        print(
            f"Dropped {num_removed}/{len(answers_list)} bad answers from: {nugget_question.question}"
        )

    # Create new question with filtered answers
    new_question = set_aggregator(nugget_question, aggregator)
    new_question.answers = filtered_answers
    refresh_metadata_nums(new_question)
    return new_question


async def main():
    args = parser.parse_args()
    print(args)

    assert args.output_path.suffix == "", "Error: Output path must be a directory"
    args.output_path.mkdir(parents=True, exist_ok=True)

    if args.merged_path.is_file():
        input_files = [args.merged_path]
        assert args.output_path != args.merged_path.parent
    else:
        input_files = list(args.merged_path.glob("*_nugget_bank.json"))
        assert args.output_path != args.merged_path

    if not input_files:
        print(f"No nugget bank files found in {args.merged_path}")
        return

    # Process each nugget bank file
    for input_file in input_files:
        print(f"--- Processing {input_file} ---")

        output_path = args.output_path / input_file.name

        # Load the nugget bank
        nugget_bank = load_nugget_bank_json(input_file)
        num_nuggets_orig = len(nugget_bank.nugget_bank) if nugget_bank.nugget_bank else 0
        print(f"Original: {num_nuggets_orig} nuggets")

        # Clean the nugget bank
        cleaned_bank, unanswered_bank = clean_nugget_bank(nugget_bank)
        num_nuggets_clean = len(cleaned_bank.nugget_bank) if cleaned_bank.nugget_bank else 0
        print(f"Cleaned: {num_nuggets_clean} nuggets")

        if args.mode == "clean_only":
            # Save the cleaned bank
            output_path = args.output_path / input_file.name
            write_nugget_bank_json(cleaned_bank, output_path)
            print(f"Saved cleaned NuggetBank to {output_path}")
        else:
            # determine aggregator for each answer set
            bank_with_aggregators = await aggregate_nugget_bank(cleaned_bank)
            num_nuggets_aggregated = (
                len(bank_with_aggregators.nugget_bank) if bank_with_aggregators.nugget_bank else 0
            )
            print(f"Aggregated: {num_nuggets_aggregated} nuggets")

            # Save the aggregated bank
            output_path = args.output_path / input_file.name
            write_nugget_bank_json(bank_with_aggregators, output_path)
            print(f"Saved aggregated NuggetBank to {output_path}")

        # Save unanswered nuggets if any
        if unanswered_bank:
            unanswered_dir = input_file.parent / "potentially_unanswerable"
            unanswered_dir.mkdir(parents=True, exist_ok=True)
            unanswered_path = unanswered_dir / input_file.name
            write_nugget_bank_json(unanswered_bank, unanswered_path)
            print(f"Saved unanswered NuggetBank to {unanswered_path}")
        # break  # For now, process only the first file


if __name__ == "__main__":
    asyncio.run(main())
