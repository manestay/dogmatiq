import argparse
import asyncio
import hashlib
import os
from pathlib import Path

from argue_eval.argue_eval.validation import save_scores_to_metadata
from argue_eval.argue_eval.validation.nugget_data import (
    NuggetQuestion,
    load_nugget_bank_json,
    write_nugget_bank_json,
)
from config import DEV_REQUEST_FILE, NEUCLIR_REQUEST_FOLDER
from evaluator import NuggetBankEvaluator
from run_example import load_requests

NUGGET_BANK_PATH = Path("<PATH_TO_NUGGET_BANK_DIRECTORY_OR_FILE>")
REQUESTS_PATH = Path(os.path.join(NEUCLIR_REQUEST_FOLDER, DEV_REQUEST_FILE))

parser = argparse.ArgumentParser(description="Run on generated nuggets")
parser.add_argument(
    "-i",
    "--nugget_bank_path",
    type=Path,
    default=NUGGET_BANK_PATH,
    help="Path to a single nugget bank or a directory containing multiple",
)
parser.add_argument(
    "--experiment_id",
    type=str,
    default="neuclir_dev_grounded",
    help="Experiment ID for saving results",
)
parser.add_argument(
    "-r",
    "--requests_path",
    type=Path,
    default=REQUESTS_PATH,
    help="Path to the requests file containing query IDs and metadata",
)
parser.add_argument(
    "-o",
    "--save_dir",
    type=Path,
    default=None,
    help="Directory to save the evaluation results",
)
parser.add_argument(
    "--limit",
    type=int,
    default=-1,
    help="Limit the number of queries to process. Use -1 for no limit.",
)
parser.add_argument(
    "-rop",
    "--run_on_paraphrases",
    action="store_true",
    help="Run on paraphrases as well as original nugget questions",
)
parser.add_argument(
    "--no_overwrite",
    dest="overwrite",
    action="store_false",
    help="Skip processing if output file already exists",
)
parser.add_argument(
    "--task",
    type=str,
    default="ragtime",
    choices=["ragtime", "biogen", "rag25"],
    help="task type to process (default: ragtime)",
)


def create_subnugget(nugget_question: NuggetQuestion, new_text: str) -> NuggetQuestion:
    """Create a sub-nugget from an existing nugget question with a new text, deleting metadata."""
    nq_copy = nugget_question.model_copy()
    attributes_to_clear = ["metadata", "answers", "sub_nuggets", "creator"]
    for attr_name in attributes_to_clear:
        if hasattr(nq_copy, attr_name):
            setattr(nq_copy, attr_name, None)
    nq_copy.question = new_text
    nq_copy.question_id = hashlib.md5(new_text.encode()).hexdigest()
    nq_copy.metadata = {"parent": nugget_question.question}
    return nq_copy


async def main(nugget_banks):
    for nugget_bank in nugget_banks:
        query_id = nugget_bank.query_id

        # Check if output file already exists and skip if no_overwrite is set
        if args.run_on_paraphrases:
            save_fname = f"{query_id}_nugget_bank_and_paraphrases_with_scores.json"
        else:
            save_fname = f"{query_id}_nugget_bank_with_scores.json"

        save_dir = args.save_dir if args.save_dir else nugget_bank_dir
        save_path = save_dir / save_fname

        if not args.overwrite and save_path.exists():
            print(f"Output file {save_path} already exists, skipping")
            continue

        nugget_questions = list(x.question for x in nugget_bank.nugget_bank.values())
        if not nugget_questions:
            print(f"No nuggets found for query {query_id}. Skipping.")
            continue

        if args.limit != -1:
            print(f"Using {args.limit}/{len(nugget_questions)} nuggets for query {query_id}")
            nugget_questions = nugget_questions[: args.limit]
        else:
            print(f"Using all {len(nugget_questions)} nuggets for query {query_id}")

        num_nuggets = len(nugget_questions)

        # load paraphrase questions if available
        if args.run_on_paraphrases:
            first_nugget = next(iter(nugget_bank.nugget_bank.values()))
            seen_qs = set(nugget_questions)
            para_to_canon = {}  # map from paraphrase to canonical question
            for nq in nugget_bank.nugget_bank.values():
                canon_q = nq.question
                if canon_q not in seen_qs:
                    continue
                para_objs = []
                for para_q in nq.metadata.get("paraphrases", []):
                    if para_q in seen_qs:
                        continue
                    assert para_q not in para_to_canon
                    para_to_canon[para_q] = canon_q
                    seen_qs.add(para_q)
                    para_obj = create_subnugget(nq, new_text=para_q)
                    para_objs.append(para_obj)
                # Add sub_nuggets to the correct nugget question
                if para_objs:
                    nq.sub_nuggets = nq.sub_nuggets or []
                    nq.sub_nuggets.extend(para_objs)
            num_nuggets_orig = num_nuggets

            nugget_questions.extend(para_to_canon.keys())
            num_nuggets = len(nugget_questions)
            print(
                f"Added {num_nuggets - num_nuggets_orig} paraphrase questions, total now {num_nuggets}"
            )

        problem = query_map[query_id]["problem_statement"]
        report_queries = [problem] * num_nuggets

        story = query_map[query_id]["background"]
        user_stories = [story] * num_nuggets

        evaluator = NuggetBankEvaluator(
            nugget_list=nugget_questions,
            report_request_list=report_queries,
            user_story_list=user_stories,
            experiment_id=args.experiment_id,
            disable_warnings_errors=False,
        )
        nugget_bank_quality_object, list_of_nugget_question_quality_objects = await evaluator.score(
            score_personalization_overall=True,
            return_as_dict=False,
        )
        save_scores_to_metadata(
            nugget_question_or_bank=nugget_bank,
            nugget_question_or_bank_quality=nugget_bank_quality_object,
            nugget_bank_questions_list=nugget_questions,
            nugget_bank_question_quality_list=list_of_nugget_question_quality_objects,
            with_paraphrases=args.run_on_paraphrases,
        )

        nugget_level_scores_only = evaluator.get_only_nugget_level_scores()
        # print("\nNugget-level scores only:", nugget_level_scores_only)

        save_dir.mkdir(parents=True, exist_ok=True)
        write_nugget_bank_json(nugget_bank, save_path)
        print(f"Saved nugget bank with scores to {save_path}")


if __name__ == "__main__":
    args = parser.parse_args()
    query_list = load_requests(args.requests_path, task=args.task)
    query_map = {query["request_id"]: query for query in query_list}

    if args.nugget_bank_path.is_file():
        nugget_bank_dir = args.nugget_bank_path.parent
        stem = args.nugget_bank_path.name.split(".")[0]
        for part in stem.split("_"):  # infer query ID from file name
            if part.isdigit():
                query_id = part
                break
        else:
            raise ValueError(
                f"Could not extract query ID from nugget bank file name {args.nugget_bank_path.name}."
            )
        if query_id not in query_map:
            raise ValueError(
                f"query ID {query_id} from the nugget bank file does not match any query in the query list."
            )

        # Load nugget questions from single file
        nugget_bank = load_nugget_bank_json(args.nugget_bank_path)
        nugget_banks = [nugget_bank]

    else:
        nugget_bank_dir = args.nugget_bank_path
        nugget_banks = []
        for query in query_list:
            query_id = query["request_id"]
            nb_path = nugget_bank_dir / f"{query_id}_nugget_bank.json"
            print(nb_path)
            if not nb_path.exists():
                raise FileNotFoundError(
                    f"Nugget bank file {nb_path} does not exist for query {query_id}."
                )
            nugget_bank = load_nugget_bank_json(nb_path)
            nugget_banks.append(nugget_bank)

    asyncio.run(main(nugget_banks))
