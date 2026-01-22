import argparse
import asyncio
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

from argue_eval.utils import batch_model_responses  # Internal evaluation framework
from argue_eval.validation.nugget_data import (  # Internal data validation utilities
    Answer,
    Creator,
    NuggetBank,
    NuggetQuestion,
    Reference,
    write_nugget_bank_json,
)
from config import (
    MERGED_OUTPUT_DIR,
    MODEL_NAME,
    PARAPHRASE_JUDGMENTS_NAME,
    PROVIDER,
    QA_PATH,
    STOP_SEQUENCES,
)
from find_pairs_lib import FindPairs, group_items, invert_group_mapping
from prompts import (
    GET_CANONICAL_QUESTION_SYSTEM,
    GET_CANONICAL_QUESTION_USER,
)

sys.path.append("..")
from lib import is_eng, json_loads

parser = argparse.ArgumentParser(description="Merge paraphrases and generate nugget banks.")
parser.add_argument(
    "-i", "--qa_path", type=Path, default=QA_PATH, help="Path to the Q&A pairs file."
)
parser.add_argument(
    "-o",
    "--merged_output_dir",
    type=Path,
    default=MERGED_OUTPUT_DIR,
    help="Path to save merged output.",
)
parser.add_argument(
    "--paraphrase_judgments_name",
    type=str,
    default=PARAPHRASE_JUDGMENTS_NAME,
    help="Name of the paraphrase judgments file.",
)
parser.add_argument(
    "--no_filter_eng",
    dest="filter_eng",
    action="store_false",
    help="Disable filtering to English text only.",
)
parser.add_argument(
    "--threshold",
    type=float,
    default=0.9,
    help="Similarity threshold for grouping paraphrases.",
)


def merge_answers(existing_answer: Answer, new_answer: Answer, **kwargs) -> Answer:
    answer = Answer.merge_answers(existing_answer, new_answer, **kwargs)
    answer.metadata = None  # Clear metadata to avoid duplication
    return answer


class Question:
    def __init__(
        self,
        text: str,
        answers: dict[str, Answer],
        doc_ids: List[str],
        paraphrases: Optional[List[str]] = None,
        paraphrase_doc_ids: Optional[List[List[str]]] = None,
        canonical_explanation: Optional[str] = None,
    ):
        self.text = text
        # Normalize answers to dict format for consistency
        self.answers = answers
        self.doc_ids = doc_ids
        self.paraphrases = paraphrases if paraphrases is not None else []
        self.paraphrase_doc_ids = paraphrase_doc_ids if paraphrase_doc_ids is not None else []
        # Precompute counts for efficiency
        self.num_answers = len(self.answers)
        self.num_paraphrases_unique = len(self.paraphrases)
        self.num_paraphrases_total = sum(len(doc_ids) for doc_ids in self.paraphrase_doc_ids)
        self.canonical_explanation = canonical_explanation if canonical_explanation else ""

    def __str__(self):
        """String representation for debugging."""
        return f"Question(text={self.text!r}, doc_ids={self.doc_ids}, num_answers={self.num_answers}, num_paraphrases_unique={self.num_paraphrases_unique}, num_paraphrases_total={self.num_paraphrases_total})"

    def to_nugget_question(self, query_id: str) -> NuggetQuestion:
        """Convert Question to NuggetQuestion with custom fields in metadata."""
        paraphrase_references = []
        for doc_ids in self.paraphrase_doc_ids:
            paraphrase_references.append([Reference(doc_id=doc_id) for doc_id in doc_ids])

        return NuggetQuestion(
            query_id=query_id,
            question=self.text,
            answers=self.answers,
            creator=[CREATOR],
            metadata={
                "paraphrases": self.paraphrases,
                "paraphrase_references": paraphrase_references,
                "canonical_explanation": self.canonical_explanation,
                "num_paraphrases_unique": self.num_paraphrases_unique,
                "num_paraphrases_total": self.num_paraphrases_total,
                "num_answers": self.num_answers,
            },
        )

    @staticmethod
    def update_questions(questions_dict: dict, question_text: str, answer_text: str, doc_id: str):
        """Update questions dictionary with new QA pair. Uses dict for O(1) lookup instead of O(n) list search."""
        if question_text in questions_dict:
            question_obj = questions_dict[question_text]
            # Add doc_id if not already present - convert to set for O(1) membership check
            doc_ids_set = set(question_obj.doc_ids)
            if doc_id not in doc_ids_set:
                question_obj.doc_ids.append(doc_id)
            # Create new answer with reference and merge with existing answers
            new_answer = Answer.from_lazy(answer=answer_text, references=[Reference(doc_id=doc_id)])
            # Check if this answer text already exists and merge if so
            if answer_text in question_obj.answers:
                existing_answer = question_obj.answers[answer_text]
                merged_answer = merge_answers(existing_answer, new_answer)
                merged_answer.metadata = None
                question_obj.answers[answer_text] = merged_answer
            else:
                # Add new answer
                question_obj.answers[answer_text] = new_answer
            # Update precomputed count
            question_obj.num_answers = len(question_obj.answers)
        else:
            # Create new question
            answer_obj = Answer.from_lazy(answer=answer_text, references=[Reference(doc_id=doc_id)])
            answer_d = {answer_text: answer_obj}
            question_obj = Question(question_text, answer_d, [doc_id])
            questions_dict[question_text] = question_obj


async def get_canonical(questions_list: list[list[str]], request: str) -> list[tuple[str, str]]:
    """Return a list of (canonical, explanation) pairs from a list of questions."""
    canonical_pairs = [tuple()] * len(questions_list)
    multi_question_indices = []
    user_prompts = []

    # Separate single questions from multi-question groups
    for i, questions in enumerate(questions_list):
        if len(questions) == 1:
            # Single question - use it directly as canonical
            canonical_pairs[i] = (questions[0], "N/A")
        else:
            # Multiple questions - need LLM to select canonical
            multi_question_indices.append(i)
            questions_d = {j: q for j, q in enumerate(questions, 1)}
            questions_str = json.dumps(questions_d, ensure_ascii=False, indent=2)
            user_prompt = GET_CANONICAL_QUESTION_USER.format(
                questions=questions_str, request=request
            )
            user_prompts.append(user_prompt)

    # Process multi-question groups with LLM if any exist
    if user_prompts:
        responses = await batch_model_responses(
            GET_CANONICAL_QUESTION_SYSTEM,
            user_prompts,
            provider=PROVIDER,
            model_name=MODEL_NAME,
            stop=STOP_SEQUENCES,
            max_tokens=None,
            modify=False,
        )

        # Insert LLM responses at correct positions
        response_idx = 0
        for i in multi_question_indices:
            response = responses[response_idx]
            questions = questions_list[i]
            llm_response = json_loads(response, {})
            idx_val = llm_response.get("index")
            try:
                selected_index = int(idx_val) - 1
            except (ValueError, TypeError):
                print(
                    f"WARNING: Could not parse index '{idx_val}' for questions {questions}. Defaulting to first. Explanation: {llm_response.get('explanation')}"
                )
                canonical_pairs[i] = (
                    questions[0],
                    f"WARNING: defaulted to first -- {llm_response.get('explanation')}, {llm_response.get('index')}",
                )
                selected_index = 0
            canonical = questions[selected_index]
            canonical_pairs[i] = (canonical, llm_response["explanation"])
            response_idx += 1

    return canonical_pairs


def verify_doc_ids(question: Question):
    """Verify that the document IDs in paraphrases match those in answers."""
    paraphrases = question.paraphrases
    paraphrase_doc_ids = question.paraphrase_doc_ids
    assert len(paraphrases) == len(paraphrase_doc_ids), (
        f"Number of paraphrases ({len(paraphrases)}) does not match number of paraphrase doc_ids ({len(paraphrase_doc_ids)}) for question: {question.text}"
    )

    # Aggregate all paraphrase doc_ids
    paraphrase_doc_ids_set = set()
    paraphrase_doc_ids_counts = 0
    for doc_ids in paraphrase_doc_ids:
        paraphrase_doc_ids_set.update(doc_ids)
        paraphrase_doc_ids_counts += len(doc_ids)

    # Aggregate all answer doc_ids from references
    answer_doc_ids_set = set()
    answer_counts = 0
    for answer in question.answers.values():
        if answer.references:
            doc_ids = [ref.doc_id for ref in answer.references]
            answer_doc_ids_set.update(doc_ids)
            answer_counts += len(doc_ids)

    # Verify that doc_ids match between paraphrases and answers
    assert answer_doc_ids_set == paraphrase_doc_ids_set, (
        f"ERROR: Answer doc_ids {answer_doc_ids_set} do not match paraphrase doc_ids {paraphrase_doc_ids_set} for question: {question.text}"
    )

    if answer_counts != paraphrase_doc_ids_counts:
        print(
            f"WARNING: Number of answer doc_ids ({answer_counts}) does not match number of paraphrase doc_ids ({paraphrase_doc_ids_counts}) for question: {question.text}"
        )


async def process_request(
    request_info: dict, request_id: str
) -> tuple[List[Question], List[int], List[int]]:
    """Process a request by extracting QA pairs, finding paraphrases, and merging similar questions."""
    questions_dict = {}  # dict for O(1) lookup
    num_qa_pairs = 0
    num_non_eng = 0

    # Extract QA pairs from documents
    for doc_id, doc_info in request_info["docs"].items():
        for qa_pair in doc_info.get("qa_pairs", []):
            question_text = qa_pair["question"]
            answer_text = qa_pair["answer"]
            if args.filter_eng:
                if not is_eng(question_text):
                    num_non_eng += 1
                    print(f"Skipping nugget with non-English question: {question_text}")
                    continue
                if not is_eng(answer_text):
                    num_non_eng += 1
                    print(f"Skipping nugget with non-English answer: {answer_text}")
                    continue

            Question.update_questions(questions_dict, question_text, answer_text, doc_id)
            num_qa_pairs += 1

    print(f"Found {num_qa_pairs} question-answer pairs")
    print(f"Found {len(questions_dict)} unique questions")
    if args.filter_eng:
        print(f"Skipped {num_non_eng} nuggets with non-English content")

    # Convert dict back to list for compatibility with rest of the function
    questions = list(questions_dict.values())

    # Analyze distribution of answers per question
    q_to_ans_count = {
        question.text: sum(len(ans.references or []) for ans in question.answers.values())
        for question in questions
    }
    num_ans_counter = Counter(q_to_ans_count.values())
    print(f"Distribution of number of answers per question: {num_ans_counter}")

    all_questions = [question.text for question in questions]

    # First pass: embedding-based similarity
    FIND_PAIRS.find_similar_pairs(all_questions, request_id)
    similar_pairs_embed = FIND_PAIRS.get_pairs(request_id)
    print(f"Found {len(similar_pairs_embed)} similar pairs based on embeddings")

    # Second pass: LLM verification
    await FIND_PAIRS.check_pairs_with_llm(request_id)
    similar_pairs_llm = FIND_PAIRS.get_pairs(request_id, verified_only=True)
    print(f"Found {len(similar_pairs_llm)} similar pairs based on LLM")

    # Group similar questions using union-find
    question_to_group, max_idx = group_items(
        [(pair["q1"], pair["q2"]) for pair in similar_pairs_llm]
    )

    num_non_singleton_groups = max_idx + 1
    # Add singleton groups for questions that have no pairs
    for question in all_questions:
        if question not in question_to_group:
            question_to_group[question] = max_idx + 1
            max_idx += 1

    group_to_questions = invert_group_mapping(question_to_group)

    print(
        f"Found {num_non_singleton_groups} non-singleton groups out of {len(group_to_questions)} total groups"
    )

    # Merge questions within each group
    merged_questions = []
    question_lookup = {q.text: q for q in questions}

    request = f"{request_info['background']}\n{request_info['problem_statement']}"

    canonical_pairs = await get_canonical(
        [question for question in group_to_questions.values()], request
    )

    for (canonical_question, canonical_explanation), question_texts in zip(
        canonical_pairs, group_to_questions.values()
    ):
        merged_answers_dict = {}
        merged_doc_ids = set()
        merged_paraphrases = []
        merged_paraphrase_doc_ids = []

        # Merge all questions in the group
        for q_text in question_texts:
            q_obj = question_lookup[q_text]
            for answer_text, answer_obj in q_obj.answers.items():
                if answer_text in merged_answers_dict:
                    merged_answers_dict[answer_text] = merge_answers(
                        merged_answers_dict[answer_text], answer_obj
                    )
                else:
                    merged_answers_dict[answer_text] = answer_obj
            merged_doc_ids.update(q_obj.doc_ids)
            # Add the question text as a paraphrase along with its doc_ids
            merged_paraphrases.append(q_text)
            merged_paraphrase_doc_ids.append(q_obj.doc_ids)

        merged_question = Question(
            canonical_question,
            merged_answers_dict,
            list(merged_doc_ids),
            paraphrases=merged_paraphrases,
            paraphrase_doc_ids=merged_paraphrase_doc_ids,
            canonical_explanation=canonical_explanation,
        )
        merged_questions.append(merged_question)

    # Verify data integrity
    [verify_doc_ids(question) for question in merged_questions]

    # Extract statistics
    num_paraphrases_unique = [q_obj.num_paraphrases_unique for q_obj in merged_questions]
    num_paraphrases_total = [q_obj.num_paraphrases_total for q_obj in merged_questions]

    print(f"Number of merged questions: {len(merged_questions)}")

    # Sort by total paraphrases (most common questions first)
    merged_questions.sort(key=lambda q: q.num_paraphrases_total, reverse=True)
    return merged_questions, num_paraphrases_unique, num_paraphrases_total


async def main(
    request_info_map: dict[str, dict[str, Any]],
) -> tuple[dict[str, List[Question]], dict[str, List[int]], dict[str, List[int]]]:
    """Main async function to process all requests in a single event loop."""

    topic_to_merged_qa_map = {}
    num_paraphrases_unique_d = {}
    num_paraphrases_total_d = {}

    for request_id, request_info in request_info_map.items():
        print("-" * 30, f"Processing request {request_id}...", "-" * 30)
        merged_qa_map, num_paraphrases_unique, num_paraphrases_total = await process_request(
            request_info, request_id
        )
        topic_to_merged_qa_map[request_id] = merged_qa_map
        num_paraphrases_unique_d[request_id] = num_paraphrases_unique
        num_paraphrases_total_d[request_id] = num_paraphrases_total

    return topic_to_merged_qa_map, num_paraphrases_unique_d, num_paraphrases_total_d


if __name__ == "__main__":
    args = parser.parse_args()

    # Use argparse paths or default to config values
    qa_path = args.qa_path
    merged_output_dir = args.merged_output_dir
    paraphrase_judgments_name = args.paraphrase_judgments_name
    FIND_PAIRS = FindPairs(similarity_threshold=args.threshold)

    if args.filter_eng:
        print("Filtering out questions or answers that contain any non-English text.")

    CREATOR = Creator(
        is_human=False,
        llm_model=MODEL_NAME,
        llm_backend=PROVIDER,
        # llm_prompt=[GET_CANONICAL_QUESTION_SYSTEM, IDENTIFY_PARAPHRASE_SYSTEM], # too long
        contact=["Author1"],
    )

    with qa_path.open("r") as f:
        request_info_map = json.load(f)

    # get the creation date from qa_path as a datetime object
    mtime = qa_path.stat().st_mtime
    creation_date = datetime.fromtimestamp(mtime)

    CREATOR.set_creation_date(creation_date)

    topic_to_merged_qa_map, num_paraphrases_unique_d, num_paraphrases_total_d = asyncio.run(
        main(request_info_map)
    )

    # Create NuggetBank format for each request
    merged_output_dir.mkdir(parents=True, exist_ok=True)
    for request_id, texts1 in topic_to_merged_qa_map.items():
        nugget_bank = NuggetBank(
            query_id=request_id,
            title_query=request_id,  # You might want to use a more descriptive title
            creator=[CREATOR],
        )

        # Convert each Question to NuggetQuestion and add to bank
        nugget_questions = [q.to_nugget_question(request_id) for q in texts1]
        nugget_bank.add_nuggets(nugget_questions)

        # Save individual NuggetBank file
        nugget_bank_path = merged_output_dir / f"{request_id}_nugget_bank.json"
        write_nugget_bank_json(nugget_bank, nugget_bank_path)
        print(
            f"Request {request_id}: saved NuggetBank with {len(nugget_questions)} questions to {nugget_bank_path}"
        )

    PARAPHRASE_PATH = merged_output_dir / paraphrase_judgments_name
    PARAPHRASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PARAPHRASE_PATH.open("w") as f:
        json.dump(FIND_PAIRS.get_pairs_d(), f, indent=2, ensure_ascii=False)
    print(f"Saved paraphrase judgments to {PARAPHRASE_PATH}")
