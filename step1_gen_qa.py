import argparse
import asyncio
import json
import sys
from collections import defaultdict
from pathlib import Path

from argue_eval.utils import ModelProvider, batch_model_responses
from config import DATA_DIR, PROVIDER, QA_PATH, REQUESTS_PATH, STOP_SEQUENCES
from prompts import SUMMARIZE_SYSTEM, SUMMARIZE_USER, WRITE_QA_PAIRS_SYSTEM, WRITE_QA_PAIRS_USER

sys.path.append("..")
from lib import json_loads

REQUEST_FIELDS = ["background", "problem_statement", "title"]


def tokenize_split(text):
    """Simple tokenizer that splits on whitespace"""
    return len(text.split())


paths = list(DATA_DIR.glob("*/*.json"))

language_full = {"zho": "Chinese", "fas": "Persian", "rus": "Russian"}


parser = argparse.ArgumentParser(description="Process documents and generate Q&A pairs.")
parser.add_argument("--data_dir", type=Path, default=DATA_DIR, help="Path to the data directory.")
parser.add_argument(
    "--requests_path", type=Path, default=REQUESTS_PATH, help="Path to the requests file."
)
parser.add_argument("--qa_path", type=Path, default=QA_PATH, help="Path to save the Q&A pairs.")
parser.add_argument(
    "--task",
    type=str,
    default="ragtime",
    choices=["ragtime", "biogen", "rag25"],
    help="task type to process (default: ragtime)",
)
parser.add_argument(
    "--token_count",
    action="store_true",
    help="Count tokens instead of making API calls",
)
parser.add_argument(
    "--provider",
    type=str,
    default=ModelProvider.HLTCOE_LOCAL,
    help="Model provider to use (see ModelProvider)",
)


async def process_documents_async(paths, requests_d, provider):
    """Process documents with async model calls"""
    request_info_map = defaultdict(lambda: defaultdict(dict))
    for rid, d in requests_d.items():
        for field in REQUEST_FIELDS:
            request_info_map[rid][field] = d[field]

    if args.task == "biogen":
        print("Loading documents...")
        for path in paths:
            request_id = path.parent.name
            doc_id = path.stem
            with open(path, "r") as f:
                data = json.load(f)

            language = language_full.get(data["language"])
            out_d = {}
            out_d["language"] = language
            out_d["summary"] = data["contents"]
            request_info_map[request_id]["docs"][doc_id] = out_d
    elif args.task == "rag25":
        print("Loading RAG25 documents...")
        for path in paths:
            request_id = path.parent.name
            doc_id = path.stem
            with open(path, "r") as f:
                data = json.load(f)

            language = data["language"]
            out_d = {}
            out_d["language"] = language
            out_d["summary"] = data["text"]
            request_info_map[request_id]["docs"][doc_id] = out_d
    else:  # ragtime (default)
        print("Summarizing documents...")
        system_prompt = SUMMARIZE_SYSTEM
        user_prompts = []
        request_doc_ids = []
        for path in paths:
            request_id = path.parent.name
            doc_id = path.stem
            with open(path, "r") as f:
                data = json.load(f)
            language = language_full.get(data["language"])

            out_d = {}
            out_d["language"] = language

            doc = data["text"]
            user_prompt = SUMMARIZE_USER.format(language=language, document=doc)
            # Collect prompts for batch processing
            user_prompts.append(user_prompt)

            request_info_map[request_id]["docs"][doc_id] = out_d
            request_doc_ids.append((request_id, doc_id))

        if args.token_count:
            # Count tokens instead of making API calls
            total_tokens = sum(
                tokenize_split(system_prompt) + tokenize_split(prompt) for prompt in user_prompts
            )
            print(f"Num documents: {len(user_prompts)}")
            print(
                f"Summarization tokens: {total_tokens} ; average: {total_tokens / len(user_prompts)}"
            )
            # Use empty responses for token counting mode
            responses = [""] * len(user_prompts)
        else:
            responses = await batch_model_responses(
                system_prompt,
                user_prompts,
                provider=provider,
                stop=STOP_SEQUENCES,
                max_tokens=None if provider != ModelProvider.ANTHROPIC else 1024,
                modify=False,
            )

        for (request_id, doc_id), response in zip(request_doc_ids, responses):
            request_info_map[request_id]["docs"][doc_id]["summary"] = response

    # now, let's generate Q&A pairs
    print("Generating Q&A pairs...")
    system_prompt = WRITE_QA_PAIRS_SYSTEM
    user_prompts = []
    request_doc_ids = []

    for request_id, doc_info in request_info_map.items():
        for doc_id in doc_info["docs"].keys():
            request_doc_ids.append((request_id, doc_id))
            curr_doc = doc_info["docs"][doc_id]
            if "summary" not in curr_doc:
                continue
            summary = curr_doc["summary"]
            request = f"{doc_info['background']}\n{doc_info['problem_statement']}"

            user_prompt = WRITE_QA_PAIRS_USER.format(
                document=summary,
                request=request,
            )
            user_prompts.append(user_prompt)

    if args.token_count:
        # Count tokens for Q&A generation instead of making API calls
        total_tokens = sum(
            tokenize_split(system_prompt) + tokenize_split(prompt) for prompt in user_prompts
        )
        print(
            f"Q&A generation tokens: {total_tokens} ; average: {total_tokens / len(user_prompts)}"
        )
        # Use empty responses for token counting mode
        qa_responses = ["[]"] * len(user_prompts)  # Empty JSON array for Q&A pairs
    else:
        qa_responses = await batch_model_responses(
            system_prompt,
            user_prompts,
            provider=provider,
            stop=STOP_SEQUENCES,
            max_tokens=None if provider != ModelProvider.ANTHROPIC else 1024,
            modify=False,
        )

    for (request_id, doc_id), response in zip(request_doc_ids, qa_responses):
        qa_pairs = json_loads(response)
        request_info_map[request_id]["docs"][doc_id]["qa_pairs"] = qa_pairs

    return request_info_map


def load_requests(requests_path):
    # Load request data
    requests_d = {}
    with open(requests_path, "r") as f:
        if args.task == "biogen":
            requests = json.load(f)
            for request in requests:
                request_id = str(request["id"])
                requests_d[request_id] = {
                    "request_id": request_id,
                    "background": request["narrative"],
                    "problem_statement": request["question"],
                    "title": request["topic"],
                }
        elif args.task == "rag25":
            for line in f:
                line = json.loads(line)
                request_id = str(line["id"])
                requests_d[request_id] = {
                    "request_id": request_id,
                    "background": "",
                    "problem_statement": line["title"],
                    "title": "",
                }
        else:  # ragtime (default)
            for line in f:
                line = json.loads(line)
                requests_d[line["request_id"]] = line

                if "backround" in line:  # some files have a typo
                    line["background"] = line["backround"]
                    del line["backround"]

    return requests_d


if __name__ == "__main__":
    args = parser.parse_args()

    # Use argparse paths or default to config values
    data_dir = args.data_dir
    requests_path = args.requests_path
    qa_path = args.qa_path

    paths = list(data_dir.glob("*/*.json"))
    if not paths:
        paths = list(data_dir.glob("*/*.jsonl"))

    requests_d = load_requests(requests_path)

    # use only the paths which have a request_id in requests_d
    request_ids = set(requests_d.keys())
    paths = [p for p in paths if p.parent.name in request_ids]

    print(f"Processing {len(paths)} documents from {data_dir}")

    # Run async processing
    request_info_map = asyncio.run(process_documents_async(paths, requests_d, args.provider))

    if not args.token_count:
        # Save the results to a file
        with open(qa_path, "w") as f:
            json.dump(request_info_map, f, indent=2, ensure_ascii=False)
        print(f"Saved Q&A pairs to {qa_path}")
