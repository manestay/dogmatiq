#!/bin/bash

# Anonymized pipeline script for SIGIR submission
# This script demonstrates the document-grounded nugget generation pipeline

# Configuration - Replace with your own paths
DOCS_DIR=<PATH_TO_DOCUMENTS_DIR>
GROUNDED_NAME=grounded_questions.json
QUESTION_DIR=merged_questions/

# Step 1: Generate questions from documents
# This step processes documents and generates QA pairs - takes significant time
python step1_gen_qa.py --data $DOCS_DIR --requests_path <PATH_TO_TOPICS> --qa_path $GROUNDED_NAME

# Step 2: Merge paraphrases and deduplicate questions
python step2_merge_paraphrases.py -i $GROUNDED_NAME -o $QUESTION_DIR --threshold 0.9 |& tee merge.log

# Step 3: Process and clean answers
python step3_process_answers.py \
    -i ${QUESTION_DIR}/ \
    -o ${QUESTION_DIR}/clean_agg

# Step 4: Apply metrics and scoring (calls external metrics module)
cd ..
export TOKENIZERS_PARALLELISM=true; export PYTHONPATH=.
python metrics/run_on_nugget_bank.py -rop \
    -i doc_grounded/${QUESTION_DIR}/clean_agg \
    -r <PATH_TO_TOPICS> \
    --experiment_id <EXPERIMENT_ID> \
    --no_over

# Step 5: Select top nuggets using trained SVC model
cd doc_grounded
python step4_select_top_nuggets.py \
    -i ${QUESTION_DIR}/clean_agg \
    -o ${QUESTION_DIR}/svc_model \
    --method model \
    --model_package <PATH_TO_SVC_MODEL>

# Step 6: Select top nuggets using hybrid most_common + SVC approach
python step4_select_top_nuggets.py \
    -i ${QUESTION_DIR}/clean_agg \
    -o ${QUESTION_DIR}/most_common_and_svc \
    --method most_common_model \
    --model_package <PATH_TO_SVC_MODEL> --max_most 0.75
