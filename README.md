# Anonymous SIGIR Submission - Code Package

This directory contains the anonymized code and data for our SIGIR submission on document-grounded nugget generation.

## Contents

### Data
- `data/` - Anonymized JSON files containing nugget banks for 5 different systems
  - `ragtime_test_common_claude/`
  - `ragtime_test_dogmatiq_claude/`
  - `ragtime_test_dogmatiq_llama/`
  - `ragtime_test_ginger/`
  - `ragtime_test_random_claude/`

### Pipeline Scripts
- `run_pipeline.sh` - Main pipeline script (anonymized version of original)
- `step1_gen_qa.py` - Generate QA pairs from documents
- `step2_merge_paraphrases.py` - Merge paraphrases and deduplicate questions
- `step3_process_answers.py` - Process and clean answers
- `step4_select_top_nuggets.py` - Select top nuggets using SVC model
- `run_on_nugget_bank.py` - Apply metrics and scoring

### Supporting Files
- `config.py` - Configuration file
- `anonymize_data.py` - Script used to anonymize the data

## Pipeline Overview

The pipeline consists of the following steps:

1. **Document Collection**: Download or prepare document collection
2. **QA Generation**: Generate question-answer pairs from documents using LLMs
3. **Paraphrase Merging**: Identify and merge paraphrased questions
4. **Answer Processing**: Clean and aggregate answers
5. **Nugget Scoring**: Apply trained models to score nugget quality
6. **Top Selection**: Select highest quality nuggets using SVC model

## Anonymization

All personal identifiers, institutional information, and contact details have been removed from the code and data. Hardcoded paths have been replaced with placeholders.

## Note for Reviewers

This code is provided for transparency and reproducibility assessment. The code demonstrates our methodology but may not run directly due to anonymization and dependency requirements. AI assistance, plus our manual verification, was used to put together this README and anonymization.
