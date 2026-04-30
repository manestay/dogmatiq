# DoGMaTiQ: Automated Generation of Question-and-Answer Nuggets for Report Evaluation

This is the official repository for the paper "DoGMaTiQ: Automated Generation of Question-and-Answer Nuggets for Report Evaluation" (arXiv, 2026). It will contain datasets, code, prompts, and other artifacts to reproduce our results.

NOTE: This repository is currently WIP and under construction. Currently, it includes the nugget bank datasets created for the RAGTIME25 shared task ([link](https://trec-ragtime.github.io/)), as well as scripts to run the DoGMaTiQ pipeline.

## Contents

### Data
- `data/` - JSON files containing nugget banks for 5 different systems evaluated on RAGTIME25 test topics
  - `ragtime_test_common_claude/` - Nuggets selected by most-common voting, generated with Claude
  - `ragtime_test_dogmatiq_claude/` - Nuggets selected by the DoGMaTiQ SVC model, generated with Claude
  - `ragtime_test_dogmatiq_llama/` - Nuggets selected by the DoGMaTiQ SVC model, generated with Llama
  - `ragtime_test_ginger/` - Nugget banks derived from the GINGER response generation system
  - `ragtime_test_random_claude/` - Nuggets selected randomly (baseline), generated with Claude

### Pipeline Scripts
- `run_pipeline.sh` - Main pipeline script
- `step1_gen_qa.py` - Generate QA pairs from documents using an LLM
- `step2_merge_paraphrases.py` - Merge paraphrases and deduplicate questions
- `step3_process_answers.py` - Process and clean answers
- `step4_select_top_nuggets.py` - Select top nuggets using the SVC model
- `run_on_nugget_bank.py` - Apply metrics and scoring to a nugget bank

### Supporting Files
- `config.py` - Configuration file

## Pipeline Overview

The pipeline consists of the following steps:

1. **Document Collection**: Download or prepare the document collection
2. **QA Generation**: Generate question-answer pairs from documents using an LLM
3. **Paraphrase Merging**: Identify and merge paraphrased questions
4. **Answer Processing**: Clean and aggregate answers
5. **Nugget Scoring**: Apply trained models to score nugget quality
6. **Top Selection**: Select highest-quality nuggets using the SVC model

## Quality Criteria

WIP

## AutoArgue Evaluation

WIP

## Baselines

WIP
