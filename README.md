# 🦉 OWL: Probing Cross-Lingual Recall of Memorized Texts via World Literature

## Overview

Code and Data to support following paper:

Alisha Srivastava\*, Emir Korukluoglu\*, Minh Nhat Le\*, Duyen Tran, Chau Minh Pham, Marzena Karpinska, Mohit Iyyer, "OWL: Probing Cross-Lingual Recall of Memorized Texts via World Literature"

This repository contains necessary pipelines in order to create a Multilingual Aligned Dataset and scripts necessary to evaluate LLM's on Cross-Lingual Knowledge Transfer and Memorization.

Read our paper at https://arxiv.org/abs/2505.22945
## 🚀 Hypotheses & Research Questions

### 1. Translation Memorization

- **Hypothesis:** Large language models (LLMs) memorize the content of translated books.
- **Follow-up Question:** Do the models perform better in English if the original work is in Turkish, Spanish, and Vietnamese?

### 2. Cross-Lingual Memorization

- **Hypothesis:** LLMs can transfer their memorization across languages.
- **Follow-up Question:** Can LLMs memorize translations into languages not present in their pre-training dataset, and will their performance remain strong for out-of-distribution languages?

### 3. Cross-Modality Knowledge Transfer

- **Hypothesis:** LLMs can transfer their knowledge across modalities.
- **Follow-up Question:** Can LLMs transfer knowledge over modalities?

## 👩🏻‍💻 Contributors

<table>
  <tr>
    <td><img src="https://avatars.githubusercontent.com/alishasrivas?s=100" width="100"/></td>
    <td><img src="https://pbs.twimg.com/profile_images/1928267818337812480/gqkgQjPJ_400x400.jpg" width="100"/></td>
    <td><img src="https://avatars.githubusercontent.com/emirkaan5?s=100" width="100"/></td>
  </tr>
  <tr>
    <td><a href="https://github.com/alishasrivas">Alisha Srivastava</a></td>
    <td><a href="https://github.com/nhminle">Nhat Minh Le</a></td>
    <td><a href="https://github.com/emirkaan5">Emir Korukluoglu</a></td>
  </tr>
</table>

### Special Thanks 🌟


- **Chau Minh Pham** - For guiding our research and being our research mentor.
- **Dr. Marzena Karpinska** - For guiding our research and for her invaluable expertise.
- **Dr. Mohit Iyyer** - For guiding our research and being our research advisor.

## 🏗️ Dataset Construction

### Collect 20 books in English, Turkish, Vietnamese, and Spanish.

-- from Project Gutenberg and Online Sources

### Process Books for passages >39 tokens & containing >one Named Entity

- Extract excerpts from different languages while ensuring they contain full sentences and at least single named entity.
- Clean text of metadata and align excerpts across four languages.
- Retain excerpts that pass length checks, contain only one named entity, and verify alignment.

### Expand Dataset to Languages that doesn't have the books in their language

- Using Microsoft Translate, translate English data to Sesotho, Yoruba, Maithili, Malagasy, Setswana, Tahitian.

## Experiments

### 1. Setup

- **Models Used:**
  - OpenAI API for GPT-4o
  - vLLM for Qwen-2.5, LLama-3.1-8B-70B, Llama-3.3-70B, and quantized models
  - OpenRouter for Llama-3.1-405B

### 2. Experiment Types

- **Experiment 0:** Direct Probing
  - Assessing accuracy based on exact and fuzzy matches.
- **Experiment 1:** Name Cloze Task
  - Input excerpts with masked names and evaluate exact matches.
- **Experiment 2:** Prefix Probing/Continuation Generation
  - Prompting models to continue provided sentences and evaluating performance metrics.
- **Experiment 3:** Cross-Modality experiments
  - Conducting first 3 experiments over audio models

## Analyses

- Model capacity and quantization effects.
- Examination of quotes and named entities prevalence.
- Examination of order of words, syntax over knowledge recall
- Investigate how prefix token counts affect model performance.
- Examine the knowledge recall across modalities.

## Contact

For any inquiries or discussions related to this research, please contact:

Emir Korukluoglu at ekorukluoglu@umass.edu
Nhat Minh Le at nhatminhle@umass.edu
Alisha Srivastava at alishasrivas@umass.edu.
