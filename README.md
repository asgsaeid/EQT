# Explain-Query-Test: Self-Evaluating LLMs Via Explanation and Comprehension Discrepancy

This repository contains a Python framework of EQT method for self-evaluation of LLMs through concept explanaion and qestion answering. 

## Features
- **Answer Consistency Score (ACS)**: Measure the stability of model predictions across paraphrased questions.
- Implements the **Explain-Query-Test (EQT)** framework:
  - Generate explanations for a given concept.
  - Derive questions from the explanations.
  - Evaluate the model's consistency and accuracy on the questions and their paraphrased versions.
- Supports multiple LLMs, including:
  - OpenAI GPT models
  - Anthropic Claude models
  - Google Gemini models
  - Meta Llama models
- Automated question generation, paraphrasing, and evaluation.
- JSON-based input for managing concepts grouped by categories.

## Installation

### Note
Before running the code, you need to add your own API keys for the language models at the beginning of the script. Replace the placeholder keys with your actual API keys in the following section of the code:
```python
OPENAI_API_KEY = "your_openai_api_key_here"
ANTHROPIC_API_KEY = "your_anthropic_api_key_here"
GOOGLE_API_KEY = "your_google_api_key_here"
LLAMA_API_KEY = "your_llama_api_key_here"
```

1. Clone this repository:
   ```bash
   git clone https://github.com/asgsaeid/EQT
   cd lm-consistency-eval
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
### 1. Prepare the JSON File
The framework requires a JSON file containing concepts grouped by categories. You can use the provided `mmlu_pro_concepts.json` file in the repository as an example:

```json
{
  "business": [
    "Supply Chain Management",
    "Marketing Strategy",
    "Financial Analysis"
  ],
  "psychology": [
    "Cognitive Psychology",
    "Social Psychology",
    "Developmental Psychology"
  ]
}
```

### 2. Run the Script
Execute the script using the command line:
```bash
python main.py --model_name "gpt-4o" --concepts_file "mmlu_pro_concepts.json" --output_dir "results/"
```

#### Arguments
- `--model_name`: The language model to use (e.g., `gpt-4o`).
- `--concepts_file`: Path to the JSON file containing concepts grouped by category.
- `--output_dir`: Directory to save the evaluation results.
- `--num_questions`: Number of questions to generate per concept (default: 10).
- `--num_options`: Number of options per question (default: 10).
- `--num_paraphrases`: Number of paraphrases to generate per question (default: 10).

### Example
To evaluate the `gpt-4o` model on the concepts provided in `mmlu_pro_concepts.json` and save the results in the `results/` directory:
```bash
python main.py --model_name "gpt-4o" --concepts_file "mmlu_pro_concepts.json" --output_dir "results/"
```

## Output
The results are saved in the specified output directory, organized by model name. Each concept has:
- Explanation text (`_explanation.txt`).
- Evaluation results in JSON format (`_results.json`).

## Explain-Query-Test (EQT) Framework
This repository implements the EQT framework as described in the paper "Exploring the Discrepancy Between Explanation and Comprehension in Large Language Models". The EQT framework consists of three stages:
1. **Explain**: Generate a detailed explanation for a given concept.
2. **Query**: Create self-contained, multiple-choice questions from the explanation.
3. **Test**: Evaluate the model's ability to answer the questions and their paraphrased versions.

This approach allows for assessing the depth of a model's comprehension and reasoning capabilities beyond surface-level text generation.

## Citation
If you use this framework in your research, please cite the corresponding paper:

```bibtex
@article{...,
  title={...},
  author={...},
  journal={},
  year={...},
  note={arXiv:2501.11721}
}
```


## Reference
The methodology and metrics used in this repository are based on the paper:

**Exploring the Discrepancy Between Explanation and Comprehension in Large Language Models**

Saeid Asgari Taghanaki, Joao Monteiro

[Read the paper on arXiv](https://arxiv.org/abs/2501.11721).





