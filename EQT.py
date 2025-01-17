import os
import time
import re
import json
import numpy as np
import argparse
import openai
import anthropic
import google.generativeai as genai

# ------------------------------ #
#       API KEY CONFIGURATION     #
# ------------------------------ #

# Define your API keys here. Replace the placeholder strings with your actual API keys.
OPENAI_API_KEY = "your_openai_api_key_here"
ANTHROPIC_API_KEY = "your_anthropic_api_key_here"
GOOGLE_API_KEY = "your_google_api_key_here"
LLAMA_API_KEY = "your_llama_api_key_here"

# ------------------------------ #
#       Language Model Classes    #
# ------------------------------ #

# Base class for language models
class LanguageModel:
    def __init__(self, model_name):
        self.model_name = model_name

    def get_completion(self, prompt, max_tokens=500, temperature=0, stop=None):
        raise NotImplementedError("This method should be overridden by subclasses.")

# OpenAI GPT Models
class OpenAIModel(LanguageModel):
    def __init__(self, model_name):
        super().__init__(model_name)
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not set.")
        openai.api_key = OPENAI_API_KEY

    def get_completion(self, prompt, max_tokens=500, temperature=0, stop=None):
        try:
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop,
            )
            return response['choices'][0]['message']['content'].strip()
        except openai.error.OpenAIError as e:
            print(f"OpenAI API error: {e}")
            return ""


class AnthropicModel(LanguageModel):
    def __init__(self, model_name):
        super().__init__(model_name)
        if not ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY is not set.")
        self.client = anthropic.Client(api_key=ANTHROPIC_API_KEY)

    def get_completion(self, prompt, max_tokens=500, temperature=0, stop=None):
        try:
            # Check if the model requires the Messages API instead of the Completions API
            if self.model_name in ["Claude-3-5-Sonnet-20240620", "claude-3-sonnet-20240229", "claude-3-opus-20240229",
                                   "Claude-3-5-Sonnet-20241022"]:
                # Use the Messages API for chat-based models
                response = self.client.messages.create(
                    model=self.model_name,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    stop_sequences=stop if stop else [],
                )
                # The content is now directly accessible from the response
                return response.content[0].text.strip()
            else:
                # Use the Completions API for other models
                response = self.client.completions.create(
                    model=self.model_name,
                    prompt=f"{anthropic.HUMAN_PROMPT} {prompt}{anthropic.AI_PROMPT}",
                    max_tokens_to_sample=max_tokens,
                    temperature=temperature,
                    stop_sequences=stop if stop else [],
                )
                return response.completion.strip()
        except Exception as e:
            print(f"Anthropic API error: {e}")
            return ""


class GeminiModel(LanguageModel):
    def __init__(self, model_name):
        super().__init__(model_name)
        genai.configure(api_key=GOOGLE_API_KEY)
        generation_config = {
            "temperature": 0.0,
            "top_p": 1,
            "max_output_tokens": 4000,
            "response_mime_type": "text/plain",
        }
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
        self.client = genai.GenerativeModel(
            model_name=model_name,
            safety_settings=safety_settings,
            generation_config=generation_config,
        )

    def get_completion(self, prompt, max_tokens=4000, temperature=0, stop=None):
        # Create a chat session and get response using the Gemini API
        try:
            chat_session = self.client.start_chat(history=[])
            response = chat_session.send_message(prompt)
            return response.text.strip()
        except Exception as e:
            print(f"Google Gemini API error: {e}")
            return ""

# Llama Models via API (e.g., DeepInfra's OpenAI-Compatible API)
class LlamaModel(LanguageModel):
    def __init__(self, model_name):
        super().__init__(model_name)
        if not LLAMA_API_KEY:
            raise ValueError("LLAMA_API_KEY is not set.")
        # Initialize OpenAI-compatible client with Llama-specific API key
        openai.api_key = LLAMA_API_KEY
        # Set the base URL for Llama API (e.g., DeepInfra)
        self.base_url = "https://api.deepinfra.com/v1/openai"
        openai.api_base = self.base_url

    def get_completion(self, prompt, max_tokens=500, temperature=0, stop=None):
        try:
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop,
            )
            return response['choices'][0]['message']['content'].strip()
        except openai.error.OpenAIError as e:
            print(f"Llama API error: {e}")
            return ""


# ------------------------------ #
#          Factory Function      #
# ------------------------------ #

def get_language_model(model_name):
    model_name_lower = model_name.lower()
    if 'gpt' in model_name_lower or 'openai' in model_name_lower or 'o1' in model_name_lower:
        return OpenAIModel(model_name)
    elif 'claude' in model_name_lower or 'anthropic' in model_name_lower:
        return AnthropicModel(model_name)
    elif 'gemini' in model_name_lower or 'google' in model_name_lower:
        return GeminiModel(model_name)
    elif 'llama' in model_name_lower or 'meta-llama' in model_name_lower:
        return LlamaModel(model_name)
    else:
        raise ValueError(f"Unsupported model: {model_name}")


# ------------------------------ #
#      Helper Functions          #
# ------------------------------ #

# Function to extract answers from LLM responses
def extract_answer(text):
    # Try to find "Answer: A, B, C"
    pattern = r"Answer\s*:\s*([A-J](?:,\s*[A-J])*)"
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return [opt.strip().upper() for opt in match.group(1).split(',')]
    else:
        return extract_again(text)


def extract_again(text):
    # Try to find options mentioned directly
    matches = re.findall(r'\b([A-J])\b', text.upper())
    if matches:
        return list(set(matches))  # Return unique options found
    else:
        return []


# Function to get the LLM's explanation of a concept
def get_explanation(concept, lm):
    prompt = f"""Please provide a comprehensive and detailed explanation of the concept '{concept}', including its 
    background, key principles, applications, and examples. Ensure the explanation is thorough and covers all 
    essential aspects."""
    explanation = lm.get_completion(prompt, temperature=0, max_tokens=1500)
    return explanation


def generate_questions(explanation, lm, num_questions=10, num_options=10):
    choices = [chr(i) for i in range(65, 65 + num_options)]  # ['A', 'B', 'C', ...]
    questions = []

    question_types = [
        "Define the concept of",
        "Explain the key characteristics of",
        "Describe how {} relates to",
        "Provide an example of",
        "Discuss the significance of",
        "Compare and contrast",
        "Outline the process of",
        "List the applications of",
        "Identify the limitations of",
        "Summarize the historical development of"
    ]

    for i in range(num_questions):
        question_type = question_types[i % len(question_types)]

        question_prompt = f"""Create a multiple-choice question about the following concept: [CONCEPT].
        Use this question type: "{question_type}"

        Requirements:
        1. The question should have {num_options} options (A to {choices[-1]}).
        2. There should be 1 to 3 correct answers.
        3. Base the question and all options ONLY on the information provided below.
        4. Make the question entirely self-contained. Do NOT refer to any explanation, provided information, or external context.
        5. Avoid phrases like "according to the text", "as described", or any similar references.
        6. Ensure the question and options are clear and complete on their own.

        Format your response exactly as follows:
        Question: [Your question here]
        Options:
        A) Option A text
        B) Option B text
        ...
        {choices[-1]}) Option {choices[-1]} text
        Correct Answers: [List of correct option letters, e.g., A, C, F]

        Information about [CONCEPT]:
        {explanation}
        """
        generated = lm.get_completion(question_prompt, temperature=0.7, max_tokens=1000)
        question_data = parse_generated_question(generated, num_options)
        if question_data:
            questions.append(question_data)
        else:
            print(f"Failed to parse question {i + 1}. Generated text:\n{generated}")

        # Add delay to respect rate limits
        time.sleep(1)

    return questions


def parse_generated_question(generated_text, num_options):
    choices = [chr(i) for i in range(65, 65 + num_options)]
    try:
        lines = generated_text.strip().split('\n')
        lines = [line for line in lines if line.strip() != '']

        question_line_idx = next((i for i, line in enumerate(lines) if line.startswith('Question:')), None)
        options_start_idx = next((i for i, line in enumerate(lines) if line.startswith('Options:')), None)
        correct_answers_idx = next((i for i, line in enumerate(lines) if line.startswith('Correct Answers:')), None)

        if question_line_idx is None or options_start_idx is None or correct_answers_idx is None:
            print("Failed to find all necessary sections in generated question.")
            return None

        question_text = lines[question_line_idx].replace('Question:', '').strip()
        options = [line.strip() for line in lines[options_start_idx + 1:correct_answers_idx] if
                   line[0] in choices and line[1] == ')']
        correct_answers = [ans.strip().upper() for ans in
                           lines[correct_answers_idx].replace('Correct Answers:', '').strip().split(',')]

        if len(options) != num_options or not all(letter in choices for letter in correct_answers):
            print(f"Invalid question format. Options: {len(options)}, Correct Answers: {correct_answers}")
            return None

        return {
            "question": question_text,
            "options": options,
            "correct_answers": correct_answers
        }
    except Exception as e:
        print(f"Error parsing question: {e}")
        return None

# Function to paraphrase questions
def paraphrase_question(question_text, lm):
    # Define the prompt to paraphrase the question without extra text
    prompt = f"""Paraphrase the following question without changing its meaning. 
    Ensure the paraphrased question is self-contained and does not reference any previous 
    explanation or use phrases like 'as mentioned earlier'. ONLY generate the paraphrased question itself, 
    and do not include any extra text such as 'Here's a paraphrased version of the question:' or similar.

'{question_text}'
"""
    for _ in range(3):  # Attempt paraphrasing up to 3 times if the output is not acceptable
        paraphrased_question = lm.get_completion(prompt, temperature=0.7, max_tokens=500).strip('"').strip("'").strip()

        # Check if the paraphrased question is exactly the same as the original
        if paraphrased_question.lower() != question_text.lower():
            # Ensure no unwanted explanatory text is included in the paraphrased question
            # Remove unwanted prefixes like "Here's a paraphrased version of the question:"
            paraphrased_question = re.sub(r"^(here('|’)s|this is|here is|a paraphrased version of the question|paraphrased:|original:|rephrased:).*", "", paraphrased_question, flags=re.IGNORECASE).strip()

            # Ensure no unwanted suffixes like "This is the paraphrased version."
            paraphrased_question = re.sub(r"(this is|this was|here is|end of|paraphrased|rephrased|paraphrased version).*", "", paraphrased_question, flags=re.IGNORECASE).strip()

            # If paraphrased question is valid, return it
            if paraphrased_question and paraphrased_question.lower() != question_text.lower():
                return paraphrased_question

    # Return None if all attempts fail
    print(f"Failed to generate a valid paraphrase for: {question_text}")
    return None


# Function to ask the LLM the questions and collect answers
def ask_questions(questions, lm):
    choices = [chr(i) for i in range(65, 75)]  # ['A', 'B', 'C', ..., 'J']
    answers = []
    for idx, q in enumerate(questions):
        prompt = f"{q['question']}\nOptions:\n"
        for option in q['options']:
            prompt += f"{option}\n"
        prompt += ("\nPlease select all correct options (e.g., A, C, D) and provide your answer in the format: "
                   "'Answer: [Your selections]'.")
        response = lm.get_completion(prompt, temperature=0, max_tokens=200)
        llm_response = response.strip()
        selected_options = extract_answer(llm_response)
        # Validate selected options
        option_letters = [option.split(')')[0] for option in q['options']]
        selected_options = [opt for opt in selected_options if opt in option_letters]
        answers.append({
            "question_index": idx,
            "question": q['question'],
            "options": q['options'],
            "correct_answers": q['correct_answers'],
            "predicted_answers": selected_options,
            "llm_response": llm_response
        })
        # Add a short delay to respect rate limits
        time.sleep(1)
    return answers


# Function to check if a results file for a given concept already exists
def check_results_file_exists(concept_name, results_dir):
    safe_concept_name = concept_name.replace(' ', '_').replace("'", "")
    output_filename = os.path.join(results_dir, f"{safe_concept_name}_results.json")
    print(f"Checking for existing results file: {output_filename}")
    return os.path.isfile(output_filename)


# ------------------------------ #
#        Experiment Function     #
# ------------------------------ #

def run_experiment(concepts_by_category, model_name, output_dir, num_questions, num_options, num_paraphrases):
    lm = get_language_model(model_name)
    safe_model_name = model_name.replace('/', '_').replace(' ', '_')
    results_dir = os.path.join(output_dir, f"{safe_model_name}")
    os.makedirs(results_dir, exist_ok=True)

    for category, concepts in concepts_by_category.items():
        print(f"\nProcessing Category: {category}\n{'=' * 80}\n")

        for concept in concepts:
            if check_results_file_exists(concept, results_dir):
                print(f"Results for concept '{concept}' already exist. Skipping...\n")
                continue

            print(f"Running experiment for concept: {concept}\n")

            # Step 1: Get and save explanation
            explanation = get_explanation(concept, lm)
            safe_concept_name = concept.replace(' ', '_').replace("'", "")
            explanation_filename = os.path.join(results_dir, f"{safe_concept_name}_explanation.txt")
            with open(explanation_filename, 'w', encoding='utf-8') as file:
                file.write(explanation)

            # Step 2: Generate questions
            original_questions = generate_questions(explanation, lm, num_questions=num_questions,
                                                    num_options=num_options)

            # Step 3 and 4: Paraphrase questions, ask questions, and evaluate
            results = []
            answer_stability_scores = []

            for q_id, original_q in enumerate(original_questions):
                # Process original question
                original_answer = ask_questions([original_q], lm)[0]
                original_result = evaluate_answer(original_q, original_answer)
                original_result.update({
                    "question_id": q_id,
                    "is_original": True,
                    "paraphrase_id": None
                })
                results.append(original_result)

                # Process paraphrased questions
                paraphrase_results = []
                for p_id in range(num_paraphrases):
                    paraphrased_text = paraphrase_question(original_q['question'], lm)
                    if paraphrased_text:
                        paraphrased_q = {
                            "question": paraphrased_text,
                            "options": original_q['options'],
                            "correct_answers": original_q['correct_answers']
                        }
                        paraphrased_answer = ask_questions([paraphrased_q], lm)[0]
                        paraphrased_result = evaluate_answer(paraphrased_q, paraphrased_answer)
                        paraphrased_result.update({
                            "question_id": q_id,
                            "is_original": False,
                            "paraphrase_id": p_id
                        })
                        results.append(paraphrased_result)
                        paraphrase_results.append(paraphrased_result)
                    time.sleep(1)  # Delay to respect rate limits

                # Calculate answer stability score for this question group
                answer_stability_score = calculate_answer_stability_score(original_result, paraphrase_results)
                answer_stability_scores.append(answer_stability_score)

            # Calculate overall metrics
            accuracy = sum(r['is_correct'] for r in results) / len(results)
            avg_answer_stability_score = sum(answer_stability_scores) / len(answer_stability_scores)

            # Save results
            output_filename = os.path.join(results_dir, f"{safe_concept_name}_results.json")
            output_data = {
                "concept": concept,
                "accuracy": accuracy,
                "total_questions": len(results),
                "avg_answer_stability_score": avg_answer_stability_score,
                "results": results
            }
            with open(output_filename, 'w', encoding='utf-8') as outfile:
                json.dump(output_data, outfile, indent=4)
            print(f"Results saved to '{output_filename}'\n")

    print("\nExperiment completed.")


def evaluate_answer(question, answer):
    selected_options = extract_answer(answer['llm_response'])
    correct = set(selected_options) == set(question['correct_answers']) if selected_options else False
    return {
        "question": question['question'],
        "options": question['options'],
        "correct_answers": question['correct_answers'],
        "predicted_answers": selected_options,
        "is_correct": correct,
        "llm_response": answer['llm_response']
    }


def calculate_answer_stability_score(original_result, paraphrase_results):
    """
    Measures how stable the model's answers are across paraphrases of the same question.
    1 means all answers (including original) were the same, 0 means all were different.
    """
    all_answers = [original_result['predicted_answers']] + [p['predicted_answers'] for p in paraphrase_results]
    unique_answers = set(tuple(ans) for ans in all_answers)  # Convert lists to tuples for set
    return 1 - (len(unique_answers) - 1) / len(all_answers)



# ------------------------------ #
#        JSON Loading Function   #
# ------------------------------ #

def load_concepts_from_json(file_path):
    """
    Load concepts grouped by category from a JSON file.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        dict: A dictionary with categories as keys and lists of concepts as values.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            concepts_by_category = json.load(file)
        print(f"Successfully loaded concepts from {file_path}.")
        return concepts_by_category
    except Exception as e:
        print(f"Error loading concepts JSON file: {e}")
        return {}


# ------------------------------ #
#        Main Execution          #
# ------------------------------ #

def main():
    parser = argparse.ArgumentParser(description="Run experiments across multiple language models.")
    parser.add_argument("--output_dir", "-o", type=str, default="eval_results_general/",
                        help="Directory to save evaluation results.")
    parser.add_argument("--model_name", "-m", type=str, required=True,
                        choices=["gpt-4o", "o1-preview", "gemini-1.5-pro-latest",
                                 "Claude-3-5-Sonnet-20240620", "Claude-3-5-Sonnet-20241022",
                                 "meta-llama/Meta-Llama-3.1-405B-Instruct", "nvidia/Llama-3.1-Nemotron-70B-Instruct"],
                        help="Name of the model to use.")
    parser.add_argument("--concepts_file", "-c", type=str, required=True,
                        help="Path to the JSON file containing concepts grouped by category.")
    parser.add_argument("--num_questions", "-q", type=int, default=10,
                        help="Number of questions to generate for each concept.")
    parser.add_argument("--num_options", "-opt", type=int, default=10,
                        help="Number of options for each question.")
    parser.add_argument("--num_paraphrases", "-p", type=int, default=10,
                        help="Number of paraphrases to generate for each question.")
    args = parser.parse_args()

    # Load concepts from JSON file
    concepts_by_category = load_concepts_from_json(args.concepts_file)

    # Validate that the concepts were loaded successfully
    if not concepts_by_category:
        print("Error: No concepts were loaded. Please check the JSON file and try again.")
        return

    # Run the experiment
    run_experiment(concepts_by_category, args.model_name, args.output_dir,
                   args.num_questions, args.num_options, args.num_paraphrases)


if __name__ == "__main__":
    main()
