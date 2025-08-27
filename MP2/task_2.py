import jsonlines
import sys
import torch
import re
import os
import subprocess
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

#####################################################
# Please finish all TODOs in this file for MP2;
#####################################################

def save_file(content, file_path):
    with open(file_path, 'w') as file:
        file.write(content)

def prompt_model(dataset, model_name = "deepseek-ai/deepseek-coder-6.7b-instruct", vanilla = True):
    print(f"Working with {model_name} prompt type {vanilla}...")
    
    # TODO: download the model
    # TODO: load the model with quantization
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        quantization_config=quantization_config,
        device_map="auto" 
    )
    
    results = []
    coverage_dir = "MP2/Coverage"
    for entry in dataset:
        # TODO: create prompt for the model
        # Tip : Use can use any data from the dataset to create 
        #       the prompt including prompt, canonical_solution, test, etc.
        task_id_sanitized = entry['task_id'].replace('/', '_')
        entry_point = entry['entry_point']
        full_program_code = entry['prompt'] + entry['canonical_solution']
        
        base_instruction = (
            "You are an AI programming assistant, utilizing the DeepSeek Coder model, "
            "developed by DeepSeek Company, and you only answer questions related to computer science. "
            "For politically sensitive questions, security and privacy issues, and other "
            "non-computer science questions, you will refuse to answer.\n"
            "### Instruction:\n"
            "{instruction}\n\n"
            "{program}\n\n"
            "### Response:"
        )

        if vanilla:
            instruction = (
                "Generate a pytest test suite for the following code.\n"
                "Only write unit tests in the output and nothing else."
            )
        else: # Crafted prompt for higher coverage
            instruction = (
                "Generate a comprehensive pytest test suite for the following Python code. "
                "Your goal is to achieve the highest possible test coverage. "
                "Ensure you cover all edge cases, such as empty inputs, null values, "
                "incorrect data types, and inputs that test different conditional branches (if/else statements) and loops. "
                "Do not write any explanatory text, only the Python test code."
            )

        prompt = base_instruction.format(instruction=instruction, program=full_program_code)
        
        # TODO: prompt the model and get the response
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
        response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)

        # TODO: process the response, generate coverage and save it to results
        # Save the original code to a temporary file.
        code_to_test_path = "temp_code.py"
        save_file(full_program_code, code_to_test_path)
        
        # Clean the LLM's response and save it as a test file.
        match = re.search(r"```python\n(.*?)```", response, re.DOTALL)
        if match:
            test_code = match.group(1).strip()
        # Add the necessary import statement to the generated test code.
        full_test_code = f"from {os.path.splitext(code_to_test_path)[0]} import {entry_point}\n\n{test_code}"
        test_file_path = "temp_test.py"
        save_file(full_test_code, test_file_path)

        # Run pytest with coverage.
        prompt_type = "vanilla" if vanilla else "crafted"
        report_path = os.path.join(coverage_dir, f"{task_id_sanitized}_{prompt_type}.json")
        
        command = [
            sys.executable, "-m", "pytest", test_file_path,
            f"--cov={os.path.splitext(code_to_test_path)[0]}", # Specify module to cover
            f"--cov-report=json:{report_path}"
        ]
        
        try:
            subprocess.run(command, check=True, capture_output=True, text=True, timeout=60)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            print(f"Pytest execution failed for {task_id_sanitized}: {e.stderr}")

        # Read the coverage from the generated JSON report.
        coverage = 0.0
        if os.path.exists(report_path):
            with open(report_path, 'r') as f:
                try:
                    coverage_data = json.load(f)
                    # Use .get() for safe access to prevent KeyErrors
                    coverage = coverage_data.get('totals', {}).get('percent_covered', 0.0)
                except json.JSONDecodeError:
                    print(f"Warning: Could not decode JSON from {report_path}")

        print(f"Task_ID {entry['task_id']}:\nprompt:\n{prompt}\nresponse:\n{response}\ncoverage:\n{coverage}")
        results.append({
            "task_id": entry["task_id"],
            "prompt": prompt,
            "response": response,
            "coverage": coverage
        })
        
    return results

def read_jsonl(file_path):
    dataset = []
    with jsonlines.open(file_path) as reader:
        for line in reader: 
            dataset.append(line)
    return dataset

def write_jsonl(results, file_path):
    with jsonlines.open(file_path, "w") as f:
        for item in results:
            f.write_all([item])

if __name__ == "__main__":
    """
    This Python script is to run prompt LLMs for code synthesis.
    Usage:
    `python3 task_2.py <input_dataset> <model> <output_file> <if_vanilla>`|& tee prompt.log

    Inputs:
    - <input_dataset>: A `.jsonl` file, which should be your team's dataset containing 20 HumanEval problems.
    - <model>: Specify the model to use. Options are "deepseek-ai/deepseek-coder-6.7b-base" or "deepseek-ai/deepseek-coder-6.7b-instruct".
    - <output_file>: A `.jsonl` file where the results will be saved.
    - <if_vanilla>: Set to 'True' or 'False' to enable vanilla prompt
    
    Outputs:
    - You can check <output_file> for detailed information.
    """
    args = sys.argv[1:]
    input_dataset = args[0]
    model = args[1]
    output_file = args[2]
    if_vanilla = args[3] # True or False
    
    if not input_dataset.endswith(".jsonl"):
        raise ValueError(f"{input_dataset} should be a `.jsonl` file!")
    
    if not output_file.endswith(".jsonl"):
        raise ValueError(f"{output_file} should be a `.jsonl` file!")
    
    vanilla = True if if_vanilla == "True" else False
    
    dataset = read_jsonl(input_dataset)
    results = prompt_model(dataset, model, vanilla)
    write_jsonl(results, output_file)
