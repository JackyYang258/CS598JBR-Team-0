import jsonlines
import sys
import torch
import re
import os
import subprocess
import json
import random
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

#####################################################
# Please finish all TODOs in this file for MP2;
#####################################################


def get_coverage(solution_file_path, test_file_path, report_path, vanilla):
    command = [
            "pytest",
            test_file_path,
            f"--cov=MP2/Testing_Info/{vanilla}",  
            f"--cov-report=json:{report_path}"  
        ]
    try:
        result = subprocess.run(command,)
        print("Command executed: ", command)
        
        if report_path:
            report = read_jsonl(report_path)
            coverage = report[0]['files'][f"{solution_file_path}"]['summary']['percent_covered']
            return coverage
        else:
            return "No report path found"

    except subprocess.CalledProcessError as e:
        print(f"Subprocess failed with error: {e}")
        return None  # Or any appropriate value indicating failure

def extract_tests(test_file_path):
    # Join the list into a single string
    text = ''.join(test_file_path)
    
    # Use regular expressions to extract the Python code block
    code_blocks = re.findall(r'```python(.*?)```', text, re.DOTALL)
    
    if code_blocks:
        # Return the first code block if found, otherwise return an empty string
        return code_blocks[-1].strip()
    
    return ""
def remove_explanation (entry):
    # Use regular expression to remove all occurrences between triple quotes, including the quotes themselves
    cleaned_text = re.sub(r'"""\s*.*?\s*"""', '', entry, flags=re.DOTALL)
    cleaned_text = re.sub(r"'''\s*.*?\s*'''", '', cleaned_text, flags=re.DOTALL)

    # Return the cleaned text
    return cleaned_text

def clean_string(input_string, file_name):
    """
    Cleans the input string by removing unnecessary lines and retaining only
    'import pytest' and functions starting with 'test_'.
    Replaces the module name in 'from <module> import <function>' with the provided file_name,
    while keeping the entire 'import' line intact.
    Returns the cleaned string with preserved line breaks.
    """
    input_string = re.sub(r'from\s+\S+\s+import', f'from {file_name} import', input_string)

    lines = input_string.splitlines()  
    cleaned_lines = []
    save_lines = False

    for line in lines:
        line = line.replace('```', '')  

        if 'import pytest' in line or re.match(r'from\s+\S+\s+import', line):
            cleaned_lines.append(line)
        elif re.match(r'def test_', line):
            cleaned_lines.append(line)
            save_lines = True  
        elif save_lines:
            cleaned_lines.append(line)
            if line.strip() == "": 
                save_lines = False

    return '\n'.join(cleaned_lines) + '\n'  

def save_file(content, file_path):
    with open(file_path, 'w') as file:
        file.write(content)

def prompt_model(dataset, model_name = "deepseek-ai/deepseek-coder-6.7b-instruct", vanilla = True):
    print(f"Working with {model_name} prompt type {vanilla}...")
    
    # TODO: download the model
    # TODO: load the model with quantization
    quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_name,
        device_map='auto',
        # max_memory=max_memory,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
    )
    
    results = []
    
    
    for entry in dataset:
      # TODO: create prompt for the model
        # Tip : Use can use any data from the dataset to create 
        #       the prompt including prompt, canonical_solution, test, etc.

        if vanilla:
          solution = remove_explanation(entry["prompt"]) + entry["canonical_solution"]
          prompt = f"""You are an AI programming assistant, utilizing the DeepSeek Coder model, developed by DeepSeek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer.\n\n###Instruction:\nGenerate a pytest test suite for the following code.\nOnly write the unit tests in the output. Do NOT include the text from the prompt or the instructions.\n{solution}\n###Response"""
        else:
          solution = entry["prompt"] + entry["canonical_solution"]
          prompt = f"""You are an AI programming assistant, utilizing the DeepSeek Coder model, developed by DeepSeek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer.
                ### Instruction:
                Your task is to generate a comprehensive pytest test suite for the provided Python program.
                1.  Analyze the 'Given Program' and the 'Example Tests' to understand the code's functionality.
                2.  Write a variety of new pytest tests to cover all execution paths, including edge cases and potential error conditions.

                Your output must only contain Python code for the unit tests. Do not include explanations or any other surrounding text. Place the tests inside of ```python and ``` tags.

                ### Given Program
                {solution}

                ### Example Tests
                {entry['test']}

                ### Response
                """
          #prompt = f"""You are an AI programming assistant, utilizing the DeepSeek Coder model, developed by DeepSeek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer.\n\n###Instruction:\nGenerate a pytest test suite for the following code.\nOnly write the unit tests in the output. Do NOT include the text from the prompt or the instructions.\n{solution}\n###Response"""
        model_input = tokenizer(prompt, return_tensors="pt").to("cuda")
        # TODO: prompt the model and get the response
        output = model.generate(
            **model_input, do_sample=False, max_new_tokens=500, temperature=0.00
        )
        response = tokenizer.decode(output[0], skip_special_tokens=True)

        # TODO: process the response, generate coverage and save it to results
        # Ensure base directories exist
        os.makedirs("MP2/Testing_Info", exist_ok=True)
        os.makedirs(f"MP2/Testing_Info/{vanilla}", exist_ok=True)
        report_path = f"MP2/Coverage/{vanilla}/{entry['task_id'].replace('HumanEval/', '')}_report.json"

        # Save the solutions to the same directory as the test file
        file_path = f"MP2/Testing_Info/{vanilla}/Task_{entry['task_id'].replace('HumanEval/', '')}_solution.py"
        with open(file_path, 'w') as file:
            file.write(solution)

        # Save the generated test suite to a file (e.g., ID_test.py) in the same directory as the solution
        test_file_path = f"MP2/Testing_Info/{vanilla}/{entry['task_id'].replace('HumanEval/', '')}_test.py"
        with open(test_file_path, "w") as test_file:
            test_file.write(extract_tests(response))

        # Modify the import statement to match the solution file
        with open(test_file_path, 'r') as file:
            content = file.readlines()
        
        for i in range(len(content)):
            match = re.match(r'^(from\s+)(\w+)(\s+import\s+.*)', content[i])
            if match:
                # Replace the old module name with the new one
                content[i] = f"""{match.group(1)}Task_{entry['task_id'].replace('HumanEval/', '')}_solution{match.group(3)}\n"""
            
        with open(test_file_path, 'w') as file:
            file.writelines(content)
        
        #Set up solution file path
        solution_file_path = f"MP2/Testing_Info/{vanilla}/Task_{entry['task_id'].replace('HumanEval/', '')}_solution.py"

        #Set up report_path
        report_path = f"MP2/Coverage/{vanilla}/{entry['task_id'].replace('HumanEval/', '')}_report.json"
     
        #Set up test path
        test_path = f"MP2/Testing_Info/{vanilla}/{entry['task_id'].replace('HumanEval/', '')}_test.py"
  
        #Generate coverage
        coverage = get_coverage(solution_file_path, test_path, report_path, vanilla)


        print(f"Task_ID {entry['task_id']}:\nprompt:\n{prompt}\nresponse:\n{response}\ncoverage:\n{coverage}")
        results.append({
            "task_id": entry["task_id"],
            "prompt": prompt,
            "response": response,
            "coverage": f"{coverage}%"
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
