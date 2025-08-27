import jsonlines
import sys
import torch
import re
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
        device_map="auto" # Automatically use GPU if available
    )


    results = []
    for entry in dataset:
        # TODO: create prompt for the model
        # Tip : Use can use any data from the dataset to create 
        #       the prompt including prompt, canonical_solution, test, etc.
        # Step 1: Extract a single test case (input and expected output).
        # We use the first assertion found in the test string for consistency.
        test_code = entry['test']
        # This regex captures the arguments passed to `candidate` and the expected result.
        # It handles various formatting by stopping at a comma, end of line, or comment.
        match = re.search(r"assert\s+candidate\((.*?)\)\s*==\s*(.*?)(?:,|$|#)", test_code, re.DOTALL)

        if not match:
            print(f"Warning: Could not parse a test case for {entry['task_id']}. Skipping.")
            continue
        
        input_args_str = match.group(1).strip()
        expected_output_str = match.group(2).strip()

        # Step 2: Combine the docstring/prompt and solution to form the full program.
        full_program_code = entry['prompt'] + entry['canonical_solution']

        # Step 3: Construct the prompt using a template.
        base_template = (
            "You are an AI programming assistant, utilizing the DeepSeek Coder model, "
            "developed by DeepSeek Company, and you only answer questions related to computer science. "
            "For politically sensitive questions, security and privacy issues, and other "
            "non-computer science questions, you will refuse to answer.\n"
            "### Instruction:\n"
            "If the function is called with the input `{input_args}`, what will the following code return?\n"
            "The return value prediction must be enclosed between [Output] and [/Output] tags. "
            "For example : [Output]prediction[/Output].\n"
            "{extra_instructions}"
            "\n{program}\n\n"
            "### Response:"
        )

        extra_instructions = ""
        if not vanilla: # This is for the "crafted" prompt
            extra_instructions = "Reason step by step to solve the problem."

        prompt = base_template.format(
            input_args=input_args_str,
            extra_instructions=extra_instructions,
            program=full_program_code
        )
        
        # TODO: prompt the model and get the response
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

        response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
        

        # TODO: process the response and save it to results
        prediction_match = re.search(r'\[Output\](.*?)\[/Output\]', response, re.DOTALL | re.IGNORECASE)
        
        verdict = False
        if prediction_match:
            predicted_output = prediction_match.group(1).strip()
            if predicted_output == expected_output_str:
                verdict = True
        else:
            predicted_output = "N/A - Output tags not found in response."

        print(f"Task_ID {entry['task_id']}:\nprompt:\n{prompt}\nresponse:\n{response}\nis_correct:\n{verdict}")
        results.append({
            "task_id": entry["task_id"],
            "prompt": prompt,
            "response": response,
            "is_correct": verdict
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
    `python3 Task_1.py <input_dataset> <model> <output_file> <if_vanilla>`|& tee prompt.log

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
