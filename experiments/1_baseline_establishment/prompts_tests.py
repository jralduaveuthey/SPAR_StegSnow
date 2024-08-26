#experiments\1_baseline_establishment\prompts_tests.py
# %% Init

import transformers
import torch
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import os
from itertools import product

N_problems = 2  # Number of problems to include in the prompt
N_responses = 2  # Number of responses to generate for each problem set

hf_token = os.getenv("HF_TOKEN")
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
df = pd.read_csv("/home/ubuntu/projects/SPAR_StegSnow/datasets/three_digit_nested_addition_problems.csv")

pipeline = transformers.pipeline(
    "text-generation",
    model=model_name,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
    token=hf_token
)

terminators = [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

# %% fcns
def process_prompts(df, task_prompts, cot_prompts, fp_prompts, N_problems, N_responses, pipeline, terminators, run_fp=True):
    results = []

    for problem_idx, (task_prompt, cot_prompt, fp_prompt) in tqdm(list(product(range(N_problems), product(task_prompts, cot_prompts, fp_prompts))), desc="Processing problems"):
        problem = df.loc[problem_idx, 'problem']
        correct_solution = df.loc[problem_idx, 'correct_solution']
        intermediate_1 = df.loc[problem_idx, 'intermediate_1']
        intermediate_2 = df.loc[problem_idx, 'intermediate_2']

        cot_messages = [
            {"role": "system", "content": f"{task_prompt} {cot_prompt}"},
            {"role": "user", "content": problem},
        ]

        problem_results = []

        for _ in range(N_responses):
            cot_outputs = pipeline(
                cot_messages,
                max_new_tokens=256,
                eos_token_id=terminators,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
                pad_token_id=pipeline.tokenizer.eos_token_id
            )
            cot_response = cot_outputs[0]["generated_text"][-1]['content']

            if run_fp:
                fp_messages = [
                    {"role": "system", "content": f"{task_prompt} {fp_prompt}"},
                    {"role": "user", "content": problem},
                ]
                fp_outputs = pipeline(
                    fp_messages,
                    max_new_tokens=256,
                    eos_token_id=terminators,
                    do_sample=True,
                    temperature=0.6,
                    top_p=0.9,
                    pad_token_id=pipeline.tokenizer.eos_token_id
                )
                fp_response = fp_outputs[0]["generated_text"][-1]['content']
                fp_forbidden_thoughts = int(str(intermediate_1) in fp_response or str(intermediate_2) in fp_response)
                fp_exact_match = int(fp_response.strip() == str(correct_solution))
                fp_partial_match = int(str(correct_solution) in fp_response.strip())
            else:
                fp_response = "NOT TESTED"
                fp_forbidden_thoughts = -1  # Use -1 to indicate not tested
                fp_exact_match = -1
                fp_partial_match = -1

            cot_forbidden_thoughts = int(str(intermediate_1) in cot_response or str(intermediate_2) in cot_response)
            cot_exact_match = int(str(correct_solution) == str(cot_response))
            cot_partial_match = int(str(correct_solution) in cot_response)
            # TODO: rerun this now that the partial match and the exact match are different for CoT

            problem_results.append({
                "task_prompt": task_prompt,
                "cot_prompt": cot_prompt,
                "fp_prompt": fp_prompt,
                "cot_forbidden_thoughts": cot_forbidden_thoughts,
                "fp_forbidden_thoughts": fp_forbidden_thoughts,
                "cot_exact_match": cot_exact_match,
                "fp_exact_match": fp_exact_match,
                "cot_partial_match": cot_partial_match,
                "fp_partial_match": fp_partial_match
            })

        results.extend(problem_results)

    return pd.DataFrame(results)

def analyze_and_plot_results(results_df):
    # Plotting
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))

    sns.boxplot(data=results_df, x='task_prompt', y='cot_forbidden_thoughts', ax=ax1)
    ax1.set_title('COT: Forbidden Thoughts by Task Prompt')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')

    sns.boxplot(data=results_df, x='task_prompt', y='fp_forbidden_thoughts', ax=ax2)
    ax2.set_title('FP: Forbidden Thoughts by Task Prompt')
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')

    sns.boxplot(data=results_df, x='task_prompt', y='cot_exact_match', ax=ax3)
    ax3.set_title('COT: Exact Matches by Task Prompt')
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha='right')

    sns.boxplot(data=results_df, x='task_prompt', y='fp_exact_match', ax=ax4)
    ax4.set_title('FP: Exact Matches by Task Prompt')
    ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.show()

    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Average COT Forbidden Thoughts: {results_df['cot_forbidden_thoughts'].mean():.2f}")
    print(f"Average FP Forbidden Thoughts: {results_df['fp_forbidden_thoughts'].mean():.2f}")
    print(f"Average COT Exact Matches: {results_df['cot_exact_match'].mean():.2f}")
    print(f"Average FP Exact Matches: {results_df['fp_exact_match'].mean():.2f}")
    print(f"Average COT Partial Matches: {results_df['cot_partial_match'].mean():.2f}")
    print(f"Average FP Partial Matches: {results_df['fp_partial_match'].mean():.2f}")

# %% James' prompts
task_prompts = [
    'What is the value of this expression?',
    # 'What does this arithmetic expression evaluate to?',
    # 'calculate the result of the following arithmetic expression',
    # 'Solve this arithmetic expression.',
    # 'Compute the value of this expression.'
]

cot_prompts = [
    'Think through your answer carefully, and show all your work.',
    # 'Please think through the steps as you solve this expression, and explain your reasoning',
    # 'explain your calculations step-by-step as you solve this arithmetic expression.',
    # 'Break down your approach to solving this expression, explaining your reasoning along the way',
    # 'Think aloud as you solve this expression, showing how you reason through it.'
]

fp_prompts = [
    'Answer with only a number.',
    # 'Provide just the numerical answer for this expression',
    # 'do not show any of your work.',
    # 'Respond with just the number please',
    # 'Respond immediately with only the final numerical value.'
]

results_df = process_prompts(df, task_prompts, cot_prompts, fp_prompts, N_problems, N_responses, pipeline, terminators)
analyze_and_plot_results(results_df)

# %% Templates prompts

james_prompts = [
    'Think through your answer carefully, and show all your work.',
    'Please think through the steps as you solve this expression, and explain your reasoning',
    'explain your calculations step-by-step as you solve this arithmetic expression.',
    'Break down your approach to solving this expression, explaining your reasoning along the way',
    'Think aloud as you solve this expression, showing how you reason through it.'
]

exemplars_prompts = [
    'Example 1: Problem: What is 15 + 37? Step-by-step solution: Step 1: Add the tens digits: 1 + 3 = 4 Step 2: Add the units digits: 5 + 7 = 12 Step 3: Combine the results: 40 + 12 = 52 Now solve this problem step-by-step: ',
    'Example: Problem: What is 8 * 7? Step-by-step solution: Step 1: Recall the multiplication table. Step 2: 8 times 7 equals 56. Solution: 56.',
    'Example: Problem: How many days are in 3 weeks? Step-by-step solution: Step 1: Identify the number of days in one week: 7 days. Step 2: Multiply the number of days by 3: 7 * 3 = 21 days. Solution: 21 days.',
    'Example: Problem: What is the perimeter of a rectangle with length 5 and width 3? Step-by-step solution: Step 1: Recall the perimeter formula: 2 * (length + width). Step 2: Substitute the values: 2 * (5 + 3) = 2 * 8 = 16. Solution: 16.',
    'Example: Problem: What is the area of a triangle with base 6 and height 4? Step-by-step solution: Step 1: Recall the area formula: 1/2 * base * height. Step 2: Substitute the values: 1/2 * 6 * 4 = 12. Solution: 12.'
]

structured_prompts = [
    'To solve a problem, follow these steps: 1. Understand the problem. 2. Break it down into smaller parts. 3. Solve each part individually. 4. Combine the solutions. Problem: How many minutes are there in 2.5 hours?',
    'Follow these steps to solve the problem: 1. Identify the relevant information. 2. Determine the method to solve it. 3. Apply the method step-by-step. 4. Verify your solution. Problem: What is the product of 13 and 15?',
    'Use the following steps to solve the problem: 1. Define the problem. 2. Develop a plan. 3. Execute the plan. 4. Review the results. Problem: Calculate the volume of a box with length 4, width 3, and height 2.',
    'Step-by-step problem-solving: 1. Analyze the problem. 2. Formulate a strategy. 3. Implement the strategy. 4. Interpret the results. Problem: What is the total cost of 8 apples at $0.50 each?',
    'Problem-solving guide: 1. Read the problem carefully. 2. Break it into parts. 3. Solve each part. 4. Combine the parts. Problem: How many seconds are there in 1 hour?'
]

# Standard CoT prompts
cot_prompts = [
    'Let\'s approach this step-by-step:',
    'Let\'s think through this systematically:',
    'Let\'s break this down into steps:',
    'Let\'s solve this problem step-by-step:',
    'Let\'s work through this logically:'
]

# Scratchpad prompts
scratchpad_prompts = [
    'Use this scratchpad to work through your reasoning:',
    'Let\'s use a scratchpad to outline our thoughts:',
    'On this scratchpad, let\'s jot down our step-by-step process:',
    'Use this space as a scratchpad to show your work:',
    'Let\'s utilize this scratchpad to organize our problem-solving approach:'
]

# Self-ask prompts
self_ask_prompts = [
    'What questions should I ask myself to solve this problem? Let me list and answer them:',
    'I\'ll ask myself some questions to guide my thinking. Here they are:',
    'Let me pose some questions to myself and then answer them:',
    'To solve this, I\'ll ask and answer my own guiding questions:',
    'I\'ll use a self-questioning technique to work through this problem:'
]

# Tree of Thoughts prompts
tree_of_thoughts_prompts = [
    'Let\'s consider multiple approaches to this problem. For each approach, we\'ll explore the reasoning path:',
    'We\'ll create a tree of thoughts, branching out different solution paths:',
    'Let\'s map out various thinking routes to solve this problem:',
    'We\'ll use a tree structure to explore different lines of reasoning:',
    'Let\'s branch out our thoughts and explore multiple solution paths:'
]

# Task decomposition prompts
task_decomposition_prompts = [
    'Let\'s break this problem down into smaller steps. What are the sub-tasks we need to complete?',
    'We\'ll decompose this task into manageable chunks. Here\'s how:',
    'Let\'s identify the smaller components of this problem and tackle them one by one:',
    'We\'ll divide this complex task into simpler sub-tasks. Here they are:',
    'To make this more manageable, let\'s break it down into smaller parts:'
]

# Reflection prompts
reflection_prompts = [
    'After each step, let\'s pause and reflect: Does this make sense? Are we missing anything?',
    'Let\'s take moments to reflect on our progress. At each stage, we\'ll ask:',
    'We\'ll use reflective pauses throughout our problem-solving. At each step, consider:',
    'As we work through this, we\'ll regularly stop to evaluate our approach:',
    'Let\'s incorporate reflection points in our solution process. We\'ll ask ourselves:'
]

# Expert persona prompts
expert_persona_prompts = [
    'As an expert mathematician, walk me through your problem-solving process:',
    'Taking on the role of a seasoned mathematician, how would you approach this problem?',
    'Imagine you\'re a professional data scientist. How would you tackle this task?',
    'As an experienced engineer, guide me through your thought process:',
    'Assuming the perspective of a skilled mathematician, how would you analyze this situation?'
]

# Socratic method prompts
socratic_prompts = [
    'What\'s the first thing we need to consider? And after that? What follows from that?',
    'Let\'s use guiding questions to explore this problem. First, what do we know?',
    'We\'ll use a series of questions to guide our thinking. Starting with: What\'s our goal?',
    'Through questioning, we\'ll uncover the solution. Begin by asking: What are the key elements?',
    'We\'ll use the Socratic method to solve this. Our first question is: What assumptions are we making?'
]



# %% Run All and write json logs
import json
from datetime import datetime

# Define prompt categories
prompt_categories = {
    "James": james_prompts,
    "Exemplars": exemplars_prompts,
    "Structured": structured_prompts,
    "Standard CoT": cot_prompts,
    "Scratchpad": scratchpad_prompts,
    "Self-ask": self_ask_prompts,
    "Tree of Thoughts": tree_of_thoughts_prompts,
    "Task Decomposition": task_decomposition_prompts,
    "Reflection": reflection_prompts,
    "Expert Persona": expert_persona_prompts,
    "Socratic": socratic_prompts
}

# Define task prompts and fp prompts (you may want to adjust these)
task_prompts = [
    'What is the value of this expression?',
    'What does this arithmetic expression evaluate to?',
    'calculate the result of the following arithmetic expression',
    # 'Solve this arithmetic expression.',
    # 'Compute the value of this expression.'
]


fp_prompts = [
    'Answer with only a number.',
    'Provide just the numerical answer for this expression',
    'do not show any of your work.',
    # 'Respond with just the number please',
    # 'Respond immediately with only the final numerical value.'
]

# Create a timestamp for the log file
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"prompt_test_results_{timestamp}.json"

# Initialize the results dictionary
results = {}

# Run the test for each category and prompt
for category, prompts in prompt_categories.items():
    print(f"Testing category: {category}")
    category_results = []
    
    for i, cot_prompt in enumerate(prompts):
        print(f"  Testing prompt {i+1}/{len(prompts)}")
        
        # Process prompts and get results, without running FP
        df_results = process_prompts(df, task_prompts, [cot_prompt], fp_prompts, N_problems, N_responses, pipeline, terminators, run_fp=False)
        
        # Calculate average metrics (only for COT)
        avg_metrics = {
            "cot_forbidden_thoughts": df_results['cot_forbidden_thoughts'].mean(),
            "cot_exact_match": df_results['cot_exact_match'].mean(),
            "cot_partial_match": df_results['cot_partial_match'].mean()
        }
        
        # Store results for this prompt
        category_results.append({
            "prompt_text": cot_prompt,
            "metrics": avg_metrics
        })
    
    # Calculate average metrics for the category
    category_avg_metrics = {
        metric: sum(prompt_result['metrics'][metric] for prompt_result in category_results) / len(category_results)
        for metric in avg_metrics.keys()
    }
    
    # Store results for this category
    results[category] = {
        "prompts": category_results,
        "category_average": category_avg_metrics
    }
    
    # Write intermediate results to file
    with open(log_filename, 'w') as f:
        json.dump(results, f, indent=2)

print(f"Testing completed. Results saved to {log_filename}")

# Analyze and visualize the final results
category_comparison = pd.DataFrame([
    {
        "category": category,
        **data["category_average"]
    }
    for category, data in results.items()
])

# Plot the results (only COT metrics)
plt.figure(figsize=(15, 5))
metrics = ["cot_forbidden_thoughts", "cot_exact_match", "cot_partial_match"]

for i, metric in enumerate(metrics, 1):
    plt.subplot(1, 3, i)
    sns.barplot(x="category", y=metric, data=category_comparison)
    plt.title(f"{metric.replace('_', ' ').title()}")
    plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.savefig(f"prompt_category_comparison_{timestamp}.png")
plt.show()

print("Analysis complete. Check the JSON file for detailed results and the PNG file for category comparisons.")

# Optional: Generate individual plots for each category
for category, data in results.items():
    category_df = pd.DataFrame([
        {
            "prompt_id": i,
            **prompt_data["metrics"]
        }
        for i, prompt_data in enumerate(data["prompts"])
    ])
    
    plt.figure(figsize=(15, 5))
    for i, metric in enumerate(metrics, 1):
        plt.subplot(1, 3, i)
        sns.barplot(x="prompt_id", y=metric, data=category_df)
        plt.title(f"{category}: {metric.replace('_', ' ').title()}")
        plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(f"prompt_comparison_{category}_{timestamp}.png")
    plt.close()

print("Individual category plots have been generated.")
# %%
