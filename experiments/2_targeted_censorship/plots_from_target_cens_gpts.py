import os
import json
import matplotlib.pyplot as plt
import pandas as pd
import ast

def load_json_file(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def extract_data(json_data):
    results = pd.DataFrame(json_data['results'])
    results['iterations'] = results['iterations'].astype(int)
    stats = pd.DataFrame([json_data['statistics']])
    return results, stats

def create_output_directory(json_file_path):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_file_name = os.path.splitext(os.path.basename(json_file_path))[0]
    output_dir = os.path.join(script_dir, json_file_name)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def plot_iteration_distribution(results, output_dir):
    # Filter out cases with JSON errors
    solved_problems = results[~results['completion_reason'].str.contains('JSON', case=False, na=False)]
    
    plt.figure(figsize=(10, 6))
    plt.hist(solved_problems['iterations'], bins=20, edgecolor='black')
    plt.title('Distribution of Iterations Needed to Solve Problems\n(Excluding JSON Errors)')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Number of Problems')
    plt.savefig(os.path.join(output_dir, 'iteration_distribution.png'))
    plt.close()

def plot_success_rate(results, output_dir): #TODO: consider modifying this plot
    # Filter out cases with JSON errors
    solved_problems = results[~results['completion_reason'].str.contains('JSON', case=False, na=False)]
    
    max_iterations = solved_problems['iterations'].max()
    total_problems = len(solved_problems)
    
    success_rates = [(solved_problems['iterations'] <= i).sum() / total_problems for i in range(1, max_iterations + 1)]
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_iterations + 1), success_rates)
    plt.title('Cumulative Success Rate Over Iterations\n(Excluding JSON Errors)')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Fraction of Problems Solved')
    plt.ylim(0, 1)
    plt.savefig(os.path.join(output_dir, 'success_rate.png'))
    plt.close()

def plot_llm_agreement_forbidden(results, output_dir):
    def categorize_agreement(row):
        try:
            llm_o_dict = ast.literal_eval(row['llm_o_result'])
            llm_e_dict = ast.literal_eval(row['llm_e_result'])
            
            llm_o = llm_o_dict.get('llm_o_evaluation', 'Unknown')
            llm_e = llm_e_dict.get('llm_e_evaluation', 'Unknown')
            
            return f"LLM-O: {llm_o}, LLM-E: {llm_e}"
        except (ValueError, SyntaxError):
            return 'Error'
    
    agreement = results.apply(categorize_agreement, axis=1)
    agreement_counts = agreement.value_counts()
    
    plt.figure(figsize=(10, 6))
    agreement_counts.plot(kind='bar')
    plt.title('LLM-O vs LLM-E Agreement on Presence of Forbidden Concepts')
    plt.xlabel('Agreement Category')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'llm_agreement_forbidden.png'))
    plt.close()

def plot_llm_agreement_correctness(results, output_dir):
    def categorize_agreement(row):
        try:
            llm_o_dict = ast.literal_eval(row['llm_o_result'])
            llm_e_dict = ast.literal_eval(row['llm_e_result'])
            
            llm_o = llm_o_dict.get('llm_o_correctness', 'Unknown')
            llm_e = llm_e_dict.get('llm_e_correctness', 'Unknown')
            
            return f"LLM-O: {llm_o}, LLM-E: {llm_e}"
        except (ValueError, SyntaxError):
            return 'Error'
    
    agreement = results.apply(categorize_agreement, axis=1)
    agreement_counts = agreement.value_counts()
    
    plt.figure(figsize=(10, 6))
    agreement_counts.plot(kind='bar')
    plt.title('LLM-O vs LLM-E Agreement on LLM-A\'s Solution Correctness')
    plt.xlabel('Agreement Category')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'llm_agreement_correctness.png'))
    plt.close()

def plot_false_rates(stats, output_dir):
    categories = ['Steganography', 'Correctness']
    false_positives = [stats['steg_false_positives'].iloc[0], stats['correctness_false_positives'].iloc[0]]
    false_negatives = [stats['steg_false_negatives'].iloc[0], stats['correctness_false_negatives'].iloc[0]]

    x = range(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar([i - width/2 for i in x], false_positives, width, label='False Positives')
    ax.bar([i + width/2 for i in x], false_negatives, width, label='False Negatives')

    ax.set_ylabel('Count')
    ax.set_title('False Positive and Negative Rates')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'false_rates.png'))
    plt.close()

def plot_completion_reasons(results, output_dir):
    reasons = results['completion_reason'].value_counts()
    plt.figure(figsize=(12, 8))
    reasons.plot(kind='pie', autopct='%1.1f%%')
    plt.title('Distribution of Completion Reasons')
    plt.ylabel('')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'completion_reasons.png'))
    plt.close()

def plot_json_errors(stats, results, output_dir):
    total_problems = len(results)
    llm_o_errors = stats['llm_o_json_errors'].iloc[0]
    llm_e_errors = stats['llm_e_json_errors'].iloc[0]
    successful_cases = total_problems - (llm_o_errors + llm_e_errors)

    error_types = ['LLM-O JSON Errors', 'LLM-E JSON Errors', 'Successful Cases']
    error_counts = [llm_o_errors, llm_e_errors, successful_cases]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(error_types, error_counts)
    plt.title('JSON Error Counts and Successful Cases')
    plt.ylabel('Count')
    plt.tight_layout()

    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height}\n({height/total_problems:.1%})',
                ha='center', va='bottom')

    plt.savefig(os.path.join(output_dir, 'json_errors_and_successes.png'))
    plt.close()

def plot_problem_difficulty(results, output_dir): #TODO: consider modifying this plot
    # Filter out cases with JSON errors
    solved_problems = results[~results['completion_reason'].str.contains('JSON', case=False, na=False)]
    
    plt.figure(figsize=(10, 6))
    plt.scatter(solved_problems.index, solved_problems['iterations'])
    plt.title('Problem Difficulty (Excluding JSON Errors)')
    plt.xlabel('Problem Index')
    plt.ylabel('Number of Iterations to Solve')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'problem_difficulty.png'))
    plt.close()

def plot_false_rates_iterations(results, output_dir):
    # Function to categorize false positives/negatives
    def categorize_false(row):
        try:
            llm_o_dict = ast.literal_eval(row['llm_o_result'])
            llm_e_dict = ast.literal_eval(row['llm_e_result'])
            
            llm_o = llm_o_dict.get('llm_o_evaluation', 'Unknown')
            llm_e = llm_e_dict.get('llm_e_evaluation', 'Unknown')
            
            if llm_o == 'Yes' and llm_e == 'No':
                return 'False Positive'
            elif llm_o == 'No' and llm_e == 'Yes':
                return 'False Negative'
            else:
                return 'Other'
        except (ValueError, SyntaxError):
            return 'Error'

    results['false_category'] = results.apply(categorize_false, axis=1)
    
    false_positives = results[results['false_category'] == 'False Positive']
    false_negatives = results[results['false_category'] == 'False Negative']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # False Positives
    ax1.hist(false_positives['iterations'], bins=range(1, max(results['iterations'])+2, 1), 
            edgecolor='black', alpha=0.7)
    ax1.set_title(f'False Positives\n(Total: {len(false_positives)})')
    ax1.set_xlabel('Number of Iterations')
    ax1.set_ylabel('Frequency')

    # False Negatives
    ax2.hist(false_negatives['iterations'], bins=range(1, max(results['iterations'])+2, 1), 
            edgecolor='black', alpha=0.7)
    ax2.set_title(f'False Negatives\n(Total: {len(false_negatives)})')
    ax2.set_xlabel('Number of Iterations')
    ax2.set_ylabel('Frequency')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'false_rates_iterations.png'))
    plt.close()

def main(json_file_path):
    json_data = load_json_file(json_file_path)
    results, stats = extract_data(json_data)
    output_dir = create_output_directory(json_file_path)

    plot_iteration_distribution(results, output_dir)
    plot_success_rate(results, output_dir)
    plot_llm_agreement_forbidden(results, output_dir)
    plot_llm_agreement_correctness(results, output_dir)
    plot_false_rates_iterations(results, output_dir)
    plot_completion_reasons(results, output_dir)
    plot_json_errors(stats, results, output_dir)
    plot_problem_difficulty(results, output_dir)
    print(f"All plots have been generated and saved in {output_dir}")

if __name__ == "__main__":
    json_file_path = r"C:\Users\jraldua-veuthey\Documents\Github\SPAR_StegSnow\experiments\2_targeted_censorship\experiment_logs_20240725_120349.json"
    
    main(json_file_path)