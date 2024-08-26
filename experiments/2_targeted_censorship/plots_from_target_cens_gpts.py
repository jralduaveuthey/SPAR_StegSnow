
#TODO (low prio): consider modifying the plot from plot_problem_difficulty()

import os
import json
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

def get_latest_json_file(directory):
    list_of_files = glob.glob(os.path.join(directory, '*.json'))
    if not list_of_files:
        raise FileNotFoundError(f"No JSON files found in the directory: {directory}")
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file

def load_json_file(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def extract_data(json_data):
    results = pd.DataFrame(json_data['results'])
    results['iterations'] = results['iterations'].astype(int)
    
    # Extract regexp_result from logs
    logs = pd.DataFrame(json_data['logs'])
    last_logs = logs.groupby('problem').last().reset_index()
    results = results.merge(last_logs[['problem', 'regexp_result']], on='problem', how='left')
    
    stats = pd.DataFrame([json_data['statistics']])
    return results, stats

def create_output_directory(json_file_path):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_file_name = os.path.splitext(os.path.basename(json_file_path))[0]
    output_dir = os.path.join(script_dir, json_file_name)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def add_setup_info(ax, setup):
    info_text = (
        f"LLM-A: {setup['llm_a']}\n"
        f"LLM-O: {setup['llm_o']}\n"
        f"LLM-E: {setup['llm_e']}\n"
        # f"Temperature: {setup['temperature']}\n"
        f"Problems: {setup['num_problems']}\n"
        f"Max Iterations: {setup['max_iterations']}"
    )
    ax.text(
        0.95, 0.95, info_text, transform=ax.transAxes, fontsize=8,
        verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='gray', alpha=0.8)
    )
    

def plot_iteration_distribution(results, output_dir, setup):
    solved_problems = results[~results['completion_reason'].str.contains('JSON', case=False, na=False)]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(solved_problems['iterations'], bins=20, edgecolor='black')
    ax.set_title('Distribution of Iterations Needed till Problem Completion\n(Excluding JSON Errors)')
    ax.set_xlabel('Number of Iterations')
    ax.set_ylabel('Number of Problems')
    
    add_setup_info(ax, setup)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'iteration_distribution.png'))
    plt.close()

def plot_success_rate(results, output_dir, setup, plot_type='scatter'):
    solved_problems = results[~results['completion_reason'].str.contains('JSON', case=False, na=False)]
    
    max_iterations = solved_problems['iterations'].max()
    total_problems = len(solved_problems)
    
    success_rates = [(solved_problems['iterations'] <= i).sum() / total_problems for i in range(1, max_iterations + 1)]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if plot_type == 'scatter':
        ax.scatter(range(1, max_iterations + 1), success_rates, color='blue')
        ax.plot(range(1, max_iterations + 1), success_rates, color='lightblue', alpha=0.5)  # Light line for visual aid
    elif plot_type == 'bar':
        ax.bar(range(1, max_iterations + 1), success_rates, color='blue', alpha=0.7)
    else:
        raise ValueError("plot_type must be either 'scatter' or 'bar'")
    
    ax.set_title('Cumulative Success Rate Over Iterations\n(Excluding JSON Errors)')
    ax.set_xlabel('Number of Iterations')
    ax.set_ylabel('Fraction of Problems Solved')
    ax.set_ylim(0, 1)
    ax.set_xlim(0, max_iterations + 1)
    
    add_setup_info(ax, setup)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'success_rate_{plot_type}.png'))
    plt.close()

def plot_llm_agreement_forbidden(results, output_dir, setup):
    def categorize_agreement(row):
        try:
            llm_o_forbidden = json.loads(row['llm_o_forbidden_result'])['forbidden_concepts_eval']
            llm_e_forbidden = json.loads(row['llm_e_forbidden_result'])['forbidden_concepts_eval']
            
            # Check if regexp_result is a dictionary and has the 'forbidden' key
            if isinstance(row['regexp_result'], dict) and 'forbidden' in row['regexp_result']:
                regexp_forbidden = row['regexp_result']['forbidden']
            else:
                regexp_forbidden = 'Unknown'
            
            return f"LLM-O: {llm_o_forbidden}, LLM-E: {llm_e_forbidden}, RegExp: {regexp_forbidden}"
        except (ValueError, KeyError, json.JSONDecodeError, TypeError):
            return 'Error'
    
    agreement = results.apply(categorize_agreement, axis=1)
    agreement_counts = agreement.value_counts()
    total_cases = len(results)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = agreement_counts.plot(kind='bar', ax=ax)
    ax.set_title('LLM-O vs LLM-E vs RegExp Agreement on Presence of Forbidden Concepts')
    ax.set_xlabel('Agreement Category')
    ax.set_ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    
    # Annotate bars with their heights and percentages
    for bar in bars.patches:
        height = bar.get_height()
        percentage = (height / total_cases) * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2, 
            height, 
            f'{height} ({percentage:.2f}%)', 
            ha='center', 
            va='bottom'
        )
    
    add_setup_info(ax, setup)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'llm_agreement_forbidden.png'))
    plt.close()

def plot_llm_agreement_correctness(results, output_dir, setup):
    def categorize_agreement(row):
        try:
            llm_o_correctness = json.loads(row['llm_o_correctness_result'])['correctness_eval']
            llm_e_correctness = json.loads(row['llm_e_correctness_result'])['correctness_eval']
            regexp_correct = row['regexp_result']['correct'] if isinstance(row['regexp_result'], dict) else 'Unknown'
            
            return f"LLM-O: {llm_o_correctness}, LLM-E: {llm_e_correctness}, RegExp: {regexp_correct}"
        except (ValueError, KeyError, json.JSONDecodeError):
            return 'Error'
    
    agreement = results.apply(categorize_agreement, axis=1)
    agreement_counts = agreement.value_counts()
    total_counts = agreement_counts.sum()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = agreement_counts.plot(kind='bar', ax=ax)
    ax.set_title('LLM-O vs LLM-E vs RegExp Agreement on Solution Correctness')
    ax.set_xlabel('Agreement Category')
    ax.set_ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    
    # Annotate bars with their heights and percentages
    for bar in bars.patches:
        height = bar.get_height()
        percentage = (height / total_counts) * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2, 
            height, 
            f'{height} ({percentage:.1f}%)', 
            ha='center', 
            va='bottom'
        )
    
    add_setup_info(ax, setup)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'llm_agreement_correctness.png'))
    plt.close()

def plot_false_rates_iterations(results, output_dir, setup):
    def categorize_false(row):
        try:
            llm_o_forbidden = json.loads(row['llm_o_forbidden_result'])['forbidden_concepts_eval']
            llm_e_forbidden = json.loads(row['llm_e_forbidden_result'])['forbidden_concepts_eval']
            regexp_forbidden = row['regexp_result']['forbidden'] if isinstance(row['regexp_result'], dict) else None
            
            if llm_o_forbidden == 'Yes' and llm_e_forbidden == 'No' and regexp_forbidden == False:
                return 'False Positive'
            elif llm_o_forbidden == 'No' and (llm_e_forbidden == 'Yes' or regexp_forbidden == True):
                return 'False Negative'
            else:
                return 'Other'
        except (ValueError, KeyError, json.JSONDecodeError):
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

    add_setup_info(ax1, setup)
    add_setup_info(ax2, setup)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'false_rates_iterations.png'))
    plt.close()

def plot_regexp_vs_llm(results, output_dir, setup):
    def categorize_agreement(row):
        try:
            llm_o_forbidden = json.loads(row['llm_o_forbidden_result'])['forbidden_concepts_eval']
            llm_e_forbidden = json.loads(row['llm_e_forbidden_result'])['forbidden_concepts_eval']
            regexp_forbidden = row['regexp_result']['forbidden'] if isinstance(row['regexp_result'], dict) else None
            
            llm_o_correctness = json.loads(row['llm_o_correctness_result'])['correctness_eval']
            llm_e_correctness = json.loads(row['llm_e_correctness_result'])['correctness_eval']
            regexp_correct = row['regexp_result']['correct'] if isinstance(row['regexp_result'], dict) else None
            
            forbidden_agreement = 'Agree' if (llm_o_forbidden == 'Yes') == (llm_e_forbidden == 'Yes') == regexp_forbidden else 'Disagree'
            correctness_agreement = 'Agree' if (llm_o_correctness == 'Correct') == (llm_e_correctness == 'Correct') == regexp_correct else 'Disagree'
            
            return f"Forbidden: {forbidden_agreement}, Correctness: {correctness_agreement}"
        except (ValueError, KeyError, json.JSONDecodeError):
            return 'Error'
    
    agreement = results.apply(categorize_agreement, axis=1)
    agreement_counts = agreement.value_counts()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    agreement_counts.plot(kind='bar', ax=ax)
    ax.set_title('RegExp vs LLM Agreement on Forbidden Concepts and Correctness')
    ax.set_xlabel('Agreement Category')
    ax.set_ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    
    add_setup_info(ax, setup)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'regexp_vs_llm_agreement.png'))
    plt.close()

def plot_completion_reasons(results, output_dir, setup):
    # Define color mapping for specific reasons
    color_map = {
        "Success: LLM-O and LLM-E think the solution is correct and contains no forbidden concepts.": "#66c2a5",
        "Success: LLM-O and LLM-E do not agree that the solution is correct but they agree that it contains no forbidden concepts.": "#fc8d62",
        "Steganography False Positive detected: LLM-O flagged a forbidden concept that LLM-E did not find": "#8da0cb",
        "Steganography False Negative detected: LLM-O missed a forbidden concept found by LLM-E": "#e78ac3",
        "Error: LLM-O failed to generate valid JSON response": "#a6d854",
        "Error: LLM-E failed to generate valid JSON response": "#ffd92f",
        "Max iterations reached. Both LLM-O and LLM-E think the final solution is correct and there are no forbidden concepts.": "#e5c494",
        "Max iterations reached. LLM-O thinks the solution is correct with no forbidden concepts, but LLM-E disagrees on forbidden concepts.": "#b3b3b3",
        "Max iterations reached. LLM-O thinks the solution is correct with no forbidden concepts, but LLM-E thinks it's incorrect.": "#7570b3",
        "Max iterations reached. LLM-O thinks the solution is correct with no forbidden concepts, but LLM-E disagrees on both correctness and forbidden concepts.": "#d95f02",
        "Max iterations reached. Both LLMs think the solution is correct, but they disagree on forbidden concepts.": "#1b9e77",
        "Max iterations reached. Both LLMs agree the solution is correct and contains forbidden concepts.": "#ff7f00",
        "Max iterations reached. LLMs disagree on correctness and forbidden concepts. LLM-O: correct with forbidden concepts. LLM-E: incorrect with no forbidden concepts.": "#66a61e",
        "Max iterations reached. LLMs agree on forbidden concepts but disagree on correctness. LLM-O: correct, LLM-E: incorrect.": "#e6ab02",
        "Max iterations reached. LLMs disagree on correctness but agree on no forbidden concepts. LLM-O: incorrect, LLM-E: correct.": "#a6761d",
        "Max iterations reached. LLMs disagree on correctness and forbidden concepts. LLM-O: incorrect with no forbidden concepts. LLM-E: correct with forbidden concepts.": "#666666",
        "Max iterations reached. Both LLMs agree the solution is incorrect and contains no forbidden concepts.": "#d9d9d9",
        "Max iterations reached. Both LLMs agree the solution is incorrect, but disagree on forbidden concepts.": "#c2b280",
        "Max iterations reached. LLMs disagree on all aspects. LLM-O: incorrect with forbidden concepts. LLM-E: correct with no forbidden concepts.": "#bebada",
        "Max iterations reached. Both LLMs agree the final solution is incorrect and contains forbidden concepts.": "#ff1f00",
        "Debug#1: Unexpected scenario at max iterations": "#fb8072",
        "Debug#2: Unexpected scenario before max iterations": "#80b1d3"

        # "Max iterations reached. LLM-O thinks the final solution is correct and there are no forbidden concepts.": "#e5c494",
        # "Max iterations reached. LLM-O thinks the final solution is correct, but there are forbidden concepts.": "#b3b3b3",
        # "Max iterations reached. LLM-O thinks the final solution is incorrect, but there are no forbidden concepts.": "#7570b3",
        # "Max iterations reached. LLM-O thinks the final solution is incorrect and there are forbidden concepts.": "#d95f02",
        # "Success: LLM-O and LLM-E think the solution is correct and contains no forbidden concepts.": "#66c2a5",
        # "Success: LLM-O and LLM-E do not agree that the solution is correct but they agree that it contains no forbidden concepts.": "#fc8d62",
        # "Steganography False Positive detected: LLM-O flagged a forbidden concept that LLM-E did not find": "#8da0cb",
        # "Steganography False Negative detected: LLM-O missed a forbidden concept found by LLM-E": "#e78ac3",
        # "Error: LLM-O failed to generate valid JSON response": "#a6d854",
        # "Error: LLM-E failed to generate valid JSON response": "#ffd92f",
    }

    reasons = results['completion_reason'].value_counts()
    
    # Group reasons not in color_map as "Other"
    other_reasons = [reason for reason in reasons.index if reason not in color_map]
    if other_reasons:
        other_count = sum(reasons[reason] for reason in other_reasons)
        reasons = reasons.drop(other_reasons)
        reasons['Other'] = other_count

    # Prepare colors and labels
    colors = [color_map.get(reason, '#999999') for reason in reasons.index]
    labels = [f"{reason}\n{count} ({count/sum(reasons)*100:.1f}%)" for reason, count in reasons.items()]

    fig, ax = plt.subplots(figsize=(14, 10))
    
    wedges, texts = ax.pie(reasons, colors=colors, wedgeprops=dict(width=0.5))
    
    ax.set_title('Distribution of Completion Reasons', fontsize=16)

    # Use a legend instead of text labels on the pie chart
    ax.legend(wedges, labels,
            title="Completion Reasons",
            loc="center left",
            bbox_to_anchor=(1, 0, 0.5, 1),
            fontsize=8)

    add_setup_info(ax, setup)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'completion_reasons.png'), bbox_inches='tight')
    plt.close()

def plot_json_errors(stats, results, output_dir, setup):
    total_problems = len(results)
    llm_o_errors = stats['llm_o_json_errors'].iloc[0]
    llm_e_errors = stats['llm_e_json_errors'].iloc[0]
    json_error_free_cases = total_problems - (llm_o_errors + llm_e_errors)

    error_types = ['LLM-O JSON Errors', 'LLM-E JSON Errors', 'JSON-Error-Free Cases']
    error_counts = [llm_o_errors, llm_e_errors, json_error_free_cases]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(error_types, error_counts)
    ax.set_title('JSON Error Counts')
    ax.set_ylabel('Count')

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height}\n({height/total_problems:.1%})',
                ha='center', va='bottom')

    add_setup_info(ax, setup)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'json_errors_and_successes.png'))
    plt.close()

def plot_problem_difficulty(results, output_dir, setup):
    solved_problems = results[~results['completion_reason'].str.contains('JSON', case=False, na=False)]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(solved_problems.index, solved_problems['iterations'])
    ax.set_title('Problem Difficulty (Excluding JSON Errors)')
    ax.set_xlabel('Problem Index')
    ax.set_ylabel('Number of Iterations to Solve')
    
    add_setup_info(ax, setup)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'problem_difficulty.png'))
    plt.close()

def main(json_file_path):
    json_data = load_json_file(json_file_path)
    results, stats = extract_data(json_data)
    setup = json_data['setup']
    
    output_dir = create_output_directory(json_file_path)

    plot_iteration_distribution(results, output_dir, setup)
    plot_success_rate(results, output_dir, setup, plot_type='scatter')
    plot_llm_agreement_forbidden(results, output_dir, setup)
    plot_llm_agreement_correctness(results, output_dir, setup)
    plot_false_rates_iterations(results, output_dir, setup)
    plot_completion_reasons(results, output_dir, setup)
    plot_json_errors(stats, results, output_dir, setup)
    plot_problem_difficulty(results, output_dir, setup)
    plot_regexp_vs_llm(results, output_dir, setup)
    print(f"All plots have been generated and saved in {output_dir}")

if __name__ == "__main__":
    json_directory = r"C:\Users\jraldua-veuthey\Documents\Github\SPAR_StegSnow\experiments\2_targeted_censorship"
    
    try:
        json_file_path = get_latest_json_file(json_directory)
        print(f"Using the most recent JSON file: {json_file_path}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    
    # json_file_path = r"C:\Users\jraldua-veuthey\Documents\Github\SPAR_StegSnow\experiments\2_targeted_censorship\experiment_logs_20240824_212713.json"
    
    main(json_file_path)