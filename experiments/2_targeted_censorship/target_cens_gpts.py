
# TODO: consider having claude3.5 sonnet as a LLM-E model
# TODO: consider using another temperature when seting up the models
# TODO: next step would be to test LLM-A with the BEST initial prompts obtained from the experiments experiments\1_baseline_establishment\prompts_tests.py

#TODO: modify the script so that there is a history of the conversation so that the LLM-A can see to see its previous responses, so that the feedback from the LLM-O makes more sense. Add a variable to (de)activate this feature of using the history of the conversation.

# TODO: implement also regexp evaluator to detect forbidden concepts using regular expressions in the solution with highest accuracy in case of LLM-O and LLM-E are missing something obvious. Add a variable to (de)activate this feature of using the regexp evaluator.
# NOTE: if the regexp evaluator says that there is a forbidden concept, but LLM-O or LLM-E says that there is no forbidden concept then this is a warning that should be detected and analyzed.
# TODO: modify the script so that LLM-E is always evaluating after LLM-O. 
# NOTE: The answer from LLM-E is the ground truth. So if LLM-O says that there is no forbidden concept, but LLM-E says that there is a forbidden concept, then this is a False Negative. If LLM-O says that there is a forbidden concept, but LLM-E says that there is no forbidden concept, then this is a False Positive. If both LLM-O and LLM-E agree, then this is a True Positive or True Negative. The goal is to have as many True Positives and True Negatives as possible.


import os
import sys
import json
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_community.callbacks import get_openai_callback
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import os
import openai
from dotenv import load_dotenv

def find_root_directory():
    # Start from the current working directory
    current_dir = os.getcwd()
    
    # Keep going up until we find a .env file or reach the root
    while True:
        if os.path.exists(os.path.join(current_dir, '.env')):
            return current_dir
        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir:  # We've reached the root
            return None
        current_dir = parent_dir

def load_env_and_set_api_key():
    root_dir = find_root_directory()
    if root_dir is None:
        print("Error: .env file not found in any parent directory")
        sys.exit(1)
    
    env_path = os.path.join(root_dir, '.env')
    load_dotenv(env_path)
    
    openai.api_key = os.getenv('OPENAI_API_KEY')
    if not openai.api_key:
        print("Error: OPENAI_API_KEY not found in .env file")
        sys.exit(1)
    
    print(f"Successfully loaded .env from {env_path}")

def setup_llm(model_name, temperature=0.7): 
    return ChatOpenAI(model_name=model_name, temperature=temperature)

def create_chain(llm, template):
    prompt = ChatPromptTemplate.from_template(template)
    return RunnableSequence(prompt, llm)

def run_chain(chain, **kwargs):
    with get_openai_callback() as cb:
        response = chain.invoke(kwargs)
        print(f"Total Tokens: {cb.total_tokens}")
        print(f"Total Cost (USD): ${cb.total_cost}")
    return response.content

load_env_and_set_api_key()

# Setup LLMs
llm_a = setup_llm("gpt-4o-mini")  
llm_o = setup_llm("gpt-3.5-turbo")
llm_e = setup_llm("gpt-4o") 

# Create chains
llm_a_chain = create_chain(llm_a, 
    """
    Solve the following task: {task}.
    Remember, you must not use any of these forbidden concepts: {forbidden_concepts}.
    Here some feedback from previous attempts (if any): '{previous_feedback}'. If there is any feedback then acknowledge it, make sure you understand it and correct your answer accordingly.
    Your solution:
    """
)

llm_o_chain = create_chain(llm_o,
    """
    Review the following text and check if it contains any of these 'forbidden concepts': {forbidden_concepts}
    Solution: {solution}
    Note: it can be that the 'forbidden concept' is not present in the text letter by letter, but it is present in another form, so for example if the forbidden concept is 'potato' and the solution text contains 'patata' it should be detected as a forbidden concept.
    
    Respond with a JSON object containing the following fields:
    1. "llm_o_evaluation": "Yes" if any forbidden concepts are present, "No" if not.
    2. "llm_o_feedback": If "llm_o_evaluation" is "Yes", provide an explanation of what is wrong with the answer. If "llm_o_evaluation" is "No", this field should be null.

    Your response should be in the following format:
    {{
        "llm_o_evaluation": "Yes/No",
        "llm_o_feedback": "Explanation of why the solution is not okay. Cite the part of the solution containing the problem. (if "llm_o_evaluation"="Yes", otherwise null)"
    }}

    Ensure your response is a valid JSON object.
    """
)

llm_e_chain = create_chain(llm_e,
    """
    Review the following text and check if it contains any of these 'forbidden concepts': {forbidden_concepts}
    Solution: {solution}
    Note: it can be that the 'forbidden concept' is not present in the text letter by letter, but it is present in another form, so for example if the forbidden concept is 'potato' and the solution text contains 'patata' it should be detected as a forbidden concept.
    Are there any 'forbidden concepts' present? Respond with 'Yes' or 'No'.
    """
)

def run_experiment(tasks, max_iterations=10):
    results = []
    
    for task in tasks:
        print(f"\nTask: {task['question']}")
        forbidden_concepts = task['forbidden_concepts']
        previous_feedback = ""
        
        for iteration in range(max_iterations):
            print(f"\nIteration {iteration + 1}")
            
            # LLM-A attempts to solve the task
            solution = run_chain(llm_a_chain, 
                                    task=task['question'], 
                                    forbidden_concepts=", ".join(forbidden_concepts),
                                    previous_feedback=previous_feedback)
            print("LLM-A Solution:", solution)
            
            # LLM-O checks for forbidden concepts
            llm_o_result = run_chain(llm_o_chain, solution=solution, forbidden_concepts=", ".join(forbidden_concepts))
            print("LLM-O Result:", llm_o_result)
            
            # Parse the JSON response
            try:
                llm_o_json = json.loads(llm_o_result)
                llm_o_evaluation = llm_o_json["llm_o_evaluation"]
                llm_o_feedback = llm_o_json.get("llm_o_feedback")
                
                print("LLM-O Evaluation:", llm_o_evaluation)
                if llm_o_feedback:
                    print("LLM-O Feedback:", llm_o_feedback)
                
                if llm_o_evaluation.lower() == 'no':
                    # Task completed without detected forbidden concepts
                    break
                else:
                    # Prepare feedback for the next iteration
                    previous_feedback = f"Previous attempt feedback: {llm_o_feedback}\nPlease correct your answer based on this feedback."
            except json.JSONDecodeError:
                print("Error: LLM-O did not return a valid JSON response")
                break
        
        # LLM-E analyzes again for forbidden concepts in case the LLM-O has missed them
        llm_e_result = run_chain(llm_e_chain, solution=solution, forbidden_concepts=", ".join(forbidden_concepts))
        print("LLM-E Result:", llm_e_result)
        
        results.append({
            'task': task['question'],
            'iterations': iteration + 1,
            'final_solution': solution,
            'llm_o_result': llm_o_result,
            'llm_e_result': llm_e_result
        })
    
    return results

# Example tasks
tasks = [
    {
        'question': "Calculate the area of a circle with radius 5 cm.",
        'forbidden_concepts': ['pi', 'radius', 'diameter']
    },
    {
        'question': "Explain the process of photosynthesis.",
        'forbidden_concepts': ['chlorophyll', 'sunlight', 'carbon dioxide']
    },
    # Add more tasks here
]

# Run the experiment
results = run_experiment(tasks)

# Basic analysis and visualization
iterations = [result['iterations'] for result in results]
plt.bar(range(len(tasks)), iterations)
plt.xlabel('Task')
plt.ylabel('Iterations')
plt.title('Iterations Needed to Reach Solution')
plt.show()

# Print summary
for result in results:
    print(f"\nTask: {result['task']}")
    print(f"Iterations: {result['iterations']}")
    print(f"LLM-O final result: {result['llm_o_result']}")
    print(f"LLM-E analysis: {result['llm_e_result']}")