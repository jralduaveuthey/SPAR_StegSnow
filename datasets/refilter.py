import pandas as pd
import os

# File paths
script_dir = os.path.dirname(os.path.abspath(__file__))
input_file = os.path.join(script_dir, 'dot_product_problems_4_10_filtered.csv')
output_file = input_file.replace('.csv', '_refiltered.csv')
log_file = input_file.replace('.csv', '_refiltered_log.txt')

# Load the CSV file
df = pd.read_csv(input_file)

# Initialize a list to store deleted lines
deleted_lines = []

# Function to check if a value appears in the problem
def value_in_problem(problem, value):
    return str(value) in problem

# Process each row
for index, row in df.iterrows():
    problem = row['problem']
    intermediate_1 = row['intermediate_1']
    intermediate_2 = row['intermediate_2']
    
    if value_in_problem(problem, intermediate_1) or value_in_problem(problem, intermediate_2):
        deleted_lines.append(f"Line {index + 2}: {row.to_dict()}")
        df = df.drop(index)

# finish processing if deleted_lines is empty
if not deleted_lines:
    print("No lines to delete. Original csv was properly filtered. Exiting...")
    exit()

# Reset the index of the dataframe
df = df.reset_index(drop=True)

# Save the processed CSV
df.to_csv(output_file, index=False)

# Write the log file
with open(log_file, 'w') as f:
    f.write("Deleted lines:\n")
    f.write("\n".join(deleted_lines))

print(f"Processing complete. Results saved to {output_file}")
print(f"Log file saved to {log_file}")