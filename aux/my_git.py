"""
added cause "source control" vscode extension is giving me problems in the remote machine. And this is faster than running all the git commands myself in the terminal.
"""
import subprocess
import os
import sys
import re

# Specify the path to your SSH key
SSH_KEY_PATH = os.path.expanduser('~/.ssh/github_repo_key')

def run_command(command, use_ssh=False):
    try:
        env = os.environ.copy()
        if use_ssh:
            env['GIT_SSH_COMMAND'] = f'ssh -i {SSH_KEY_PATH}'
        
        result = subprocess.run(command, check=True, text=True, capture_output=True, env=env)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        print(e.stderr)
        return False

def initialize_ssh_agent():
    try:
        # Start the ssh-agent
        ssh_agent_output = subprocess.check_output("ssh-agent -s", shell=True, text=True)
        
        # Extract SSH_AUTH_SOCK and SSH_AGENT_PID using regex
        auth_sock_match = re.search(r'SSH_AUTH_SOCK=([^;]+)', ssh_agent_output)
        agent_pid_match = re.search(r'SSH_AGENT_PID=(\d+)', ssh_agent_output)
        
        if auth_sock_match:
            os.environ['SSH_AUTH_SOCK'] = auth_sock_match.group(1)
        if agent_pid_match:
            os.environ['SSH_AGENT_PID'] = agent_pid_match.group(1)
        
        # Add the specific SSH key
        subprocess.run(["ssh-add", SSH_KEY_PATH], check=True)
        
        print("SSH agent initialized and key added.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error initializing SSH agent: {e}")
        return False

def git_pull():
    print("Pulling latest changes...")
    return run_command(["git", "pull"], use_ssh=True)

def git_status():
    print("Current status:")
    return run_command(["git", "status"])

def git_add():
    print("Adding all changes...")
    return run_command(["git", "add", "."])

def git_commit(message):
    full_message = f"{message} from a100"
    print(f"Committing changes with message: {full_message}")
    return run_command(["git", "commit", "-m", full_message])

def git_push():
    print("Pushing changes...")
    return run_command(["git", "push"], use_ssh=True)

def commit_and_push():
    git_status()

    if not git_add():
        print("Failed to add changes. Aborting.")
        return

    commit_message = input("Enter your commit message: ")
    if not commit_message:
        print("Commit message cannot be empty. Aborting.")
        return

    if not git_commit(commit_message):
        print("Failed to commit changes. Aborting.")
        return

    if not git_push():
        print("Failed to push changes.")
        return

    print("Commit and push completed successfully.")

def main():
    if not os.path.isfile(SSH_KEY_PATH):
        print(f"SSH key not found at {SSH_KEY_PATH}. Please check the path.")
        return

    if not initialize_ssh_agent():
        print("Failed to initialize SSH agent. Git operations might fail.")
    
    while True:
        choice = input("What would you like to do? (commit/pull/quit): ").lower()

        if choice == 'commit':
            commit_and_push()
        elif choice == 'pull':
            if git_pull():
                print("Pull completed successfully.")
            else:
                print("Failed to pull latest changes.")
        elif choice == 'quit':
            print("Exiting the script.")
            break
        else:
            print("Invalid choice. Please enter 'commit', 'pull', or 'quit'.")

if __name__ == "__main__":
    main()