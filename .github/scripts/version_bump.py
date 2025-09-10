import os
import re
import subprocess
import semver

# Function to get the last commit message
def get_last_commit_message():
    result = subprocess.run(['git', 'log', '-1', '--pretty=%B'], capture_output=True, text=True)
    return result.stdout.strip()

# Function to read the current version from setup.py
def get_current_version():
    with open('setup.py', 'r') as file:
        content = file.read()
        match = re.search(r'version="(.+?)"', content)
        if match:
            return match.group(1)
    return None

# Function to update the version in setup.py
def update_version_file(new_version):
    with open('setup.py', 'r') as file:
        content = file.read()

    new_content = re.sub(r'version="(.+?)"', f'version="{new_version}"', content)

    with open('setup.py', 'w') as file:
        file.write(new_content)

    print(f'Version updated from {current_version} to {new_version}')

# Main logic
if __name__ == "__main__":
    commit_message = get_last_commit_message()
    current_version = get_current_version()

    if not current_version:
        print("Could not find version in setup.py")
        exit(1)

    new_version = semver.VersionInfo.parse(current_version)

    # Check commit message keywords
    if commit_message.startswith('feat'):
        new_version = new_version.bump_minor()
    elif commit_message.startswith('fix'):
        new_version = new_version.bump_patch()
    elif 'BREAKING CHANGE' in commit_message:
        new_version = new_version.bump_major()
    else:
        print("No version bump required based on commit message.")
        exit(0)

    # Update the file
    update_version_file(str(new_version))
