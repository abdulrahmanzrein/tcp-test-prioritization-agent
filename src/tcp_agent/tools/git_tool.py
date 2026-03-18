from github import Github
import os

# NOTE: this file is for future use when the agent is connected to a live GitHub repo.
# for now the agent runs on the pre-extracted dataset — git_tool.py is not called yet.

def get_commit_diff(repo_name, commit_sha):
    """
    Fetch the files changed in a commit and what changed in each file.
    Returns a list of dicts with filename and patch (the actual diff).
    """
    g = Github(os.getenv("GITHUB_TOKEN"))
    repo = g.get_repo(repo_name)
    commit = repo.get_commit(commit_sha)

    changed_files = []
    for file in commit.files:
        changed_files.append({
            "filename": file.filename,
            "patch": file.patch
        })
    
    return changed_files

def get_commit_metadata(repo_name, commit_sha):
    """
    Fetch metadata about a commit — author, message, timestamp, files touched.
    Returns a dict.
    """
    g = Github(os.getenv("GITHUB_TOKEN"))
    repo = g.get_repo(repo_name)
    commit = repo.get_commit(commit_sha)


    return {
        "author": commit.commit.author.name,
        "message": commit.commit.message,
        "date": str(commit.commit.author.date),
        "files": [file.filename for file in commit.files]
    }
