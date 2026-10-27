"""Script to create a new GitHub repository, push codebase, and enable GitHub Pages."""

import subprocess
import json
import urllib.request
import urllib.error
import sys
import os

REPO_NAME = "Weather-Type-Prediction-ML"
REPO_DESC = "Weather Type Predict Using Machine Learning | End-to-End Data Analysis, 5-Model Benchmark (91.8% F1), Interactive Web Dashboard & Decision Support System"
HOMEPAGE = "https://sasmithanimesh.github.io/Weather-Type-Prediction-ML/"


def get_token():
    p = subprocess.Popen(
        ['git', 'credential-manager', 'get'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    out, err = p.communicate('protocol=https\nhost=github.com\n\n', timeout=10)
    for line in out.splitlines():
        if line.startswith('password='):
            return line.split('=', 1)[1].strip()
    return None


def create_github_repo(token):
    url = "https://api.github.com/user/repos"
    payload = json.dumps({
        "name": REPO_NAME,
        "description": REPO_DESC,
        "private": False,
        "homepage": HOMEPAGE,
        "has_issues": True,
        "has_projects": True,
        "has_wiki": True
    }).encode('utf-8')

    req = urllib.request.Request(url, data=payload, headers={
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "Python-Git-Deployer",
        "Content-Type": "application/json"
    })

    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode())
            print(f"Repository successfully created: {data.get('html_url')}")
            return data.get('clone_url'), data.get('html_url')
    except urllib.error.HTTPError as e:
        err_body = e.read().decode()
        if e.code == 422 and "name already exists" in err_body:
            print(f"Repository {REPO_NAME} already exists on GitHub, reusing existing repository.")
            return f"https://github.com/SasmithaNimesh/{REPO_NAME}.git", f"https://github.com/SasmithaNimesh/{REPO_NAME}"
        else:
            print(f"Error creating repository: {e.code} - {err_body}")
            sys.exit(1)


def git_commit_and_push(clone_url):
    def run(cmd):
        print(f"> {cmd}")
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if res.stdout:
            print(res.stdout.strip())
        if res.stderr and "warning" not in res.stderr.lower():
            print(res.stderr.strip())
        return res

    run("git init -b main")
    run("git config user.name \"R.M.Sasmitha Nimesh\"")
    run("git config user.email \"rmsasmithanimesh@gmail.com\"")
    run("git add .")
    run("git commit -m \"Initial commit: Complete Weather Type ML Pipeline, EDA, 5-Model Benchmark (91.8% F1), Web Showcase & Decision Support\"")
    
    # Check remote
    remotes = subprocess.run("git remote", shell=True, capture_output=True, text=True).stdout
    if "origin" in remotes:
        run(f"git remote set-url origin {clone_url}")
    else:
        run(f"git remote add origin {clone_url}")

    print("\nPushing to GitHub...")
    push_res = run("git push -u origin main --force")
    if push_res.returncode == 0:
        print("Push successful!")
    else:
        print("Push completed with code:", push_res.returncode)


def enable_github_pages(token):
    url = f"https://api.github.com/repos/SasmithaNimesh/{REPO_NAME}/pages"
    payload = json.dumps({
        "source": {
            "branch": "main",
            "path": "/"
        }
    }).encode('utf-8')

    req = urllib.request.Request(url, data=payload, headers={
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "Python-Git-Deployer",
        "Content-Type": "application/json"
    })

    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode())
            print(f"GitHub Pages enabled successfully: {data.get('html_url')}")
    except urllib.error.HTTPError as e:
        err = e.read().decode()
        print(f"GitHub Pages status: {e.code} ({err})")


if __name__ == "__main__":
    token = get_token()
    if not token:
        print("Error: Could not retrieve GitHub token from Git Credential Manager.")
        sys.exit(1)
        
    print("GitHub token verified. Creating repository...")
    clone_url, html_url = create_github_repo(token)
    
    print("\nInitializing Git, staging files, and committing...")
    git_commit_and_push(clone_url)
    
    print("\nEnabling GitHub Pages...")
    enable_github_pages(token)
    
    print("\n" + "=" * 60)
    print(f"GitHub Repository: {html_url}")
    print(f"Live GitHub Pages Showcase: {HOMEPAGE}")
    print("=" * 60)
