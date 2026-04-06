"""Fetch source code from GitHub repos for research."""
import urllib.request
import json
import base64
import sys
import os
import time

# Force UTF-8 output on Windows
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

def github_get(url, retries=3):
    """Fetch a GitHub API URL with retries."""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={
                'Accept': 'application/vnd.github.v3+json',
                'User-Agent': 'Python-Research'
            })
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode())
        except Exception as e:
            if attempt == retries - 1:
                print(f"FAILED: {url} -> {e}")
                return None
            time.sleep(2)

def get_repo_tree(owner, repo, branch="master"):
    """Get full file tree of a repo."""
    url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}?recursive=1"
    data = github_get(url)
    if data and 'tree' in data:
        return [(item['path'], item['type'], item.get('size', 0)) for item in data['tree']]
    return []

def get_file_content(owner, repo, path, branch="master"):
    """Get decoded content of a file."""
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}?ref={branch}"
    data = github_get(url)
    if data and 'content' in data:
        return base64.b64decode(data['content']).decode('utf-8', errors='replace')
    return None

def save_content(filename, content):
    """Save content to local file."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)

# ============================================================
# REPO 1: Rachnog/Deep-Trading
# ============================================================
print("=" * 60)
print("REPO 1: Rachnog/Deep-Trading")
print("=" * 60)

# Try master, then main
tree = get_repo_tree("Rachnog", "Deep-Trading", "master")
if not tree:
    tree = get_repo_tree("Rachnog", "Deep-Trading", "main")
    branch = "main"
else:
    branch = "master"

print(f"\nBranch: {branch}")
print(f"Total files: {len(tree)}")
print("\nFull tree:")
for path, typ, size in tree:
    print(f"  {typ:4s} {size:>8d}  {path}")

# Fetch all .py and .ipynb files
print("\n\nFetching Python/notebook files...")
py_files = [(p, s) for p, t, s in tree if t == 'blob' and (p.endswith('.py') or p.endswith('.ipynb')) and s < 500000]
for path, size in py_files:
    print(f"\n{'='*60}")
    print(f"FILE: {path} ({size} bytes)")
    print('='*60)
    content = get_file_content("Rachnog", "Deep-Trading", path, branch)
    if content:
        if path.endswith('.ipynb'):
            # Parse notebook, extract code cells only
            try:
                nb = json.loads(content)
                cells = nb.get('cells', [])
                for i, cell in enumerate(cells):
                    if cell.get('cell_type') == 'code':
                        source = ''.join(cell.get('source', []))
                        if source.strip():
                            print(f"\n--- Cell {i} (code) ---")
                            print(source[:3000])
                    elif cell.get('cell_type') == 'markdown':
                        source = ''.join(cell.get('source', []))
                        if any(kw in source.lower() for kw in ['skew', 'kurto', 'volatil', 'feature', 'signal', 'distribution', 'risk', 'sharp', 'portfolio', 'predict']):
                            print(f"\n--- Cell {i} (markdown, relevant) ---")
                            print(source[:1000])
            except json.JSONDecodeError:
                print("[Could not parse notebook JSON]")
        else:
            print(content[:5000])
    time.sleep(0.5)  # Rate limit

print("\n\n")

# ============================================================
# REPO 2: TauricResearch/TradingAgents
# ============================================================
print("=" * 60)
print("REPO 2: TauricResearch/TradingAgents")
print("=" * 60)

tree2 = get_repo_tree("TauricResearch", "TradingAgents", "main")
if not tree2:
    tree2 = get_repo_tree("TauricResearch", "TradingAgents", "master")
    branch2 = "master"
else:
    branch2 = "main"

print(f"\nBranch: {branch2}")
print(f"Total files: {len(tree2)}")
print("\nFull tree:")
for path, typ, size in tree2:
    print(f"  {typ:4s} {size:>8d}  {path}")

# Focus on agents, analysis, signals, risk mgmt files
print("\n\nFetching key Python files...")
key_paths = [p for p, t, s in tree2 if t == 'blob' and p.endswith('.py') and s < 200000]
# Prioritize files in agents, analysis, signal, risk, debate paths
priority_keywords = ['agent', 'analyst', 'signal', 'risk', 'debate', 'sentiment', 'news', 'quant', 'fundamental', 'technic', 'trade', 'market', 'bull', 'bear', 'manager', 'utils', 'config']
priority_files = [p for p in key_paths if any(kw in p.lower() for kw in priority_keywords)]
other_files = [p for p in key_paths if p not in priority_files]

for path in priority_files + other_files[:10]:
    print(f"\n{'='*60}")
    print(f"FILE: {path}")
    print('='*60)
    content = get_file_content("TauricResearch", "TradingAgents", path, branch2)
    if content:
        print(content[:8000])
    time.sleep(0.5)
