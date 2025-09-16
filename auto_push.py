#!/usr/bin/env python3
"""
auto_push.py — Watch a Git repo and auto-push when selected changes are detected.

USAGE (from inside your repo or point --repo to it):

  python auto_push.py \
    --include "src/**" --include "*.py" \
    --exclude "docs/**" --exclude "*.md" \
    --interval 3 --debounce 10 \
    --remote origin --branch "" \
    --message "chore(auto): push"

Notes: Belal is a pussy
- Make sure `git` is installed and your auth to GitHub is already set up (SSH key or credential manager).
- By default, it includes only *.py and src/** and excludes docs/** and *.md. Adjust as you like.
- If --branch is omitted (empty), the current branch is used.
- Debounce waits for the repo to be quiet for N seconds before committing & pushing.

Stop with Ctrl+C.
"""
import argparse
import datetime as dt
import fnmatch
import subprocess
import sys
import time
from pathlib import Path, PurePosixPath

# ----------------------- helpers -----------------------

def run(cmd, cwd, check=False):
    """Run a shell command, return (code, stdout, stderr)."""
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        shell=False,
    )
    if check and proc.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{proc.stderr}")
    return proc.returncode, proc.stdout.strip(), proc.stderr.strip()

def ensure_repo(repo: Path):
    if not repo.exists():
        sys.exit(f"[!] Repo path does not exist: {repo}")
    code, out, _ = run(["git", "rev-parse", "--is-inside-work-tree"], repo)
    if code != 0 or out != "true":
        sys.exit(f"[!] Not a git repository: {repo}")
    return True

def current_branch(repo: Path) -> str:
    code, out, err = run(["git", "rev-parse", "--abbrev-ref", "HEAD"], repo, check=True)
    return out

def has_upstream(repo: Path) -> bool:
    code, out, err = run(["git", "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"], repo)
    return code == 0

def list_changed_paths(repo: Path) -> list[str]:
    """
    Return modified, deleted, and untracked files (respecting .gitignore).
    This avoids parsing 'git status' output.
    """
    code, out, err = run(
        ["git", "ls-files", "-m", "-o", "-d", "--exclude-standard"],
        repo,
        check=True,
    )
    paths = [p for p in out.splitlines() if p.strip()]
    return [PurePosixPath(Path(p).as_posix()).as_posix() for p in paths]


def glob_filter(paths: list[str], includes: list[str], excludes: list[str]) -> list[str]:
    # If includes is empty, treat as ["**"] (everything)
    if not includes:
        includes = ["**"]
    def match_any(pat_list, path):
        return any(fnmatch.fnmatch(path, pat) for pat in pat_list)
    results = []
    for p in paths:
        if match_any(includes, p) and not match_any(excludes, p):
            results.append(p)
    return results

def stage_selected(repo: Path, files: list[str]) -> bool:
    if not files:
        return False
    # Use '--' to terminate pathspec and add explicit files
    cmd = ["git", "add", "--"] + files
    code, out, err = run(cmd, repo)
    if code != 0:
        print(f"[!] git add failed:\n{err}")
        return False
    return True

def is_nothing_to_commit(repo: Path) -> bool:
    code, out, err = run(["git", "diff", "--cached", "--name-only"], repo)
    return out.strip() == ""

def commit(repo: Path, message: str, files_for_body: list[str]) -> bool:
    # Add a short body listing files (max 20 to keep it tidy)
    listed = files_for_body[:20]
    more = len(files_for_body) - len(listed)
    ts = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    body_lines = [f"auto-push at {ts}"]
    body_lines += ["", "files:"] + [f"- {p}" for p in listed]
    if more > 0:
        body_lines.append(f"... and {more} more file(s)")
    cmd = ["git", "commit", "-m", message, "-m", "\n".join(body_lines)]
    code, out, err = run(cmd, repo)
    if code != 0:
        # nothing to commit is not necessarily an error here
        if "nothing to commit" in (out + err).lower():
            return False
        print(f"[!] git commit failed:\n{err or out}")
        return False
    return True

def pull_rebase_if_possible(repo: Path, remote: str, branch: str):
    if has_upstream(repo):
        # Use upstream as configured
        run(["git", "pull", "--rebase", "--autostash"], repo)
    else:
        # Try explicit remote/branch if provided
        if remote and branch:
            run(["git", "pull", "--rebase", "--autostash", remote, branch], repo)

def push(repo: Path, remote: str, branch: str):
    if has_upstream(repo):
        code, out, err = run(["git", "push"], repo)
        if code != 0:
            print(f"[!] git push failed:\n{err or out}")
    else:
        if not branch:
            branch = current_branch(repo)
        code, out, err = run(["git", "push", "-u", remote, branch], repo)
        if code != 0:
            print(f"[!] git push failed:\n{err or out}")

# ----------------------- main loop -----------------------

def main():
    parser = argparse.ArgumentParser(description="Auto-push Git repo when selected changes are detected.")
    parser.add_argument("--repo", type=str, default=".", help="Path to the Git repository (default: current dir)")
    parser.add_argument("--include", action="append", default=[], help="Glob of files to include (can repeat). Default: '*.py' and 'src/**'")
    parser.add_argument("--exclude", action="append", default=[], help="Glob of files to exclude (can repeat). Default: '*.md' and 'docs/**'")
    parser.add_argument("--interval", type=int, default=3, help="Polling interval in seconds (default: 3)")
    parser.add_argument("--debounce", type=int, default=10, help="Debounce seconds after last change before commit (default: 10)")
    parser.add_argument("--remote", type=str, default="origin", help="Git remote name (default: origin)")
    parser.add_argument("--branch", type=str, default="", help="Branch to push (default: current branch)")
    parser.add_argument("--message", type=str, default="chore(auto): push", help="Commit message (default: 'chore(auto): push')")
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    ensure_repo(repo)

    # Defaults if none specified
    includes = args.include[:] if args.include else ["src/**", "*.py"]
    excludes = args.exclude[:] if args.exclude else ["docs/**", "*.md", ".git/**"]

    if not args.branch:
        try:
            args.branch = current_branch(repo)
        except Exception:
            pass

    print(f"[auto-push] repo={repo}")
    print(f"[auto-push] watching includes={includes} excludes={excludes}")
    print(f"[auto-push] remote={args.remote} branch={args.branch or '(current)'}")
    print(f"[auto-push] interval={args.interval}s debounce={args.debounce}s")
    print("[auto-push] Press Ctrl+C to stop.\n")

    last_change_detected_at = None
    last_seen_set = set()  # to detect transitions; not strictly required

    try:
        while True:
            changed_all = list_changed_paths(repo)
            # Only consider included/excluded
            interesting = sorted(set(glob_filter(changed_all, includes, excludes)))

            # For visibility
            if set(interesting) != last_seen_set:
                if interesting:
                    print(f"[{dt.datetime.now().strftime('%H:%M:%S')}] detected changes: {', '.join(interesting)}")
                last_seen_set = set(interesting)

            if interesting:
                if last_change_detected_at is None:
                    last_change_detected_at = time.time()
                # If it's been quiet enough, do the commit/push
                quiet_for = time.time() - last_change_detected_at
                if quiet_for >= args.debounce:
                    print(f"[auto-push] Debounce elapsed ({args.debounce}s). Committing & pushing...")
                    # Stage only the interesting files
                    if stage_selected(repo, interesting):
                        if not is_nothing_to_commit(repo):
                            # Pull/rebase (best effort)
                            pull_rebase_if_possible(repo, args.remote, args.branch)
                            # Commit
                            if commit(repo, args.message, interesting):
                                # Push
                                push(repo, args.remote, args.branch)
                                print("[auto-push] ✔ pushed.\n")
                            else:
                                print("[auto-push] nothing to commit or commit failed.\n")
                        else:
                            print("[auto-push] nothing staged to commit.\n")
                    # Reset debounce window after an attempt
                    last_change_detected_at = None
                    last_seen_set = set()
            else:
                last_change_detected_at = None

            time.sleep(max(1, args.interval))
    except KeyboardInterrupt:
        print("\n[auto-push] Stopped by user.")

if __name__ == "__main__":
    main()
