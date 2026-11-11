# GBUS 562: AI Language Models

This repository holds coursework for GBUS 562 (AI Language Models) at the University of Mary Washington. The work is building software tools that use large language models, mostly Claude.

## Repository

- Remote: https://github.com/robb-commits/UMW-GBUS-562, branch `main`.
- It is organized into weekly folders (`week-01/`, `week-02/`, ...). Put each week's work in its own folder.
- The root `README.md` has a `## Weekly Work` section with a numbered list, one item per week. Add an item when a new week's work is added.
- Week 1: a single-page Connect 4 game (`week-01/connect-4.html`).
- Week 2: Prompt Lab, a single-file web app for testing prompts against Claude (`week-02/prompt-lab.html`), plus exported runs and a writeup.
- Week 3: a RAG command-line tool over Copperline Manufacturing's finance documents (`week-03/ask.py`). Its Chroma database (`week-03/chroma_db/`) is gitignored; rebuild it with `.venv/bin/python week-03/ask.py --rebuild`.
- Commit and push only when asked. Before pushing, fetch first: the user sometimes edits files directly on GitHub.

## Python

- Always use the virtual environment at `.venv` (top of this repository) for running Python and installing packages: `.venv/bin/python` and `.venv/bin/python -m pip install ...`, run from the repository root. Never use the system `python3` or install packages outside `.venv`.
- This is a Mac: the interpreter is `.venv/bin/python`, not the Windows path `.venv/Scripts/python`.
- `.venv` uses Python 3.13 and is gitignored; never commit it.

## Anthropic API key

- The key is in a plain-text file in the folder that contains this repository (one level up, outside the repo): `API-Key.txt`, or `API Key.txt` as currently named.
- Use it whenever something we build needs to call Claude. Code reads it from that file at run time (as `week-03/ask.py` does), from the `ANTHROPIC_API_KEY` environment variable, or from a `.env` file listed in `.gitignore` (the repo's `.gitignore` already ignores `.env`). Browser apps take the key as user input and never store it.
- Keep the key secret:
  - Never write it into a source file, README, commit, or any file in this repository other than a gitignored `.env`.
  - Never push it to GitHub. Before every commit, check the staged changes for the key (`git diff --cached`) and stop if it appears.
  - Never put it in a published artifact or web page. Browser code is readable by anyone who opens the page, so stored keys and Claude calls with a built-in key belong in server-side or command-line code.
  - Don't print or repeat the key in the conversation.

## Agent email file

- `Agent-Email.txt` is in the folder above this repository (outside the repo). Programs we build may read it at run time.
- Never open, read, print, or search it yourself (no Read, cat, grep, or similar), and never write its contents to any file.
