# 🛠 Code Quality & Formatting Guidelines

This project enforces standard Python code quality using **Ruff** (formatting and linting) and **pre-commit** hooks.

## Environment Setup

Automated checks run on `git commit` via pre-commit, and on CI pipelines via GitHub Actions.

1. **Install hooks locally:**
   ```bash
   pip install pre-commit ruff
   pre-commit install
   pre-commit install --hook-type commit-msg
   ```
   *This ensures formatting, linting, and conventional commit validations run before every commit.*

## Manual Execution Commands

**Run all checks on all files (Repository-wide):**
```bash
pre-commit run --all-files
```

**Run Ruff exclusively:**
```bash
# Linting
ruff check .
# Formatting
ruff format .
```

## Emergency Bypassing
If a hotfix must be committed urgently, you can bypass the pre-commit checks using:
```bash
git commit --no-verify -m "fix: emergency patch"
```
*Warning: CI pipelines will still enforce linting, so use this bypass carefully.*