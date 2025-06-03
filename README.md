# ABM Project — *Title TBD*

[![Pytest](https://github.com/MarcellSzegedi/abm-project/actions/workflows/pytest.yaml/badge.svg)](https://github.com/MarcellSzegedi/abm-project/actions/workflows/pytest.yaml)
&nbsp;
[![Mypy Type Check](https://github.com/MarcellSzegedi/abm-project/actions/workflows/mypy.yaml/badge.svg)](https://github.com/MarcellSzegedi/abm-project/actions/workflows/mypy.yaml)
&nbsp;
[![Linting & Formatting](https://github.com/MarcellSzegedi/abm-project/actions/workflows/ruff.yaml/badge.svg)](https://github.com/MarcellSzegedi/abm-project/actions/workflows/ruff.yaml)

<br>

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)  
![Version](https://img.shields.io/badge/version-0.1.0-orange)

---

## 🚀 &nbsp; Running the Model

*Instructions coming soon.*

---

## 🤝 &nbsp; Contributing

To contribute to this project, please follow the guidelines below:

#### 🔧 &nbsp; 1. &nbsp; Setup

To synchronize all dependencies for the current version of the project, run:

```bash
uv sync
```
Also, don't forget to set the Python interpreter to the following path:

`.venv/bin/python3`

and activate the virtual environment in your terminal using the following command:

```bash
source .venv/bin/activate
```

#### 🌿 &nbsp; 2. &nbsp; Create a Feature Branch

```bash
git checkout -b <BRANCH_GROUP>/<BRANCH_NAME>
```
Use one of the following branch group prefixes based on the nature of your contribution:
- feature
- test
- refactor

Please always use meaningful branch names!

#### ✅ &nbsp; 3. &nbsp; Run Checks Before Commiting

Before committing any changes, run the following command to ensure the code is properly linted, formatted, and that all static type checks pass:

```bash
pre-commit
```

Before committing, also make sure that all tests pass. Upon failure, don't forget to stage the changes using:

```bash
git add .
```
This ensures that your fixes are included in the test run. Otherwise, unstaged changes will be stashed and not considered during testing.

#### 🫛 &nbsp; 4. &nbsp; Merging Changes

The `main` branch of the project is protected by multiple branch rules. All checks — including linting, formatting, unit tests, and static type checks — must pass before a branch can be considered for merging into `main`. Additionally, at least one administrator's approval is required.
