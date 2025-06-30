# ABM for the Influence of Urban Design on Riot Emergence

[![Pytest](https://github.com/MarcellSzegedi/abm-project/actions/workflows/pytest.yaml/badge.svg)](https://github.com/MarcellSzegedi/abm-project/actions/workflows/pytest.yaml)
&nbsp;
[![Linting & Formatting](https://github.com/MarcellSzegedi/abm-project/actions/workflows/ruff.yaml/badge.svg)](https://github.com/MarcellSzegedi/abm-project/actions/workflows/ruff.yaml)

<br>

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)  
![Version](https://img.shields.io/badge/version-1.1.0-orange)

---

## 🚀 &nbsp; Running the Model

#### 🔧 &nbsp; 1. &nbsp; Setup

Before running the model, ensure all tools and dependencies used in the ABM package are synchronised.
Run the following command in your terminal:

```bash
uv sync
```
Then, activate the virtual environment created by UV:
```bash
source .venv/bin/activate
```
#### 🎬 &nbsp; 2. &nbsp; Animation

To run a single simulation of the model and view the results as an animation, use:
```bash
uv run abm animate --n-step <MAX_NUMBER_OF_STEPS_TO_ANIMATE>
```
The resulting animation will be saved to the root folder of the project as `abm_animation.html`.

#### 📊 &nbsp; 3. &nbsp; Sobol Sensitivity Analysis

To generate the Sobol sensitivity analysis for the ABM model using the default settings from the report, run:
```bash
uv run abm sobol --results-type <RESULTS_TYPE>
```
where `<RESULTS_TYPE>` can be either `"Rioter"` or `"Injured"`, depending on the agent state you wish to analyse.

The output files will be saved in the project root as:
- `sobol_sensitivity_analysis.png`
- `sobol_interaction_heatmap.png`

#### ⚙️ &nbsp; 4. &nbsp; Parameter Variation Analysis

To analyse how the number of rioters or injured agents changes when varying different model parameters (street width, exit spacing, number of streets, and exit door spacing), run:
```bash
uv run abm plot-riot-all --results-type <RESULTS_TYPE>
```
where `<RESULTS_TYPE>` can be either `"Rioter"` or `"Injured"`, depending on the agent state you wish to analyse.

The result will be saved in the root folder named `parameter_variation.png`.

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
