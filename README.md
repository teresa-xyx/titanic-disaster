# Titanic Disaster: Survival Prediction in Python and R
 
This project predicts passenger survival on the Titanic with a logistic regression model. The same analysis is written twice, once in **Python** and once in **R**, and each version runs in its own **Docker** container. 
 
Both scripts:
 
1. Load and explore `train.csv`, `test.csv`, and `gender_submission.csv`
2. Add features (`Title`, `FamilySize`, `IsAlone`, `HasCabin`) and fill missing values using values learned from the train data only
3. Fit a logistic regression model on the train data
4. Report accuracy on the training set
5. Predict survival for every passenger in the test set and print the predictions
 
---
 
## Installation and running
 
### 0. Prerequisites
 
You need **Git** and **Docker Desktop** installed:
 
| Tool | Download |
| --- | --- |
| Git | https://git-scm.com/downloads |
| Docker Desktop | https://www.docker.com/products/docker-desktop/ |
 
**Open Docker Desktop and wait until it shows "Engine running"** in the bottom-left corner before building or running the containers.

### 1. Clone the repository and enter it
 
```bash
git clone https://github.com/teresa-xyx/titanic-disaster.git
cd titanic-disaster
```
  
### 2. Download the data
 
The data is **not** included in this repository (the `src/data/` folder is excluded by `.gitignore`), so you need to download it from Kaggle:
 
1. Sign in to https://www.kaggle.com
2. Go to the Titanic competition data page: https://www.kaggle.com/competitions/titanic/data
3. Download `titanic.zip`.
4. Unzip it. You should have three files:
   - `train.csv` (891 rows, 12 columns, includes `Survived`)
   - `test.csv` (418 rows, 11 columns, no `Survived` column)
   - `gender_submission.csv` (418 rows, 2 columns)
5. Create the `src/data/` folder, which does not exist after cloning:
```bash
   mkdir -p src/data          # macOS / Linux / Git Bash
   mkdir src\data             # Windows PowerShell or Command Prompt
```
6. Move all three files into `src/data/`.
When you are done, the folder should look like this. The file names must match exactly (all lowercase), because both scripts read these names.
 
```
src/data/
├── gender_submission.csv
├── test.csv
└── train.csv
```

### 3. Run the Python version
 
**Build the image:**
 
```bash
docker build -t titanic-app .
```
 
This uses the `Dockerfile` in the repository root.
 
**Run the container:**
 
The data is not copied into the image, so the `-v` option mounts your local `src/data` folder into the container at `/app/src/data`. Use the command for your terminal:
 
| Terminal | Command |
| --- | --- |
| macOS / Linux | `docker run --rm -v "$(pwd)/src/data:/app/src/data" titanic-app` |
| Windows PowerShell | `docker run --rm -v "${PWD}\src\data:/app/src/data" titanic-app` |
| Windows Command Prompt | `docker run --rm -v "%cd%\src\data:/app/src/data" titanic-app` |
| Windows Git Bash | `MSYS_NO_PATHCONV=1 docker run --rm -v "$(pwd -W)/src/data:/app/src/data" titanic-app` |
 

### 4. Run the R version
 
**Build the image:**
 
```bash
docker build -f src/r_app/Dockerfile -t titanic-r .
```
 
- `-f src/r_app/Dockerfile` selects the R Dockerfile, since the repository root already contains the Python one.
- The final `.` must still be the repository root, so that the paths inside the Dockerfile and the root `.dockerignore` work correctly.
 
**Run the container** (same as the Python version, with the image name `titanic-r`):
 
| Terminal | Command |
| --- | --- |
| macOS / Linux | `docker run --rm -v "$(pwd)/src/data:/app/src/data" titanic-r` |
| Windows PowerShell | `docker run --rm -v "${PWD}\src\data:/app/src/data" titanic-r` |
| Windows Command Prompt | `docker run --rm -v "%cd%\src\data:/app/src/data" titanic-r` |
| Windows Git Bash | `MSYS_NO_PATHCONV=1 docker run --rm -v "$(pwd -W)/src/data:/app/src/data" titanic-r` |
 
---
 
## Expected output
 
**Python** starts with:
 
```
============================================================
Problem 13
============================================================
   PassengerId  Survived  Pclass  ...     Fare Cabin Embarked
0            1         0       3  ...   7.2500   NaN        S
1            2         1       1  ...  71.2833   C85        C
2            3         1       3  ...   7.9250   NaN        S
3            4         1       1  ...  53.1000  C123        S
4            5         0       3  ...   8.0500   NaN        S

[5 rows x 12 columns]
```
 
**R** starts with:
 
```
============================================================
Problem 23: Load packages and train.csv
============================================================
R version: R version 4.4.2 (2024-10-31)
readr version: 2.1.5 | dplyr version: 1.1.4
Reading data from: /app/src/data
train.csv loaded: 891 rows, 12 columns
```
---
 
## Notes on the analysis
 
- **The test set has no true labels.** `test.csv` does not include `Survived`. Test set accuracy therefore cannot be measured, so both scripts print the predicted survival for every test passenger instead.
- All fill values (median age by title, most common port, median fare) are learned from the training data and then applied to the test data.
- `Sex` is not used as a separate feature because `Title` (Mr, Mrs, Miss, Master, Rare) already encodes it; using both caused  multi-collinearity, and the unregularized model failed to converge. `SibSp` and `Parch` are combined into `FamilySize`.
---
 
## Repository structure
 
```
titanic-disaster/
├── README.md                # This file
├── Dockerfile               # Python container (Python 3.14)
├── requirements.txt         # Python packages with pinned versions
├── .dockerignore            # Keeps data, .venv, and .git out of the images
├── .gitignore               # Keeps data, .venv, and system files out of Git
├── .github/
│   └── CODEOWNERS
└── src/
    ├── data/                # Put the CSV files here (not tracked by Git)
    ├── app/
    │   └── main.py          # Python script
    └── r_app/
        ├── Dockerfile       # R container (R 4.4.2)
        ├── install_packages.R
        └── titanic.R        # R script  
```
