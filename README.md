We will be using readme file for giving important notes.


## 🚀 Team Setup

### 1. Clone the Repository

```bash
git clone <REPOSITORY-URL>
cd AI-Based_Plantation_Recommendation_System
```

### 2. Create Conda Environment

Make sure Miniconda/Anaconda is installed, then run:

```bash
conda env create -f environment.yml

conda activate plantation-ml
```

### 3. Select Python Interpreter

In Antigravity/VS Code:

`Ctrl + Shift + P` → **Python: Select Interpreter** → select **plantation-ml**.

### 4. Verify Setup

```bash
python --version
python -c "import numpy, pandas, matplotlib, sklearn; print('Everything is working!')"
```


* Do **not** commit `.env` files, API keys, passwords, or other secrets.
* Pull the latest `main` before starting new work.
