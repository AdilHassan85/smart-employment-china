# Smart Employment Prediction & Skill Recommendation System (Global AI/Tech Job Market, China Focus)

ML system jo global AI/Tech job market data (14 countries) par employment
probability predict karta hai, `country` ko ek feature ki tarah use karta hai
taake China-specific predictions bhi diye ja sakein, aur personalized
skill/career recommendations deta hai.

## Problem statement

AI adoption ki wajah se global job market tezi se badal raha hai, aur China
is shift ka ek central player hai — manufacturing-se-tech shift, aging
workforce, aur graduate "slow employment" jaisay masail ke sath. Fresh
graduates aksar yeh nahi jaante ke unki konsi skills employment probability
improve karti hain, aur yeh sawal China jaise fast-evolving markets mein aur
bhi mushkil ho jata hai jahan trends global benchmarks se alag ho sakte hain.
Existing research is problem ko do alag hisso mein handle karta hai - ya sirf
prediction, ya sirf recommendation. Yeh project ek ML system develop karta
hai jo (1) global AI/Tech job data (14 countries) par train ho kar
profile-based employment probability predict kare, (2) `country` feature ke
zariye China-specific predictions de sake, (3) SHAP explainability ke sath
predictions ko transparent banaye, (4) ranked skills aur career paths
recommend kare, aur (5) China vs global market ka comparative analysis pesh
kare (salary, skill demand, remote-work trends mein farq).

## Project structure

```
smart-employment-china/
├── data/               # raw, interim, processed, external data
├── notebooks/          # EDA aur exploration notebooks
├── src/
│   ├── data/            # cleaning
│   ├── features/        # feature engineering
│   ├── models/           # training, tuning, evaluation
│   └── recommend/        # skill recommendation, job matching
├── models/             # saved .pkl files
├── reports/            # figures, research paper
├── app/                 # Streamlit application
└── tests/
```

## How to run

```bash
pip install -r requirements.txt --break-system-packages

# Pipeline chalao (dataset data/raw/ mein hona chahiye)
python src/data/clean_data.py
python src/features/build_features.py
python src/models/train.py

# App chalao
streamlit run app/streamlit_app.py
```

## Dataset

- AI Jobs Market 2025-2026 | Salaries (Kaggle) - 1,500 AI/ML job postings,
  25 roles, 14 countries including China
- Reference stats: National Bureau of Statistics of China, World Bank

## China-focus strategy

Poora dataset (saare 14 countries) training ke liye use hota hai taake model
robust rahe. `country` ko ek feature ki tarah treat kiya jata hai, is liye
inference ke waqt `country = "China"` set kar ke China-specific predictions
li ja sakti hain. EDA mein ek dedicated "China vs Global" comparison section
hoga.

## Author

Adil Hassan - Computer Engineering, UET Lahore (Faisalabad campus)
