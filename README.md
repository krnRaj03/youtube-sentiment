# youtube-sentiment

This project performs sentiment analysis on YouTube comments. It covers the complete ML pipeline — from data ingestion, preprocessing, exploratory analysis, to model training and evaluation. The project uses LightGBM as the final model and is version-controlled with DVC.

🚀 Project Workflow

Data Ingestion

Collect YouTube comments data.

Store in structured format for reproducibility.

Data Preprocessing

Cleaning (remove stopwords, punctuation, emojis, etc.)

Tokenization and vectorization (TF-IDF / embeddings).

Handling missing values.

Exploratory Data Analysis (EDA)

Visualize sentiment distribution.

Word clouds, top positive/negative keywords.

Correlation and feature importance plots.

Model Training

Trained with multiple tree-based algorithms (Random Forest, Gradient Boosting, etc.).

LightGBM chosen as the final model for best performance.

Experiment Tracking & Versioning

DVC used to track datasets, experiments, and pipelines.

Pipeline stages defined in dvc.yaml.

🛠️ Tech Stack

Python 3.x

Pandas, NumPy – data handling

Scikit-learn – preprocessing & baselines

LightGBM – final ML model

Matplotlib, Seaborn – visualization

DVC – experiment tracking & reproducibility
