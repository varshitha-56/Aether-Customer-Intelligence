# app/main.py
import pandas as pd
import numpy as np
import joblib
import os
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, confusion_matrix, roc_curve
from imblearn.over_sampling import SMOTE
from sklearn.cluster import KMeans

warnings.filterwarnings('ignore')
os.makedirs('artifacts', exist_ok=True)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates().reset_index(drop=True)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        df[col] = df[col].fillna(df[col].median() if df[col].notna().sum() > 0 else 0)
    
    for col in cat_cols:
        mode_val = df[col].mode()
        df[col] = df[col].fillna(mode_val[0] if not mode_val.empty else 'Unknown')
    return df


def auto_detect_columns(df):
    id_col = None
    target_col = None
    for col in df.columns:
        if col.lower() in ['customerid', 'customer_id', 'id', 'userid']:
            id_col = col
        if col.lower() in ['churn', 'churnvalue', 'churn_label', 'attrition', 'target', 'churned']:
            target_col = col
    return id_col or df.columns[0], target_col or df.columns[-1]


def apply_smote(X_train, y_train):
    if len(np.unique(y_train)) < 2:
        return X_train, y_train, "Single class detected - SMOTE skipped"
    minority_ratio = np.min(np.bincount(y_train)) / len(y_train)
    if minority_ratio < 0.30:
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X_train, y_train)
        return X_res, y_res, f"✅ SMOTE Applied (ratio: {minority_ratio:.3f})"
    return X_train, y_train, f"SMOTE skipped (balanced)"


def train_and_evaluate(X_train, y_train, X_test, y_test, preprocessor):
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    
    if len(np.unique(y_train)) < 2:
        return {"Error": ["Only one class in target column. Please select a different target."]}, None, None, "Failed", preprocessor, "Single class"
    
    X_train_res, y_train_res, smote_msg = apply_smote(X_train_proc, y_train)
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, C=0.5, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.1, random_state=42, eval_metric='logloss'),
        "ANN": MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=400, alpha=0.01, random_state=42)
    }
    
    results = {}
    trained = {}
    
    for name, model in models.items():
        model.fit(X_train_res, y_train_res)
        y_pred = model.predict(X_test_proc)
        y_prob = model.predict_proba(X_test_proc)[:, 1] if hasattr(model, "predict_proba") else np.zeros(len(y_test))
        
        results[name] = {
            "Accuracy": round(accuracy_score(y_test, y_pred), 4),
            "ROC AUC": round(roc_auc_score(y_test, y_prob), 4),
            "Recall": round(recall_score(y_test, y_pred, pos_label=1), 4)
        }
        trained[name] = model
    
    # Voting Ensemble
    estimators = [(n, trained[n]) for n in list(models.keys())[:3]]
    voting = VotingClassifier(estimators=estimators, voting='soft')
    voting.fit(X_train_res, y_train_res)
    y_pred = voting.predict(X_test_proc)
    y_prob = voting.predict_proba(X_test_proc)[:, 1]
    
    results["Voting Ensemble"] = {
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "ROC AUC": round(roc_auc_score(y_test, y_prob), 4),
        "Recall": round(recall_score(y_test, y_pred, pos_label=1), 4)
    }
    
    
    best_name = max(results, key=lambda k: results[k]["ROC AUC"])
    return results, trained, voting, best_name, preprocessor, smote_msg


def perform_segmentation(df, n_clusters=4):
    numeric_cols, _ = get_column_types(df)
    if len(numeric_cols) < 2:
        return df.copy(), None, None, numeric_cols
    seg_cols = numeric_cols[:min(6, len(numeric_cols))]
    X = df[seg_cols].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df = df.copy()
    df['Segment'] = kmeans.fit_predict(X_scaled)
    return df, kmeans, scaler, seg_cols


def get_column_types(df):
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    return numeric, categorical


def save_artifacts(best_model, preprocessor, results_df, df, cm, roc_df):
    joblib.dump(best_model, 'artifacts/best_model.pkl')
    joblib.dump(preprocessor, 'artifacts/preprocessor.pkl')
    results_df.to_csv('artifacts/model_results.csv', index=False)
    pd.DataFrame(cm).to_csv('artifacts/confusion_matrix.csv', index=False)
    roc_df.to_csv('artifacts/roc_curve.csv', index=False)
    df.to_csv('artifacts/processed_customers.csv', index=False)