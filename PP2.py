import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Load your cleaned dataset
df = pd.read_csv("cleaned_product_details.csv")

# --- Feature Selection ---
features = ['Final Quantity', 'Overall Revenue', 'Category']
target = 'Return_Flag'

# Drop missing target values
df = df.dropna(subset=[target])

# Define X and y
X = df[features]
y = df[target]

# --- Preprocessing Pipelines ---
numeric_features = ['Final Quantity', 'Overall Revenue']
categorical_features = ['Category']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='mean'), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# --- Full Pipeline ---
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])

# --- Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Model Training ---
pipeline.fit(X_train, y_train)

# --- Predictions ---
y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]

# --- Evaluation ---
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_proba))


# Load the cleaned dataset (if starting fresh)
df = pd.read_csv("cleaned_product_details.csv")

# Ensure 'Return_Flag' exists and model was trained on these features
features = ['Final Quantity', 'Overall Revenue', 'Category']
X = df[features]

# === Apply Trained Model ===
# Predict return probabilities using the trained pipeline (from Step 3)
df['Return_Probability'] = pipeline.predict_proba(X)[:, 1]

# === Define High-Risk Threshold ===
# Products with return probability > 0.8 are considered high risk
high_risk = df[df['Return_Probability'] > 0.8]

# === Optional: Add Risk Labels for Power BI readability ===
df['Risk_Level'] = pd.cut(df['Return_Probability'], 
                          bins=[0, 0.3, 0.7, 1], 
                          labels=['Low', 'Medium', 'High'],
                          include_lowest=True)

# === Export CSV ===
# 1. Full dataset with predicted probabilities and risk labels
df.to_csv("product_risk_predictions.csv", index=False)

# 2. Only high-risk products
high_risk.to_csv("high_risk_products.csv", index=False)

# === Summary Output ===
print(f"✅ Total high-risk products exported: {len(high_risk)}")
print("📁 Files saved:\n- product_risk_predictions.csv\n- high_risk_products.csv") 