from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import shap  # SHAP for explainability
import matplotlib.pyplot as plt  # For beautiful plots
import numpy as np  # For Top 10 features selection

# 1. Load & Clean
df = pd.read_csv(r"C:\Users\Lavanya\Downloads\archive\WA_Fn-UseC_-HR-Employee-Attrition.csv")
df.drop(['EmployeeCount', 'Over18', 'StandardHours', 'EmployeeNumber'], axis=1, inplace=True)

# 2. Encode
df['Attrition_encoded'] = df['Attrition'].astype('category').cat.codes
cat_cols = df.select_dtypes(include='object').columns.tolist()
cat_cols.remove('Attrition')
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

X = df.drop(['Attrition', 'Attrition_encoded'], axis=1)
y = df['Attrition_encoded']

# 3. Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Feature Selection
model = LogisticRegression(solver='liblinear')
rfe = RFE(model, n_features_to_select=15)
X_rfe = rfe.fit_transform(X_scaled, y)
selected_features = X.columns[rfe.support_]

# 5. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_rfe, y, test_size=0.2, stratify=y, random_state=42)

# 6. Grid Search for Best Params
param_grid = {
    'C': [0.01, 0.1, 1, 10],
    'penalty': ['l1', 'l2']
}
grid = GridSearchCV(LogisticRegression(solver='liblinear', class_weight='balanced'), param_grid, scoring='f1', cv=5)
grid.fit(X_train, y_train)

best_lr = grid.best_estimator_

# 7. Evaluation
y_pred = best_lr.predict(X_test)
print("Best Logistic Regression Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 8. SHAP Analysis
print("\nGenerating SHAP values...")

# Create an explainer
explainer = shap.LinearExplainer(best_lr, X_train, feature_perturbation="interventional")
shap_values = explainer.shap_values(X_test)

# 8.1 Bar Plot - All Features
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, features=X_test, feature_names=selected_features.tolist(), plot_type="bar", show=False)
plt.title("Feature Importance - SHAP Summary (Bar Plot)", fontsize=18)
plt.xticks(fontsize=12, rotation=45, ha='right')
plt.tight_layout()
plt.show()

# 8.2 Beeswarm Plot - All Features
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, features=X_test, feature_names=selected_features.tolist(), show=False)
plt.title("Feature Importance - SHAP Summary (Beeswarm Plot)", fontsize=18)
plt.xticks(fontsize=12, rotation=45, ha='right')
plt.tight_layout()
plt.show()

# 8.3 (Optional) Bar Plot - Top 10 Features Only
mean_shap = np.abs(shap_values).mean(axis=0)
top_indices = np.argsort(mean_shap)[-10:]
X_test_top10 = X_test[:, top_indices]
selected_features_top10 = [selected_features.tolist()[i] for i in top_indices]

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values[:, top_indices], features=X_test_top10, feature_names=selected_features_top10, plot_type="bar", show=False)
plt.title("Top 10 Important Features (SHAP Bar Plot)", fontsize=18)
plt.xticks(fontsize=12, rotation=45, ha='right')
plt.tight_layout()
plt.show()

# 8.4 (Optional) Force Plot - Single Prediction Explanation
index = 5  # Any index you want to visualize
shap.force_plot(
    explainer.expected_value, 
    shap_values[index], 
    features=X_test[index], 
    feature_names=selected_features.tolist(), 
    matplotlib=True
)
