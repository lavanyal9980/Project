from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd

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
