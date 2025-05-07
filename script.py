import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import shap
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv(r"C:\Users\Lavanya\Downloads\archive\WA_Fn-UseC_-HR-Employee-Attrition.csv")

print(df.head())

print(df.info())

print(df.describe())

print(df.isnull().sum())

df.drop(['EmployeeCount', 'Over18', 'StandardHours', 'EmployeeNumber'], axis=1, inplace=True)

df['Attrition_encoded'] = df['Attrition'].astype('category').cat.codes

df[['Attrition', 'Attrition_encoded']].head()

attrition_count = df['Attrition'].value_counts()

plt.figure(figsize= (10,5))
sns.barplot(x= attrition_count.index, y= attrition_count.values, palette= 'viridis')

plt.title('Employee Attrition Count')
plt.xlabel('Attrition')
plt.ylabel('Employee Count')
plt.show()

plt.figure(figsize= (10,6))
sns.countplot(data=df, x='Department', hue='Attrition', palette='viridis')
plt.title('Attrition by Department')
plt.xlabel('Department')
plt.ylabel('Number of Employees')
plt.xticks(rotation=45)  
plt.show()

plt.figure(figsize= (10,5))
sns.countplot(data= df, x= 'JobRole', hue= 'Attrition', palette= 'Set2')
plt.title("Attrition by Job Role", fontsize=16)
plt.xlabel("Job Role", fontsize=12)
plt.ylabel("Number of Employees", fontsize=12)
plt.xticks(rotation=45)
plt.legend(title='Attrition')
plt.tight_layout()
plt.show()


attrition_counts = (
    df.groupby('JobRole')['Attrition']
    .value_counts(normalize=True)
    .unstack()
    .fillna(0)
)

attrition_rate = attrition_counts['Yes'].sort_values(ascending=False) * 100

plt.figure(figsize=(12, 6))
sns.barplot(x=attrition_rate.index, y=attrition_rate.values, palette='coolwarm')
plt.title("Attrition Rate (%) by Job Role", fontsize=16)
plt.xlabel("Job Role", fontsize=12)
plt.ylabel("Attrition Rate (%)", fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

satisfaction_cols = ['EnvironmentSatisfaction', 'JobSatisfaction', 'WorkLifeBalance']
for col in satisfaction_cols:
    plt.figure(figsize=(7,5))
    sns.countplot(data=df, x=col, hue='Attrition', palette='Set2')
    plt.title(f"Attrition by {col}")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.show()

plt.figure(figsize=(10,6))
sns.kdeplot(data=df, x='DistanceFromHome', hue='Attrition', fill=True, common_norm=False, palette='viridis')
plt.title("Attrition vs. Distance From Home")
plt.show()

plt.figure(figsize= (10,6))
sns.boxplot(x= 'Attrition', y= 'MonthlyIncome', data= df, palette= 'Set2')
plt.title('Attrition by Monthly Income')
plt.show()

plt.figure(figsize=(10,5))
sns.histplot(data=df, x='Age', hue='Attrition', kde=True, element='step', stat='density', common_norm=False)
plt.title("Attrition by Age")
plt.show()

plt.figure(figsize= (16,12))
sns.heatmap(df.corr(numeric_only= True), cmap= 'rocket', annot=True, fmt=".1f")
plt.title('Correlation Heatmap')
plt.show()


# Encode categorical features
cat_cols = df.select_dtypes(include='object').columns.tolist()
cat_cols.remove('Attrition')  # Already encoded

le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# 🔀 Step 4: Train-Test Split
X = df.drop(['Attrition', 'Attrition_encoded'], axis=1)
y = df['Attrition_encoded']

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)

# ⚙️ Step 5: Train Logistic Regression Model
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)

# 🔍 Step 6: Make Predictions
y_pred = log_model.predict(X_test)

# 📊 Step 7: Evaluate Model
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 📈 Step 8: SHAP Value Analysis (Model Explainability)
explainer = shap.Explainer(log_model, X_train)
shap_values = explainer(X_test)

# SHAP summary plot
shap.summary_plot(shap_values, X_test, feature_names=X.columns, plot_type="bar")
shap.summary_plot(shap_values, X_test, feature_names=X.columns)