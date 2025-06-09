1. Customer-Booking-Prediction-Using-Machine-Learning-British-Airways
This project focuses on developing a data-driven model to estimate the percentage of customers eligible for British Airways (BA) lounge access at Heathrow Terminal 3. Lounge access is a key component of BA’s premium customer experience, and accurate eligibility forecasting helps optimize space, resources, and customer satisfaction.

2. This project focuses on developing a data-driven model to estimate the percentage of customers eligible for British Airways (BA) lounge access at Heathrow Terminal 3. Lounge access is a key component of BA’s premium customer experience, and accurate eligibility forecasting helps optimize space, resources, and customer satisfaction.

Python Code:
import pandas as pd
df = pd.read_csv(r"C:\Users\Dell\Downloads\customer_booking.csv", encoding="ISO-8859-1")
df
df.head()
df.info()
df.tail()
df.shape
df.describe()
df.isnull().sum()
df = df.dropna()
df.isnull().sum()
from sklearn.preprocessing import LabelEncoder
for col in df.select_dtypes(include = 'object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    df.head()
    target = 'booking_complete'
X = df.drop(target, axis=1)
y = df[target]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
import matplotlib.pyplot as plt

feature_importance = model.feature_importances_
feature_names = X.columns
plt.figure(figsize=(10,6))
plt.barh(feature_names, feature_importance)
plt.xlabel("Importance")
plt.title("Feature Importance from RandomForest")
plt.show()
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, cv=5)
print("Cross-validation scores:", scores)
print("Mean CV accuracy:", scores.mean())
from sklearn.metrics import roc_auc_score

y_proba = model.predict_proba(X_test)[:,1]
roc_auc = roc_auc_score(y_test, y_proba)
print("ROC-AUC Score:", roc_auc)
import seaborn as sns
import matplotlib.pyplot as plt
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_test, y_proba)

plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, color='darkorange', label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend()
plt.show()
