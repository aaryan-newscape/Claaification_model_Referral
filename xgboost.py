import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

# Step 1: Load the Encoded Dataset
file_path = 'C:\\Users\\Admin\\Desktop\\Prioritization_2\\encoded_referral_data.xlsx'
data_encoded = pd.read_excel(file_path)

# Step 2: Define Features and Target
X = data_encoded.drop('Referral Type', axis=1)  # Features (everything except the target column)
y = data_encoded['Referral Type']  # Target (the Referral Type column)

# Step 3: Encode the Target Labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Step 4: Split the Data into Training and Test Sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# Step 5: Initialize the XGBoost Model
xgb_model = XGBClassifier(random_state=42)

# Step 6: Train the XGBoost Model
xgb_model.fit(X_train, y_train)

# Step 7: Evaluate the XGBoost Model
y_pred = xgb_model.predict(X_test)
print("XGBoost Model Accuracy:", accuracy_score(y_test, y_pred))
print("XGBoost Model Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("XGBoost Model Classification Report:\n", classification_report(y_test, y_pred))

# Step 8: Hyperparameter Tuning with GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 6, 9],
    'learning_rate': [0.01, 0.1, 0.3]
}

grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Step 9: Get the Best Parameters and Train the Model with Them
best_params = grid_search.best_params_
print("Best parameters found: ", best_params)

# Re-train the model using the best parameters
xgb_model = XGBClassifier(**best_params, random_state=42)
xgb_model.fit(X_train, y_train)

# Step 10: Evaluate the Tuned XGBoost Model
y_pred_tuned = xgb_model.predict(X_test)
print("Tuned XGBoost Model Accuracy:", accuracy_score(y_test, y_pred_tuned))
print("Tuned XGBoost Model Confusion Matrix:\n", confusion_matrix(y_test, y_pred_tuned))
print("Tuned XGBoost Model Classification Report:\n", classification_report(y_test, y_pred_tuned))

# Step 11: Analyze Feature Importance
importances = xgb_model.feature_importances_
indices = np.argsort(importances)[::-1]
feature_names = X.columns

# Plot the feature importances
plt.figure(figsize=(12, 8))
plt.title("XGBoost Feature Importances")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), feature_names[indices], rotation=90)
plt.xlim([-1, X.shape[1]])
plt.show()
