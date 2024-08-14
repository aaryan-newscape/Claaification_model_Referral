import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import numpy as np

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

# Step 5: Scale the Features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 6: Initialize the Logistic Regression Model
log_reg_model = LogisticRegression(random_state=42, multi_class='multinomial', solver='lbfgs', max_iter=1000)

# Step 7: Train the Logistic Regression Model
log_reg_model.fit(X_train, y_train)

# Step 8: Evaluate the Logistic Regression Model
y_pred = log_reg_model.predict(X_test)
print("Logistic Regression Model Accuracy:", accuracy_score(y_test, y_pred))
print("Logistic Regression Model Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Logistic Regression Model Classification Report:\n", classification_report(y_test, y_pred))

# Step 9: (Optional) Hyperparameter Tuning with GridSearchCV
param_grid = {
    'C': [0.1, 1, 10, 100],
    'penalty': ['l2'],
    'solver': ['lbfgs', 'saga'],
    'max_iter': [1000, 2000]
}

grid_search = GridSearchCV(estimator=log_reg_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Step 10: Get the Best Parameters and Train the Model with Them
best_params = grid_search.best_params_
print("Best parameters found: ", best_params)

# Re-train the model using the best parameters
log_reg_model = LogisticRegression(**best_params, random_state=42, multi_class='multinomial')
log_reg_model.fit(X_train, y_train)

# Step 11: Evaluate the Tuned Logistic Regression Model
y_pred_tuned = log_reg_model.predict(X_test)
print("Tuned Logistic Regression Model Accuracy:", accuracy_score(y_test, y_pred_tuned))
print("Tuned Logistic Regression Model Confusion Matrix:\n", confusion_matrix(y_test, y_pred_tuned))
print("Tuned Logistic Regression Model Classification Report:\n", classification_report(y_test, y_pred_tuned))
