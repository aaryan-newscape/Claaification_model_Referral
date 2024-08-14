import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Load the Encoded Dataset
file_path = 'C:\\Users\\Admin\\Desktop\\Prioritization_2\\encoded_referral_data.xlsx'
data_encoded = pd.read_excel(file_path)

# Step 2: Define Features and Target
X = data_encoded.drop('Referral Type', axis=1)  # Features (everything except the target column)
y = data_encoded['Referral Type']  # Target (the Referral Type column)

# Step 3: Split the Data into Training and Test Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Step 4: Initialize the Model
model = RandomForestClassifier(random_state=42)

# Step 5: Define the Parameter Grid for Hyperparameter Tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

# Step 6: Initialize GridSearchCV with the Model
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

# Step 7: Perform the Grid Search to Tune Hyperparameters
grid_search.fit(X_train, y_train)

# Step 8: Get the Best Parameters and Train the Model with Them
best_params = grid_search.best_params_
print("Best parameters found: ", best_params)

# Re-train the model using the best parameters
model = RandomForestClassifier(**best_params, random_state=42)
model.fit(X_train, y_train)

# Step 9: Evaluate the Tuned Model
y_pred = model.predict(X_test)
print("Tuned Model Accuracy:", accuracy_score(y_test, y_pred))
print("Tuned Model Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Tuned Model Classification Report:\n", classification_report(y_test, y_pred))

# Step 10: Analyze Feature Importance
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
feature_names = X.columns

# Plot the feature importances
plt.figure(figsize=(12, 8))
plt.title("Feature Importances")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), feature_names[indices], rotation=90)
plt.xlim([-1, X.shape[1]])
plt.show()
