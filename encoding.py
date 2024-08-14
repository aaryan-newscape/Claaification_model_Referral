
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the dataset
file_path = 'C:\\Users\\Admin\\Desktop\\Prioritization_2\\referral.csv'
data = pd.read_csv(file_path)

# One-hot encode categorical variables
data_encoded = pd.get_dummies(data, columns=['Client State', 'Type of Service', 'Assigned Office Location', 'Payor Organization', 'Billing Method'])

# Feature Scaling: Scale the 'Contract Rate' column
scaler = StandardScaler()
data_encoded['Contract Rate'] = scaler.fit_transform(data_encoded[['Contract Rate']])

# Save the encoded dataset to an Excel file in a valid directory
output_file_path = 'C:\\Users\\Admin\\Desktop\\Prioritization_2\\encoded_referral_data.xlsx'  # Update with a valid path
data_encoded.to_excel(output_file_path, index=False)

print(f"Encoded data has been saved to {output_file_path}")
