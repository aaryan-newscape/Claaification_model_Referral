import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = 'C:\\Users\\Admin\\Desktop\\Prioritization_2\\referral.csv'
data = pd.read_csv(file_path)

# Display the first few rows of the dataset to verify it's loaded correctly
print("First few rows of the dataset:")
print(data.head())

# Check data types and null values
print("\nData types and null values:")
print(data.info())

# Alternatively, you can get the data types directly
print("\nData types of each column:")
print(data.dtypes)

# Check the unique values in the target variable (Referral Type)
print("\nUnique values in the target variable 'Referral Type':")
print(data['Referral Type'].unique())

# Analyze the distribution of the target variable
print("\nDistribution of 'Referral Type':")
print(data['Referral Type'].value_counts())

# EDA: Analyze the distribution of a numerical feature (Contract Rate)
print("\nVisualizing the distribution of 'Contract Rate':")
sns.histplot(data['Contract Rate'], kde=True)
plt.title('Distribution of Contract Rate')
plt.show()

# EDA: Analyze the distribution of a categorical feature (Client State)
print("\nVisualizing the distribution of 'Client State':")
sns.countplot(y='Client State', data=data, order = data['Client State'].value_counts().index)
plt.title('Distribution of Client State')
plt.show()

# EDA: Visualize the relationship between Contract Rate and Referral Type
print("\nVisualizing the relationship between 'Contract Rate' and 'Referral Type':")
sns.boxplot(x='Referral Type', y='Contract Rate', data=data)
plt.title('Contract Rate by Referral Type')
plt.show()

# EDA: Visualize the relationship between Client State and Referral Type
print("\nVisualizing the relationship between 'Client State' and 'Referral Type':")
sns.countplot(x='Referral Type', hue='Client State', data=data)
plt.title('Referral Type Distribution by Client State')
plt.show()

# Check for missing values in the dataset
print("\nChecking for missing values in the dataset:")
print(data.isnull().sum())

# Visualize missing data
print("\nVisualizing missing data heatmap:")
sns.heatmap(data.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Data Heatmap')
plt.show()

# Box plot to identify outliers in Contract Rate
print("\nVisualizing outliers in 'Contract Rate':")
sns.boxplot(data['Contract Rate'])
plt.title('Box Plot of Contract Rate')
plt.show()

# Distribution of Referral Type
print("\nVisualizing the distribution of 'Referral Type':")
sns.countplot(data['Referral Type'])
plt.title('Distribution of Referral Type')
plt.show()
