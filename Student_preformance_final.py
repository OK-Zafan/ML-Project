# Import necessary libraries
import pandas as pd  # For data manipulation and analysis
import matplotlib.pyplot as plt  # For data visualization
import seaborn as sns  # For advanced data visualization
from sklearn.preprocessing import StandardScaler # For features scalling
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
student_mat = pd.read_csv("student-mat.csv", sep=";")  # Load math performance dataset
student_por = pd.read_csv("student-por.csv", sep=";")  # Load Portuguese performance dataset

# Standardize formatting for shared columns
merged_columns = ["school", "sex", "age", "address", "famsize", "Pstatus", "Medu", "Fedu", "Mjob", "Fjob", "reason", "nursery", "internet"]
for col in merged_columns:
    if student_mat[col].dtype == 'object':
        # Remove leading/trailing whitespace and convert to lowercase for consistency
        student_mat[col] = student_mat[col].str.strip().str.lower()
        student_por[col] = student_por[col].str.strip().str.lower()
    elif student_mat[col].dtype in ['int64', 'float64']:
        # Ensure numeric columns have consistent data types
        student_mat[col] = student_mat[col].astype(float)
        student_por[col] = student_por[col].astype(float)

# Remove duplicates in the datasets
student_mat = student_mat.drop_duplicates(subset=merged_columns)  # Ensure no duplicate rows based on key columns
student_por = student_por.drop_duplicates(subset=merged_columns)

# Determine the number of students in each dataset
total_students_math = student_mat.shape[0]
total_students_port = student_por.shape[0]
print(f"Number of students in Math: {total_students_math}")
print(f"Number of students in Portuguese: {total_students_port}")

# Identify common students based on shared attributes
# Suffixes differentiate overlapping column names in merged datasets
common_students = pd.merge(student_mat, student_por, on=merged_columns, suffixes=("_math", "_port"))

# Include all other features for common students
for col in student_mat.columns:
    if col not in merged_columns + ["G1", "G2", "G3"]:
        # Use combine_first to prefer non-null values from math dataset, fallback to Portuguese dataset
        common_students[col] = common_students[col + "_math"].combine_first(common_students[col + "_port"])

# Calculate the averages for G1, G2, and G3 for common students
common_students["G1"] = (common_students["G1_math"] + common_students["G1_port"]) / 2
common_students["G2"] = (common_students["G2_math"] + common_students["G2_port"]) / 2
common_students["G3"] = (common_students["G3_math"] + common_students["G3_port"]) / 2
common_students = common_students[merged_columns + list(student_mat.columns.difference(merged_columns + ["G1", "G2", "G3"])) + ["G1", "G2", "G3"]]

# Print number of shared students
print(f"Number of shared students: {common_students.shape[0]}")

# Find unique students in Math
unique_math = pd.merge(
    student_mat,
    common_students[merged_columns],
    on=merged_columns,
    how="left",  # Perform a left join to retain all rows from the math dataset
    indicator=True  # Add a column indicating whether a row exists only in left dataset or both
).query("_merge == 'left_only'").drop(columns=["_merge"])

# Find unique students in Portuguese
unique_port = pd.merge(
    student_por,
    common_students[merged_columns],
    on=merged_columns,
    how="left",  # Perform a left join to retain all rows from the Portuguese dataset
    indicator=True  # Add a column indicating whether a row exists only in left dataset or both
).query("_merge == 'left_only'").drop(columns=["_merge"])

# Validate the unique student counts
print(f"Number of unique students in Math: {unique_math.shape[0]}")
print(f"Number of unique students in Portuguese: {unique_port.shape[0]}")

# Combine the datasets
combined_student_data = pd.concat([common_students, unique_math, unique_port], ignore_index=True)

# Display the final number of students in the combined dataset
print(f"Total students in the combined dataset: {combined_student_data.shape[0]}")


# Inspect the dataset
# Display the first few rows of the dataset to understand its structure
print("First 5 rows of the dataset:")
print(combined_student_data.head())

# Check the general information about the dataset, including column data types
print("\nDataset Information:")
print (combined_student_data.info())

# Summary statistics for numerical columns
print("\nSummary Statistics for Numerical Columns:")
print(combined_student_data.describe())


# Check for missing values
missing_values = combined_student_data.isnull().sum()
print("\nMissing Values in Each Column:")
print(missing_values)
print("- No missing values were detected, which simplifies preprocessing.")

# Visualize the distribution of the target variable (G3 - Final Grades)
# to help understanding how grades are spread and identify outliers that might affect model accuracy
plt.figure(figsize=(6, 4))
sns.histplot(combined_student_data['G3'], kde=True, bins=20, color='blue')
plt.title('Distribution of Final Grades (G3)')
plt.xlabel('Final Grade (G3)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Correlation heatmap for numerical features
# visualize the relationships between numerical features in the dataset. 
#Features with a high correlation (negative or postive) with the target variable (like G3) are likely important for predictions.
#Identify features that are highly correlated (e.g., G1 and G2) to choose only one of them to avoid redandncy and avoid overfitting
#visual view can lead to discover any unexpected correlation 
plt.figure(figsize=(9, 6))
sns.heatmap(combined_student_data.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()

'''
# Highlight conclusion from the initial heatmap
# Observation: G1 and G2 have a strong correlation with each other 
# indicating potential multicollinearity.
# To address this, we will combine G1 and G2 into a single feature (G1_G2_avg).
'''
# Combine G1 and G2 into a single feature to avoid multicollinearity
# G1 and G2 represent the first two grades for each student and are highly correlated, which could cause multicollinearity.
# Combining them ensures we reduce redundancy while preserving their overall impact.
combined_student_data['G1_G2_avg'] = (combined_student_data['G1'] + combined_student_data['G2']) / 2
combined_student_data = combined_student_data.drop(columns=['G1', 'G2'])


# Correlation heatmap after combining G1 and G2
# This will confirm that the multicollinearity issue is resolved.
plt.figure(figsize=(9, 6))
sns.heatmap(combined_student_data.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap After Combining G1 and G2')
plt.show()
# Preprocessing Non-Numeric Features
# Identify non-numeric (categorical) columns
non_numeric_columns = combined_student_data.select_dtypes(include=['object']).columns.tolist()
print("\nNon-numeric columns:", non_numeric_columns)

# Display unique values for non-numeric features which will be encoded
for col in non_numeric_columns:
    print(f"\nUnique values in {col}:")
    print(combined_student_data[col].unique())

# Encode categorical features 
# This converts non-numeric columns into numeric, creating separate columns for each category
encoded_data = pd.get_dummies(combined_student_data, columns=non_numeric_columns, drop_first=True)
# Copy the encoded data
encoded_data_numeric = encoded_data.copy()
# Convert boolean columns to integers (0/1, instead of false/true) for better performance
for col in encoded_data_numeric.select_dtypes(include=['bool']).columns:
    encoded_data_numeric[col] = encoded_data_numeric[col].astype(int)
# Display the shape and a preview of the dataset after encoding
print("\nShape of the dataset after encoding:", encoded_data_numeric.shape)
print(encoded_data_numeric.info())


#following will analyse features changed/added after encoding
# Identify original categorical columns
categorical_columns = combined_student_data.select_dtypes(include=['object']).columns.tolist()
# Identify new columns after encoding
encoded_columns = encoded_data_numeric.columns
#Map old names to new names
changed_features_mapping = []
for original in categorical_columns:
    new_features = [col for col in encoded_columns if col.startswith(original + '_')]
    if new_features:  # Only consider features that were changed
        for new_feature in new_features:
            changed_features_mapping.append((original, new_feature))
# Convert mapping to a DataFrame for easy display
changed_features_df = pd.DataFrame(changed_features_mapping, columns=["Old Feature Name", "New Feature Name"])
# Display the DataFrame
print ('Fetures maping after encoding:\n' , changed_features_df)
print ('Encoding result analsys : \nAll non-numeric features that has 2 values only mapped to one new column only')
print ('but if the feature has more than 2 value mapped to #features = #values-1 for this feature (as we use drop_first=True)')


# Select only encoded features and the target variable 'G3'
encoded_features_and_target = encoded_data_numeric[[col for col in encoded_data_numeric.columns if col.startswith(tuple(categorical_columns))] + ['G3']]
# Correlation heatmap for encoded features and G3
plt.figure(figsize=(14, 10))
sns.heatmap(encoded_features_and_target.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap (Encoded Features + G3)')
plt.show()


# remove rows with Outlier values in Numerical Features
# Define a function to detect outliers using the IQR method
def detect_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)  # First quartile value which has 25% data below it
    Q3 = data[column].quantile(0.75)  # Third quartile value which has 75% data below it
    IQR = Q3 - Q1  # Interquartile range contain 50% midle data (75% -25%)
    lower_bound = Q1 - 1.5 * IQR # detect points that are exceptionally far below the middle 50% range
    upper_bound = Q3 + 1.5 * IQR # detect points that are exceptionally far above the middle 50% range
    return (data[column] < lower_bound) | (data[column] > upper_bound) # return values to be removed

# Apply outlier detection and removal for key numerical features
numeric_columns = encoded_data_numeric.select_dtypes(include=['int64', 'float64']).columns

print("Numeric columns:", numeric_columns)
totalOut =0
for column in numeric_columns:
    outliers = detect_outliers_iqr(encoded_data_numeric, column)
    print(f"Number of outliers in '{column}': {outliers.sum()}")
    totalOut += outliers.sum()
    # Remove outliers
    encoded_data_numeric = encoded_data_numeric[~outliers]

# Reset index after removing outliers
encoded_data_numeric = encoded_data_numeric.reset_index(drop=True)
print(f"\nShape of dataset after outlier removal: {encoded_data_numeric.shape}")
print(totalOut," row was removed due to outlier")

# Scale Numerical Features (reduce the range of values for each numeric col)
# Initialize the scaler
scaler = StandardScaler()
# Scale only numerical features
scaled_features = scaler.fit_transform(encoded_data_numeric[numeric_columns])

# Replace original numerical columns with scaled values
scaled_data = encoded_data_numeric.copy()
scaled_data[numeric_columns] = scaled_features
print("\nScaled Data Preview:")
print(scaled_data.head())


# Separate features and target variable (G3)
X = scaled_data.drop(columns=['G3'])  # Features
y = scaled_data['G3']  # Target variable

print(f"\nFeature Matrix (X) Shape: {X.shape}")
print(f"Target Vector (y) Shape: {y.shape}") 


# 2nd step Split the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train the Model
# Initialize the Linear Regression model
model = LinearRegression()

# Train the model using the training data
model.fit(X_train, y_train)

# Step 4: Evaluate the Model
# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate performance metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Model Evaluation Results:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R2): {r2:.2f}")

# Step 6: Visualize Predictions vs. Actual Values
plt.figure(figsize=(6, 4))
plt.scatter(y_test, y_pred, alpha=0.6, color='blue', label='Predicted Values')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--', color='red', label='Perfect Prediction Line')
plt.title('Actual vs Predicted Final Grades')
plt.xlabel('Actual Grades (G3)')
plt.ylabel('Predicted Grades (G3)')
plt.legend()
plt.grid(True)
plt.show()
# From the view above, it is clear that normal linear regression may have 
# limitations if trends are non-linear, but from the result it is enough here.

# Summary Report
print("\n--- Project Summary ---")
print("1. Data Preprocessing: Completed")
print("2. Model Training: Linear Regression")
print("3. Evaluation Metrics:")
print(f"   - Mean Squared Error (MSE): {mse:.2f}")
print(f"   - R-squared (R2): {r2:.2f}")
print("4. Visualization:\nScatter plot for Actual vs Predicted Grades displayed, The blue points represent the predicted G3 plotted against the actual values, these points aligen closely with the red dashed so model predictions are accurate.")



