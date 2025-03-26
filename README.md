# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import necessary libraries (pandas, LabelEncoder, train_test_split, etc.).
2. Load the dataset using pd.read_csv().
3. Create a copy of the dataset and drop unnecessary columns (sl_no, salary).
4. Check for missing and duplicate values using isnull().sum() and duplicated().sum().
5. Encode categorical variables using LabelEncoder() to convert them into numerical values.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Divya R V
RegisterNumber: 212223100005 
*/
```
```
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load Dataset
data = pd.read_csv("/content/Placement_Data.csv")
print(data.head())

# Create a Copy and Drop Unwanted Columns
data1 = data.copy()
data1 = data1.drop(["sl_no", "salary"], axis=1)
print(data1.head())

# Check for Missing and Duplicate Values
print(data1.isnull().sum())  # Check for missing values
print("Duplicate Rows:", data1.duplicated().sum())  # Count duplicate rows

# Encode Categorical Variables
le = LabelEncoder()
categorical_columns = ["gender", "ssc_b", "hsc_b", "hsc_s", "degree_t", "workex", "specialisation", "status"]
for col in categorical_columns:
    data1[col] = le.fit_transform(data1[col])

print(data1.head())

# Define Features (X) and Target (y)
X = data1.iloc[:, :-1]
y = data1["status"]

# Split Dataset into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train Logistic Regression Model
lr = LogisticRegression(solver="liblinear")
lr.fit(X_train, y_train)

# Predict on Test Data
y_pred = lr.predict(X_test)
print("Predictions:", y_pred)

# Evaluate Model Performance
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

classification_report1 = classification_report(y_test, y_pred)
print("Classification Report:\n", classification_report1)

# Predict Placement for a New Student
new_student = [[1, 80, 1, 90, 1, 1, 90, 1, 0, 85, 1, 85]]
prediction = lr.predict(new_student)
print("Placement Prediction (0=Not Placed, 1=Placed):", prediction[0])

```

## Output:
![the Logistic Regression Model to Predict the Placement Status of Student](sam.png)

![Screenshot 2025-03-26 152105](https://github.com/user-attachments/assets/443683fc-900b-47f2-95be-4e280ca887da)


![Screenshot 2025-03-26 152124](https://github.com/user-attachments/assets/b4b27aba-5545-4a60-b623-d3c52202355a)







## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
