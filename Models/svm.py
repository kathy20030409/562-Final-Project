import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv('heart_2020_cleaned.csv')
df['HeartDisease'] = df['HeartDisease'].replace({'Yes': 1, 'No': 0})

# Identify categorical variables (assuming they are of type 'object')
categorical_cols = df.columns[df.dtypes == object].tolist()

# Convert categorical variables using one-hot encoding
df = pd.get_dummies(df, columns=categorical_cols)

# Display the first few rows of the DataFrame to verify changes
print(df.head())

# Select features and target
X = df.drop('HeartDisease', axis=1)  # all columns except 'HeartDisease'
y = df['HeartDisease']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create and evaluate SVM models as before
# SVM with a linear kernel
svm_linear = SVC(kernel='linear', C=1.0, random_state=42)
svm_linear.fit(X_train, y_train)
y_pred_linear = svm_linear.predict(X_test)
print("Accuracy with Linear Kernel:", accuracy_score(y_test, y_pred_linear))

# SVM with a polynomial kernel
svm_poly = SVC(kernel='poly', degree=3, C=1.0, random_state=42)
svm_poly.fit(X_train, y_train)
y_pred_poly = svm_poly.predict(X_test)
print("Accuracy with Polynomial Kernel:", accuracy_score(y_test, y_pred_poly))

# SVM with an RBF kernel
svm_rbf = SVC(kernel='rbf', gamma='scale', C=1.0, random_state=42)
svm_rbf.fit(X_train, y_train)
y_pred_rbf = svm_rbf.predict(X_test)
print("Accuracy with RBF Kernel:", accuracy_score(y_test, y_pred_rbf))

# SVM with a sigmoid kernel
svm_sigmoid = SVC(kernel='sigmoid', gamma='scale', C=1.0, random_state=42)
svm_sigmoid.fit(X_train, y_train)
y_pred_sigmoid = svm_sigmoid.predict(X_test)
print("Accuracy with Sigmoid Kernel:", accuracy_score(y_test, y_pred_sigmoid))

