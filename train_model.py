import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib

# Load the dataset
data = pd.read_csv("heart.csv")

# Define features and target
X = data.drop(columns='output')
y = data['output']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the Logistic Regression model with cross-validation
model = LogisticRegression()
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')  # 5-fold cross-validation

# Fit the model on the entire training set
model.fit(X_train, y_train)

# Evaluate on the test set
y_pred = model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)

# Display cross-validation and test accuracy
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean cross-validation accuracy: {cv_scores.mean():.2f}")
print(f"Test accuracy: {test_accuracy:.2f}")

# Save the model and scaler
joblib.dump(model, "heart_model.joblib")
joblib.dump(scaler, "scaler.joblib")
