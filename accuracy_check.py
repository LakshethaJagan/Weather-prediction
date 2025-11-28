import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# --- Load dataset ---
df = pd.read_csv("dataset.csv")

# Select a few relevant columns
df = df[['MinTemp', 'MaxTemp', 'Humidity3pm', 'Pressure9am', 'RainTomorrow']]
df.dropna(inplace=True)

# Encode target column
df['RainTomorrow'] = df['RainTomorrow'].map({'Yes':1, 'No':0})

# Split into features and target
X = df.drop('RainTomorrow', axis=1)
y = df['RainTomorrow']

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print(f"✅ Model Accuracy: {accuracy * 100:.2f}%")
