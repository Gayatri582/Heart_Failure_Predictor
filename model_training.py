import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

print("✅ Step 1: Loading dataset...")

# Dataset load
try:
    df = pd.read_csv('heart_failure_clinical_records_dataset.csv')
    print("✅ Step 2: Dataset loaded successfully!")
except Exception as e:
    print("🚨 Error loading dataset:", e)
    exit()

print("✅ Step 3: Preparing data...")

X = df.drop('DEATH_EVENT', axis=1)
y = df['DEATH_EVENT']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("✅ Step 4: Training model...")

clf = RandomForestClassifier()
clf.fit(X_train, y_train)

print("✅ Step 5: Evaluating model...")

y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Accuracy: {accuracy * 100:.2f}%")

print("✅ Step 6: Saving model...")

with open('model.pkl', 'wb') as file:
    pickle.dump(clf, file)

print("🎉 Model training complete — model.pkl saved!")



