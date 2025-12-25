# Download the data from your GitHub repository
!wget https://raw.githubusercontent.com/yotam-biu/ps9/main/parkinsons.csv -O /content/parkinsons.csv
!wget https://raw.githubusercontent.com/yotam-biu/python_utils/main/lab_setup_do_not_edit.py -O /content/lab_setup_do_not_edit.py
import lab_setup_do_not_edit
import pandas as pd

df = pd.read_csv('/content/parkinsons.csv')
display(df.head())
display(df.info())
X = df[['MDVP:Fo(Hz)', 'MDVP:Jitter(%)']]
y = df['status']

print("Input features (X) shape:", X.shape)
print("Output feature (y) shape:", y.shape)
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Convert the scaled array back to a DataFrame for easier handling if needed
X = pd.DataFrame(X_scaled, columns=X.columns)

display(X.head())
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print("X_train shape:", X_train.shape)
print("X_val shape:", X_val.shape)
print("y_train shape:", y_train.shape)
print("y_val shape:", y_val.shape)
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=3)
print("Model selected: K-Nearest Neighbors with n_neighbors=3")
from sklearn.metrics import accuracy_score

# Train the model
model.fit(X_train, y_train)

# Make predictions on the validation set
y_pred = model.predict(X_val)

# Calculate accuracy
accuracy = accuracy_score(y_val, y_pred)

print(f"Model Accuracy: {accuracy:.4f}")

if accuracy >= 0.8:
    print("Accuracy is at least 0.8. Good job!")
else:
    print("Accuracy is below 0.8. You might need to adjust the model or features.")
