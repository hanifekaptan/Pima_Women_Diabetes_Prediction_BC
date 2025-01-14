import pandas as pd
import pickle
from keras.api.models import Sequential
from keras.api.layers import Dense, Dropout, BatchNormalization
from sklearn.preprocessing import StandardScaler

# Load the data
data = pd.read_csv("diabetes.csv")
data = data[data.BMI >= 5]  # Filter out data with BMI less than 5

X = data.drop("Outcome", axis=1)  # Features (independent variables)
y = data["Outcome"]  # Target variable (dependent variable)

# Preprocessing step
sc = StandardScaler()  # Initialize the StandardScaler
X = sc.fit_transform(X)  # Fit and transform the feature data

# Create the model
model = Sequential()  # Initialize a Sequential model
model.add(Dense(120, activation="relu", input_shape=(X.shape[1],)))  # Add the input layer with ReLU activation
model.add(BatchNormalization())  # Add Batch Normalization layer
model.add(Dropout(0.3))  # Add Dropout layer with a rate of 0.3
model.add(Dense(64, activation="relu"))  # Add a hidden layer with ReLU activation
model.add(BatchNormalization())  # Add Batch Normalization layer
model.add(Dropout(0.3))  # Add Dropout layer with a rate of 0.3
model.add(Dense(1, activation="sigmoid"))  # Add the output layer with sigmoid activation
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])  # Compile the model

# Train the model
model.fit(X, y,
          epochs=10,  # Number of epochs
          batch_size=32,  # Size of each batch
          validation_split=0.2,  # Split 20% of data for validation
          verbose=1)  # Verbosity mode

model.save("model.keras")  # Save the model
joblib.dump(sc, "diabetes_scaler.pkl")  # Save the scaler
