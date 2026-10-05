import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout

# -------------------------------
# STEP 1: Generate training data
# -------------------------------

np.random.seed(42)

N = 5000

age = np.random.randint(18, 81, N)

smoker = np.random.randint(0, 2, N)

smoking_years = np.where(
    smoker == 1,
    np.random.randint(1, 41, N),
    0
)

cigarettes = np.where(
    smoker == 1,
    np.random.randint(1, 41, N),
    0
)

cough = np.random.randint(0, 2, N)

breathing_problem = np.random.randint(0, 2, N)

chest_pain = np.random.randint(0, 2, N)


# -------------------------------
# STEP 2: Create risk score
# -------------------------------

score = (
    age * 0.15
    + smoker * 20
    + smoking_years * 1.5
    + cigarettes * 1.0
    + cough * 8
    + breathing_problem * 12
    + chest_pain * 10
)

score += np.random.normal(0, 5, N)


# -------------------------------
# STEP 3: Create risk classes
# -------------------------------

# 0 = Low
# 1 = Moderate
# 2 = High

risk = np.where(
    score < 25,
    0,
    np.where(score < 55, 1, 2)
)


# -------------------------------
# STEP 4: Prepare input data
# -------------------------------

X = np.column_stack([
    age,
    smoker,
    smoking_years,
    cigarettes,
    cough,
    breathing_problem,
    chest_pain
])

y = risk


# -------------------------------
# STEP 5: Train/Test split
# -------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# -------------------------------
# STEP 6: Scaling
# -------------------------------

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)


# -------------------------------
# STEP 7: Deep Learning Model
# -------------------------------

model = Sequential([

    Dense(
        64,
        activation="relu",
        input_shape=(7,)
    ),

    Dropout(0.2),

    Dense(
        32,
        activation="relu"
    ),

    Dropout(0.2),

    Dense(
        16,
        activation="relu"
    ),

    Dense(
        3,
        activation="softmax"
    )
])


# -------------------------------
# STEP 8: Compile
# -------------------------------

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# -------------------------------
# STEP 9: Train
# -------------------------------

print("\nTraining Deep Learning Model...\n")

model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)


# -------------------------------
# STEP 10: Test model
# -------------------------------

loss, accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print("\n==============================")
print("MODEL TRAINING COMPLETE")
print("==============================")

print(
    f"Test Accuracy: {accuracy * 100:.2f}%"
)


# -------------------------------
# STEP 11: Save model
# -------------------------------

os.makedirs("model", exist_ok=True)

model.save(
    "model/smoking_risk_model.keras"
)

np.save(
    "model/scaler_mean.npy",
    scaler.mean_
)

np.save(
    "model/scaler_scale.npy",
    scaler.scale_
)

print("\nModel saved successfully!")
print(
    "model/smoking_risk_model.keras"
)