import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, callbacks
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Function to normalize data
def scaling(S, S_max, S_min):
    return (S - S_min) / (S_max - S_min)

# Function to inverse normalize data
def inverse_scaling(S, S_max, S_min):
    return (S_max - S_min) * S + S_min

# Load data
S = np.load('Snapshots_two_params.npy')  # Snapshots matrix
Brb = np.load('Brb_two_params.npy').real  # Reduced POD basis
P = np.load('Parameters_two_params.npy')  # Parameter matrix with two columns

# Compute reduced solutions
Urb_POD = np.dot(Brb.T, S).T  # Transpose to get correct shape
print("Before scaling=",np.shape(Urb_POD))
# Shuffle data
shuffle = np.arange(len(Urb_POD))
np.random.shuffle(shuffle)
Urb_POD = Urb_POD[shuffle]
P = P[shuffle]

# Normalize parameters
P_max = np.max(P, axis=0)
P_min = np.min(P, axis=0)
P = P/P_max

# Normalize reduced solutions
Urb_POD_max = np.max(Urb_POD, axis=0)
Urb_POD_min = np.min(Urb_POD, axis=0)
Urb_POD = scaling(Urb_POD, Urb_POD_max, Urb_POD_min)
print("After scaling=",np.shape(Urb_POD))
# Split data into training and validation sets
P_train, P_val, Urb_POD_train, Urb_POD_val = train_test_split(P, Urb_POD, test_size=0.2, random_state=42)

# Define a neural network model with regularization and batch normalization
def create_model(input_shape):
    model = models.Sequential([
        layers.Dense(128, activation='relu', input_shape=input_shape, kernel_regularizer=regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        layers.Dense(Urb_POD.shape[1], activation='linear')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

# Create the model
input_shape = (P_train.shape[1],)
model = create_model(input_shape)

# Early stopping callback
early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = model.fit(P_train, Urb_POD_train, epochs=350, batch_size=32, validation_data=(P_val, Urb_POD_val), callbacks=[early_stopping])

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()

# Save the model
model.save('Neural-network-two-params.h5')

# Load the pre-trained neural network model
model = models.load_model('Neural-network-two-params.h5', compile=False)

# Recompile the model manually
model.compile(optimizer='adam', loss='mean_squared_error')

# Online phase: Predict the new solution
mu1 = float(input("New value for mu1: "))
mu2 = float(input("New value for mu2: "))
diffus = np.array([[mu1, mu2]])

# Normalize the input parameters
diffus_norm = diffus/P_max

# Predict the reduced basis coefficients
U_rb_pred_normalized = model.predict(diffus_norm).flatten()

# Inverse normalization (rescaling) of the output
U_rb_pred = inverse_scaling(U_rb_pred_normalized.copy(), Urb_POD_max, Urb_POD_min)

# Reconstruct the full-order (FE) solution from reduced prediction
uh_NN = np.dot(Brb, U_rb_pred) # Shape (N_h,)


# Load FE and POD solutions
uh_FE = np.load('Uh_FE.npy')
uh_POD = np.load('Uh_POD.npy')

#Compute the normalized L2 error 
uh_NN_norm = scaling(uh_NN, np.max(uh_NN), np.min(uh_NN))
uh_FE_norm = scaling(uh_FE, np.max(uh_FE), np.min(uh_FE))
uh_POD_norm = scaling(uh_POD, np.max(uh_POD), np.min(uh_POD))

l2_error_FE_norm = np.linalg.norm(uh_NN_norm - uh_FE_norm)
l2_error_POD_norm = np.linalg.norm(uh_NN_norm - uh_POD_norm)
print(f"L2 Error Finite Element Norm: {l2_error_FE_norm}")
print(f"L2 Error POD Norm: {l2_error_POD_norm}")


