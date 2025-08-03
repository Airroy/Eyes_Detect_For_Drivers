from tensorflow.keras.models import load_model

model = load_model("my_model.keras")
print("Input shape:", model.input_shape)
print("Model summary:")
model.summary()
