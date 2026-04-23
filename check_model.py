import tensorflow as tf

# Load your model
model = tf.keras.models.load_model('model.h5')

# Print the number of categories the model outputs
num_classes = model.output_shape[-1]
print(f"Success! Your model is trained to recognize {num_classes} different categories.")