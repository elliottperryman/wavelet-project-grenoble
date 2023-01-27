from tensorflow.keras.models import load_model
from tensorflow.keras.utils import image_dataset_from_directory
model = load_model('./checkpoints/', compile=False)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
val = image_dataset_from_directory('BarkVN-50/BarkVN-50_mendeley/', label_mode='categorical', seed=0, subset='validation', validation_split=0.2)

print(model.evaluate(val))
