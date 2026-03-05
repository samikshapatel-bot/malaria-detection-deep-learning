from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Dataset path
train_path = "New Plant Diseases Dataset/New Plant Diseases Dataset(Augmented)/Dataset/train"

# Image augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2   # 80% train , 20% validation
)

# Training dataset
train_data = train_datagen.flow_from_directory(
    train_path,
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

# Validation dataset
val_data = train_datagen.flow_from_directory(
    train_path,
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

print("Training images:", train_data.samples)
print("Validation images:", val_data.samples)