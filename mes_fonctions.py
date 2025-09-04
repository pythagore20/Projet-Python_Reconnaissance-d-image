
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def creation_generators(train_dir, validation_dir, target_size=(64,4), batch_size=16, class_mode='binary', augment=True):
    
    if augment:
        train_datagen = ImageDataGenerator(
            rescale=1/255.0,               # normaliser les pixels entre 0 et 1
            rotation_range=40,             # va faire une rotation aléatoire de l'image entre 0 et 40 degrés
            width_shift_range=0.2,         # va déplacer l'objet de 20% de sa largeur (peut être négatif)
            height_shift_range=0.2,        # va déplacer l'objet de 20% de sa hauteur
            shear_range=0.2,               # va cisailler l'image de 20%
            zoom_range=0.2,                # zoomer dans l'image entre 0 et 0.2
            horizontal_flip=True,          # va faire une symétrie horizontale (image miroir)
            fill_mode='nearest'            # comment remplir les pixels perdus après transformation
        )
    else:
        
        train_datagen = ImageDataGenerator(rescale=1./255)
    
    
    validation_datagen = ImageDataGenerator(rescale=1./255)
    
    
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode=class_mode
    )
    
    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode=class_mode
    )
    
    return train_generator, validation_generator

  




def model_cnn(input_shape=(128,128,3), dense_units=256, dropout_rate=0.4):
    
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((2,2)),
        
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2,2)),
        
        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2,2)),
        
        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2,2)),
        
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(dense_units, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    return model




