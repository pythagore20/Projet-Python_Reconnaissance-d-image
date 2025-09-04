from mes_fonctions import *
def train_cnn_model(train_dir, validation_dir,
                    input_shape=(128,128,3),
                    dense_units=256,
                    dropout_rate=0.4,
                    target_size=(128,128),
                    batch_size=20,
                    epochs=10,
                    class_mode='binary',
                    augment=True):
    
    train_gen, val_gen = creation_generators(
        train_dir, validation_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode=class_mode,
        augment=augment
    )
    
    # Modèle
    model = model_cnn(input_shape=input_shape, dense_units=dense_units, dropout_rate=dropout_rate)
    
    # Compilation
    model.compile(optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001),loss='binary_crossentropy',metrics = ['acc'])
    
    # Entraînement
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        steps_per_epoch=100,
        validation_steps=50 
    )
    
    return model, history
