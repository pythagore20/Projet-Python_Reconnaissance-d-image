# Paramètres du modèle (selon mes_fonctions.py)
MODEL_CONFIG = {
    'input_shape': (128, 128, 3),
    'dense_units': 256,
    'dropout_rate': 0.4,
    'target_size': (128, 128)
}

# Paramètres des générateurs de données (selon mes_fonctions.py)
GENERATOR_CONFIG = {
    'target_size': (64, 64),  # Comme dans leur code
    'batch_size': 16,
    'class_mode': 'binary'
}

# Chemins des données (à adapter selon votre structure)
DATA_PATHS = {
    'train_dir': './datasets/train',
    'validation_dir': './datasets/validation',
    'test_dir': './datasets/test'
}

# Classes
CLASS_NAMES = ['Chat', 'Chien']

# Seuil de confiance pour les prédictions
CONFIDENCE_THRESHOLD = 0.5
