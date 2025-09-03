import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import image as mpimg

# chemins des dossiers
train_cats_dir = "cats_and_dogs_filtered/train/cats"
train_dogs_dir = "cats_and_dogs_filtered/train/dogs"
valid_cats_dir = "cats_and_dogs_filtered/validation/cats"
valid_dogs_dir = "cats_and_dogs_filtered/validation/dogs"

# affichage des données d'entraînement
train_set_cats = os.listdir(train_cats_dir)
train_set_dogs = os.listdir(train_dogs_dir)

print("Quelques chats:", train_set_cats[:5])
print("Quelques chiens:", train_set_dogs[:5])

# affichage des données de validation
valid_set_cats = os.listdir(valid_cats_dir)
valid_set_dogs = os.listdir(valid_dogs_dir)

print("Validation chats:", valid_set_cats[:5])
print("Validation chiens:", valid_set_dogs[:5])
print("\n")

# dimensions des données
print("Total entraînement cats:", len(train_set_cats))
print("Total entraînement dogs:", len(train_set_dogs))
print("****************\n")
print("Total validation cats:", len(valid_set_cats))
print("Total validation dogs:", len(valid_set_dogs))

# choix aléatoires d'entraînement (chiens et chats)
choice_dogs = np.random.choice(train_set_dogs, size=12, replace=False)
choice_cats = np.random.choice(train_set_cats, size=12, replace=False)

# affichage des chiens
plt.figure(figsize=(12, 8))
for i, dog in enumerate(choice_dogs):
    ax = plt.subplot(3, 4, i+1)
    img_path = os.path.join(train_dogs_dir, dog)
    img = mpimg.imread(img_path)
    plt.imshow(img)
    plt.axis("off")
    ax.set_title(f"Dog: {dog}")
plt.suptitle("Chiens (train set)", fontsize=16)
plt.tight_layout()
plt.savefig("echantillon_chiens.png")  # <<<<< fichier sauvegardé
plt.show()

# affichage des chats
plt.figure(figsize=(12, 8))
for i, cat in enumerate(choice_cats):
    ax = plt.subplot(3, 4, i+1)
    img_path = os.path.join(train_cats_dir, cat)
    img = mpimg.imread(img_path)
    plt.imshow(img)
    plt.axis("off")
    ax.set_title(f"Cat: {cat}")
plt.suptitle("Chats (train set)", fontsize=16)
plt.tight_layout()
plt.savefig("echantillon_chats.png")  # <<<<< fichier sauvegardé
plt.show()
#