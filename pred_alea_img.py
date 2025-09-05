import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image

def predict_img(model, train_generator, train, target_size=(180,180)):

    doss_choix = np.random.choice(os.listdir(train))
    img_nom = np.random.choice(os.listdir(os.path.join(train, doss_choix)))
    img_path = os.path.join(train, doss_choix, img_nom)
    
    
    img = image.load_img(img_path, target_size=target_size)
    x = image.img_to_array(img) / 255.0
    x = np.expand_dims(x, axis=0)
    

    pred = model.predict(x)
    
    class_indices = train_generator.class_indices
    idx_to_class = {i: j for j, i in class_indices.items()}  
    pred_class = 0 if pred[0][0] < 0.5 else 1
    pred_label = idx_to_class[pred_class]
    
   
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"Prédit : {pred_label}")
    plt.show()
    print(f"Valeur du modèle: {pred[0][0]:.4f}")
    print(f"L'animal sur l'image est un : {pred_label}")
    
    return pred_label
