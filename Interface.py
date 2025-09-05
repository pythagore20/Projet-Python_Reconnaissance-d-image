import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image


#model_path = r"C:\Users\PC\OneDrive\Desktop\IA reconnaisssance\Projet-Python_Reconnaissance-d-image\model3.keras"
#model = tf.keras.models.load_model(model_path)
#model = keras.models.load_model("model3.keras")
model = keras.models.load_model(r"C:\Users\PC\OneDrive\Desktop\IA reconnaisssance\Projet-Python_Reconnaissance-d-image\model3.keras")


def open_image():
    file_path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")]
    )
    if not file_path:
        return

    
    img = Image.open(file_path)
    img_resized = img.resize((200, 200))
    img_tk = ImageTk.PhotoImage(img_resized, master=root)
    img_label.config(image=img_tk)
    img_label.image = img_tk

    # Préparer l'image pour la prédiction
    x = image.img_to_array(img.resize((150, 150))) / 255.0
    x = np.expand_dims(x, axis=0)

    prediction = model.predict(x)
    pred_label = "L'animal sur la photo est un chien" if prediction[0][0] >= 0.5 else "L'animal sur la phtoto est un chat"

    result_label.config(text=f" {pred_label} ({prediction[0][0]:.4f})")


root = tk.Tk()
root.title("IA Reconnaissance Chat/Chien")
root.geometry("600x500")
root.iconbitmap(r"C:\Users\PC\OneDrive\Desktop\IA reconnaisssance\Projet-Python_Reconnaissance-d-image\icone.ico")
root.configure(bg="#D8E8E9")
frame = tk.Frame(root, bg="white", bd=5, relief="ridge")
frame.pack(pady=10, padx=20, fill="both", expand=True)

exp= tk.Label(frame, text="Bienvenue sur l'application de reconnaissance d'image", 
                 font=("Arial", 20, "bold"), bg="white", fg="#2d3436")
exp.pack(pady=20)

img_label = tk.Label(frame,bg="#D8E8E9")
img_label.pack(pady=10)

description_frame = tk.Frame(frame, bg="white", bd=3, relief="ridge")
description_frame.pack(pady=15, padx=20, fill="x")

result_label = tk.Label(frame, 
                        text="Une application qui vous permet de passer des images d'animaux et elle déterminera si ce sont des chiens ou des chats.\n\nElle est basée sur le CNN.",
                        font=("Arial", 12), bg="white", fg="black", wraplength=500, justify="center")
result_label.pack(pady=10, padx=10)

btn_frame = tk.Frame(root, bg="white")
btn_frame.pack(pady=20)

btn_select = tk.Button(btn_frame, text="📂 Choisir une image", command=open_image,
                       font=("Arial", 14), bg="#74b9ff", fg="white", width=18, relief="raised")
btn_select.grid(row=0, column=0, padx=10)

btn_clear = tk.Button(btn_frame, text="❌ Effacer", command=lambda: [img_label.config(image=""), result_label.config(text="")],
                      font=("Arial", 14), bg="#ff7675", fg="white", width=12)
btn_clear.grid(row=0, column=1, padx=10)

btn_quit = tk.Button(btn_frame, text="🚪 Quitter", command=root.quit,
                     font=("Arial", 14), bg="#636e72", fg="white", width=12)
btn_quit.grid(row=0, column=2, padx=10)

root.mainloop()
