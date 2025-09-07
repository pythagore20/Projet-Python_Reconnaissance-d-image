import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

class CatDogGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Classificateur Chat vs Chien")
        self.root.geometry("600x400")
        
        # Bouton pour charger image
        self.load_btn = tk.Button(root, text="Charger Image", command=self.load_image)
        self.load_btn.pack(pady=20)
        
        # Label pour afficher résultat
        self.result_label = tk.Label(root, text="Résultat apparaîtra ici")
        self.result_label.pack(pady=20)
    
    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.png")])
        if file_path:
            self.result_label.config(text=f"Image chargée: {file_path}")

if __name__ == "__main__":
    root = tk.Tk()
    app = CatDogGUI(root)
    root.mainloop()
