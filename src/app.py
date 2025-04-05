import re
import sys
import torch
import numpy as np
import tkinter as tk
import torch.nn as nn
import torchvision.transforms as T

from tkinter import filedialog
from PIL import Image, ImageTk
from tifffile import imread
from transformers import SegformerForSemanticSegmentation

class PrintLogger(object):
    def __init__(self, textbox):
        self.textbox = textbox
        self.ansi_escape = re.compile(r'(?:\x1B[@-_]|[\x08\x0A\x0D])[\x30-\x7E]*')

    def write(self, text):
        cleaned_text = self.ansi_escape.sub('', text)
        if cleaned_text.strip():
            self.textbox.configure(state="normal")
            if not cleaned_text.endswith('\n'):
                cleaned_text += '\n'
            self.textbox.insert("end", cleaned_text)
            self.textbox.see("end")
            self.textbox.configure(state="disabled")

    def flush(self):
        pass

class ImageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Fire Detection")
        self.root.geometry("810x375")
        self.root.resizable(False, False)
        
        self.input = None
        self.setup_gui()
        self.redirect_logging()

    def setup_gui(self):
        self.label_input = tk.Label(self.root, text="Input Image", font=("Arial", 10, "bold"))
        self.label_input.grid(row=0, column=0, padx=5, pady=(10, 0))

        self.label_output = tk.Label(self.root, text="Output Image", font=("Arial", 10, "bold"))
        self.label_output.grid(row=0, column=1, padx=5, pady=(10, 0))

        self.label_overlay = tk.Label(self.root, text="Overlay", font=("Arial", 10, "bold"))
        self.label_overlay.grid(row=0, column=2, padx=5, pady=(10, 0))

        blank_image = Image.new("RGB", (256, 256), (200, 200, 200))
        blank_photo = ImageTk.PhotoImage(blank_image)

        self.label1 = tk.Label(self.root, borderwidth=2, relief="groove", image=blank_photo)
        self.label1.image = blank_photo  
        self.label1.grid(row=1, column=0, padx=5, pady=5)

        self.label2 = tk.Label(self.root, borderwidth=2, relief="groove", image=blank_photo)
        self.label2.image = blank_photo  
        self.label2.grid(row=1, column=1, padx=5, pady=5)

        self.label3 = tk.Label(self.root, borderwidth=2, relief="groove", image=blank_photo)
        self.label3.image = blank_photo  
        self.label3.grid(row=1, column=2, padx=5, pady=5)

        self.btn_select = tk.Button(self.root, text="Select Image", command=self.select_image)
        self.btn_select.grid(row=2, column=0, padx=5, pady=(5,0), sticky="we")

        self.btn_run = tk.Button(self.root, text="Run Model", command=self.run_model)
        self.btn_run.grid(row=3, column=0, padx=5, pady=(5,0), sticky="we")

        self.OUTPUT = tk.Text(self.root, width=5, height=4, state='disabled', font=("Courier", 8))
        self.OUTPUT.grid(row=2, rowspan=2, column=1, columnspan=2, padx=5, pady=(5,0), sticky="we")

    def select_image(self):
        self.input = filedialog.askopenfilename(filetypes=[("TIFF Images", "*.tif")])
        if self.input:
            image = imread(self.input).astype(float)
            rgb = np.dstack((image[:, :, 3], image[:, :, 2], image[:, :, 1]))  # Bands 4, 3, 2
            rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())
            rgb = (rgb * 255).astype(np.uint8)
            img = Image.fromarray(rgb)
            img = img.resize((256, 256), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            self.label1.config(image=photo)
            self.label1.image = photo 

    def run_model(self):
        if not self.input:
            print("No image selected!")
            return
        
        print('[1/6] Loading model...')
        model = SegformerForSemanticSegmentation.from_pretrained("./model/Segformer").cuda()
        model.eval()

        print('[2/6] Pre-processing input...')
        image = imread(self.input).astype(float)
        infrared = np.dstack((image[:, :, 6], image[:, :, 5], image[:, :, 1]))  # Bands 6, 5, 1
        infrared = (infrared - infrared.min()) / (infrared.max() - infrared.min())
        infrared = torch.tensor(infrared, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).cuda()

        print('[3/6] Processing input...')
        with torch.no_grad():
            output = model(infrared)
        logits = nn.functional.interpolate(output.logits, size=256, mode='bilinear', align_corners=False)
        mask = torch.argmax(logits, dim=1).squeeze().cpu().numpy()
        
        print('[4/6] Displaying output...')
        mask = (mask * 255).astype(np.uint8)
        mask_image = Image.fromarray(mask)
        mask_photo = ImageTk.PhotoImage(mask_image)
        self.label2.config(image=mask_photo)
        self.label2.image = mask_photo 

        print('[5/6] Displaying overlayed output...')
        image_array = imread(self.input).astype(float)  # Read TIFF as a NumPy array
        rgb = np.dstack((image_array[:, :, 3], image_array[:, :, 2], image_array[:, :, 1]))  # Example: Bands 4,3,2
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())  # Normalize
        rgb = (rgb * 255).astype(np.uint8)  # Convert to 8-bit image
        original_image = Image.fromarray(rgb).convert("RGBA")  # Convert NumPy array to RGBA
        mask_rgba = Image.new("RGBA", (256, 256), (255, 100, 0, 0))  # Orange color
        mask_rgba.putalpha(mask_image)
        overlay_image = Image.alpha_composite(original_image, mask_rgba)
        overlay_photo = ImageTk.PhotoImage(overlay_image)
        self.label3.config(image=overlay_photo)
        self.label3.image = overlay_photo  

        

        fire_pixels = np.sum(mask == 255)
        print(f'[6/6] Classification Result: {"🔥 Fire detected!" if fire_pixels > 1 else "✅ No fire detected."}')

    def redirect_logging(self):
        logger = PrintLogger(self.OUTPUT)
        sys.stdout = logger
        sys.stderr = logger
    
if __name__ == "__main__":
    root = tk.Tk()
    app = ImageApp(root)
    root.mainloop()
