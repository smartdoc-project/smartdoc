import os
import random
from PIL import Image, ImageDraw, ImageFont

import numpy as np
from PIL import ImageFilter
random.seed(42)
np.random.seed(42)

for split in ["train", "val", "test"]:
    os.makedirs(f"data/{split}/clean", exist_ok=True)
os.makedirs("data/test/noisy", exist_ok=True)

MOTS = ["facture", "client", "montant", "total", "date", "contrat", "prix", "TVA",
        "paiement", "adresse", "commande", "livraison", "service", "dossier"]

def abimer(page):
    page = page.filter(ImageFilter.GaussianBlur(random.uniform(0.3, 1.2)))  # flou
    img = np.array(page) / 255.0                                            # pixels entre 0 et 1
    img = 1 - (1 - img) * random.uniform(0.5, 0.9)                          # texte plus gris (contraste)
    img = img + np.random.normal(0, random.uniform(0.02, 0.1), img.shape)   # bruit
    img = np.clip(img, 0, 1)                                                # rester entre 0 et 1
    return Image.fromarray((img * 255).astype(np.uint8))

for i in range(300):
    page = Image.new("L", (640, 896), 255)
    dessin = ImageDraw.Draw(page)
    police = ImageFont.load_default(size=16)

    for y in range(40, 860, 25):
        ligne = " ".join(random.choices(MOTS, k=9))
        dessin.text((40, y), ligne, font=police, fill=0)

    if i < 210:
        split = "train"
    elif i < 255:
        split = "val"
    else:
        split = "test"

    page.save(f"data/{split}/clean/page_{i}.png")
    if split == "test":
        abimer(page).save(f"data/test/noisy/page_{i}.png")

print("300 pages créées: 210 train, 45 val, 45 test")


