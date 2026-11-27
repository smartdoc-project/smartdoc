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
LARGEUR, HAUTEUR, MARGE, INTERLIGNE = 640, 896, 50, 20
LARGEUR_UTILE = LARGEUR - 2 * MARGE
# TeX Gyre Termes: free Times clone (GUST Font License), shipped in the repo so every OS renders the same pages.
POLICES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")
police = ImageFont.truetype(os.path.join(POLICES, "texgyretermes-regular.otf"), 14)
police_titre = ImageFont.truetype(os.path.join(POLICES, "texgyretermes-bold.otf"), 20)

def abimer(page):
    page = page.filter(ImageFilter.GaussianBlur(random.uniform(0.3, 1.2)))  
    img = np.array(page) / 255.0                                            
    img = 1 - (1 - img) * random.uniform(0.5, 0.9)                          
    img = img + np.random.normal(0, random.uniform(0.02, 0.1), img.shape)   
    img = np.clip(img, 0, 1)                                                
    return Image.fromarray((img * 255).astype(np.uint8))

def ecrire_paragraphe(dessin, mots, y):
    lignes, ligne = [], []
    for mot in mots:                                  
        if police.getlength(" ".join(ligne + [mot])) > LARGEUR_UTILE:
            lignes.append(ligne)
            ligne = []
        ligne.append(mot)
    lignes.append(ligne)
    for k, ligne in enumerate(lignes):                
        if y > HAUTEUR - MARGE - INTERLIGNE:
            break
        if k == len(lignes) - 1:
            espace = police.getlength(" ")
        else:
            espace = (LARGEUR_UTILE - sum(police.getlength(m) for m in ligne)) / (len(ligne) - 1)
        x = MARGE
        for m in ligne:
            dessin.text((x, y), m, font=police, fill=0)
            x += police.getlength(m) + espace
        y += INTERLIGNE
    return y


for i in range(300):
    page = Image.new("L", (LARGEUR, HAUTEUR), 255)
    dessin = ImageDraw.Draw(page)
    titre = random.choice(MOTS).upper()
    dessin.text(((LARGEUR - police_titre.getlength(titre)) / 2, MARGE), titre, font=police_titre, fill=0)
    y = MARGE + 50
    while y < HAUTEUR - MARGE - INTERLIGNE:
        y = ecrire_paragraphe(dessin, random.choices(MOTS, k=random.randint(25, 60)), y) + 12
    split = "train" if i < 210 else "val" if i < 255 else "test"
    page.save(f"data/{split}/clean/page_{i}.png")
    if split == "test":
        abimer(page).save(f"data/test/noisy/page_{i}.png")

print("300 pages créées: 210 train, 45 val, 45 test")