import requests
from PIL import Image, ImageDraw
from transformers import pipeline

# 1. Initialisation du pipeline (sur le GPU cuda:1 comme demandé)
pipe = pipeline("object-detection", model="hustvl/yolos-base", device_map="cuda:1")

# 2. Téléchargement de l'image PIL directement depuis l'URL
url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

# 3. Exécution de la prédiction (on passe l'objet image directement)
predictions = pipe(image)

# 4. Préparation du dessin sur l'image
draw = ImageDraw.Draw(image)

# 5. Boucle pour tracer chaque boîte englobante
for pred in predictions:
    box = pred["box"]
    label = pred["label"]
    score = pred["score"]
    
    # Extraction des coordonnées
    xmin, ymin, xmax, ymax = box["xmin"], box["ymin"], box["xmax"], box["ymax"]
    
    # Dessiner le rectangle (en rouge, épaisseur de 3 pixels)
    draw.rectangle([(xmin, ymin), (xmax, ymax)], outline="red", width=3)
    
    # Ajouter le texte du label et du score juste au-dessus du rectangle
    text = f"{label} ({score:.2f})"
    draw.text((xmin, max(0, ymin - 12)), text, fill="red")
image.save("resultat_detection.jpg")
# 6. Afficher l'image finale
image.show()

# (Optionnel) Sauvegarder l'image sur le disque :
