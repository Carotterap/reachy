from reachy_mini.daemon.app.routers.state import get_antenna_joint_positions
from shutil import move

from reachy_mini import ReachyMini
import transformers
from time import sleep
import numpy
import PIL
from PIL import Image, ImageDraw
from transformers.pipelines.object_detection import ObjectDetectionPipeline
import requests
from PIL import Image, ImageDraw
from transformers import pipeline
pipe: ObjectDetectionPipeline = pipeline("object-detection", model="hustvl/yolos-base", device_map="cuda:1")
from quen import qwen_predict

with ReachyMini(media_backend="default") as mini:

 while True:
  frame = mini.media.get_frame()
  if frame is None : continue
  rgb_frame = frame[:, :, ::-1]
  image= PIL.Image.fromarray(rgb_frame)
  print("picture")
  

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
  sleep(75)
  if (label):= ("person") :
    print ("alarm")