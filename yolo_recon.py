from transformers import pipeline
def yolo_recognition(image,pipe):
  predictions = pipe(image)
  for element in predictions:
    if element['label']== "person":
      return True

  return False













    