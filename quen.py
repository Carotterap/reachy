from transformers import pipeline
p = pipeline("image-text-to-text", model="Qwen/Qwen3.5-9B",device_map="auto")


def qwen_predict(image,pipe):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Is the person in a good posture for work ?"}
            ]
        },
    ]
    return pipe(messages)

import requests
from PIL import Image, ImageDraw
url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)


out =qwen_predict(image=image, pipe=p)
print(out)
