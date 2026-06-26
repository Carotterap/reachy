from sympy import true
from reachy_mini import ReachyMini
import transformers
from time import sleep
import numpy
import PIL
from datasets import load_dataset
import torch
import soundfile as sf
from PIL import Image, ImageDraw
from transformers.pipelines.object_detection import ObjectDetectionPipeline
import requests
from transformers import pipeline
yolo_pipeline: ObjectDetectionPipeline = pipeline("object-detection", model="hustvl/yolos-base", device_map="cuda:1")
qwen_pipeline = pipeline("image-text-to-text", model="Qwen/Qwen3.5-0.8B", device_map="auto")
from yolo_recon import yolo_recognition
from quen import qwen_predict   
import PIL.Image
OUT_PATH = "tts_output.wav"
pipe = pipeline("text-to-audio",model="microsoft/speecht5_tts",device_map="cuda:1")
# Fetch real voice profiles from Hugging Face
embeddings_dataset = load_dataset("regisss/cmu-arctic-xvectors", split="validation", trust_remote_code=True)

# Create the missing speaker_embedding variable
speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0).to(pipe.model.device)

with ReachyMini(media_backend="default") as mini:
    while True:
        frame = mini.media.get_frame()
        if frame is None : continue
        rgb_frame = frame[:, :, ::-1]
        image= PIL.Image.fromarray(rgb_frame)
        has_human = yolo_recognition(image=image, pipe=yolo_pipeline)
        if has_human: 
            print ("human found analysing the posture")
            good_posture= qwen_predict(image=image,pipe=qwen_pipeline)

            if good_posture.lower().endswith("yes") :
                print ("good posture")
                sleep(75)
                continue
            else :
                result = pipe (good_posture,
                forward_params={"speaker_embeddings": speaker_embedding})
            sf.write(OUT_PATH, result["audio"], result["sampling_rate"])
            print(f"Saved: {OUT_PATH}")
            print ("audio ready")
            mini.media.play_sound("tts_output.wav")
            sleep(75)
        else :
            print("no humans found sleeeeeep")
            sleep(75)
      
        