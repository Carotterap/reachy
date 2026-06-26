from transformers import pipeline
def speech_cat (text,pipe):
 OUT_PATH = "tts_output.wav"
 pipe = pipeline(
    "text-to-audio",
    model="microsoft/speecht5_tts",
    device_map="cuda:1"
) 
speaker_embedding = torch.randn(1, 512).to(pipe.model.device)
result = pipe(
    "Hey JD, you are not sitting properly.",
    forward_params={"speaker_embeddings": speaker_embedding}
)
sf.write(OUT_PATH, result["audio"], result["sampling_rate"])

print(f"Saved: {OUT_PATH}")




from transformers import pipeline

pipe = pipeline("text-to-audio", "microsoft/speecht5_tts", device_map="auto")
speaker_embedding = torch.randn(1, 512).to(pipe.model.device)
result = pipe("Hey JD, you are not sitting properly.", forward_params={"speaker_embeddings": speaker_embedding})
sf.write("tts_output.wav", result["audio"], result["sampling_rate"])
print(f"Saved: {OUT_PATH}")
```