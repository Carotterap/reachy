from transformers import pipeline
def qwen_predict(image,pipe):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Is the person's posture ergonomic ? you need to watch closly the hands on the keybord and the back if it's not strait it's a bad posture and if the person is leaning forward is alsow a bad posture, answer in a single sentence that MUST ends with yes or no in 40 words"}
            ]
        },
    ]
    predictions = pipe(messages)[-1]['generated_text'][-1]['content']
    return predictions
