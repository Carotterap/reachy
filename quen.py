from transformers import pipeline
def qwen_predict(image,pipe):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": """Act as an ergonomic posture coach. Closely analyze the user's posture in the image, paying specific attention to whether their hands are flat on the keyboard, if their back is slouching/not straight, or if they are leaning forward.

First, provide a quick evaluation score from 1 to 10 based on how ergonomic their setup is. Then, deliver your final assessment in in a single, friendly sentence starting with 'hey mister...' and ending strictly with a definitive 'Yes' (for good posture) or 'No' (for bad posture)and you MUST end your answer by yes or no if it's a bad position ."""}
            ]
        },
    ]
    predictions = pipe(messages)[-1]['generated_text'][-1]['content']
    return predictions
