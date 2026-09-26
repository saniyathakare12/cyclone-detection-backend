# Placeholder for Person 2's trained model.
# Once they hand off a real .pt/.h5 file, only load_model() and run_inference()
# need to change — nothing in main.py should need to change.

import random

MODEL = None  # will hold the loaded model object later

def load_model():
    """
    Called once at server startup.
    Later: replace this with actual model loading, e.g.
        import torch
        model = torch.load("cyclone_model.pt")
        model.eval()
        return model
    """
    global MODEL
    MODEL = "placeholder-model"  # fake for now
    print("Model loaded (placeholder)")

def run_inference(image_bytes: bytes):
    """
    Takes raw image bytes, returns (category, confidence).
    Later: preprocess image_bytes exactly as Person 2 did during training,
    run it through MODEL, and return the real prediction.
    """
    categories = ["Depression", "Category 1", "Category 2", "Category 3", "Category 4", "Category 5"]
    category = random.choice(categories)
    confidence = round(random.uniform(0.70, 0.99), 2)
    return category, confidence