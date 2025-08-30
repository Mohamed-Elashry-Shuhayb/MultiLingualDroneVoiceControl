import airsim
import spacy
import os
import re
import time
import threading
import queue
import speech_recognition as sr
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
from difflib import SequenceMatcher

# ==========================================
# 1. Load Models
# ==========================================
print("Loading NLP models...")
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-mul-en")
nlp = spacy.load("en_core_web_md")  # still useful for negation
embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# ==========================================
# 2. AirSim Client
# ==========================================
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)
print("Drone Connected!")

# ==========================================
# 3. Intent Dictionary
# ==========================================
INTENTS = {
    "takeoff": ["take off", "lift off", "launch"],
    "land": ["land", "descend", "touch down"],
    "hover": ["hover", "stay still", "stop moving"],
    "fly_forward": ["fly forward", "move ahead", "go straight"],
    "fly_backward": ["fly backward", "move back"],
    "fly_left": ["fly left", "move left", "strafe left"],
    "fly_right": ["fly right", "move right", "strafe right"],
    "rotate_left": ["rotate left", "turn left", "yaw left"],
    "rotate_right": ["rotate right", "turn right", "yaw right"],
    "scan": ["scan", "take picture", "capture image"],
    "analyse": ["analyse", "detect", "recognize"],
    "stop": ["stop", "halt", "freeze"],
}

# Precompute embeddings
INTENT_EMBEDS = {
    intent: embedder.encode(phrases, convert_to_tensor=True)
    for intent, phrases in INTENTS.items()
}

# ==========================================
# 4. Command Queue (Non-blocking Speech Input)
# ==========================================
command_queue = queue.Queue()
recognizer = sr.Recognizer()
mic = sr.Microphone()

def listen_loop():
    with mic as source:
        recognizer.adjust_for_ambient_noise(source)
        print("🎤 Listening for commands...")
        while True:
            try:
                audio = recognizer.listen(source, timeout=None)
                text = recognizer.recognize_google(audio, language="auto")
                command_queue.put(text)
            except sr.UnknownValueError:
                print("Could not understand audio.")
            except Exception as e:
                print(f"[ERROR] Speech recognition: {e}")

threading.Thread(target=listen_loop, daemon=True).start()

# ==========================================
# 5. Intent Detection
# ==========================================
def detect_intent(command: str):
    # --- Handle negation ---
    if any(word in command for word in ["don't", "do not", "not"]):
        print("Negation detected, skipping command.")
        return None

    # --- Exact match ---
    for intent, phrases in INTENTS.items():
        if command in phrases:
            return intent

    # --- Embedding similarity ---
    command_vec = embedder.encode(command, convert_to_tensor=True)
    best_intent, best_score = None, -1
    for intent, phrase_vecs in INTENT_EMBEDS.items():
        sim = util.cos_sim(command_vec, phrase_vecs).max().item()
        if sim > best_score:
            best_intent, best_score = intent, sim
    if best_score > 0.7:
        return best_intent

    # --- Fuzzy fallback ---
    for intent, phrases in INTENTS.items():
        for phrase in phrases:
            score = SequenceMatcher(None, command, phrase).ratio()
            if score > 0.75:
                return intent

    return None

# ==========================================
# 6. Drone Control Functions
# ==========================================
def takeoff():
    print("Drone taking off...")
    client.takeoffAsync().join()

def land():
    print("Drone landing...")
    client.landAsync().join()

def hover():
    print("Drone hovering...")
    client.hoverAsync().join()

def fly_forward():
    client.moveByVelocityAsync(2, 0, 0, 2)

def fly_backward():
    client.moveByVelocityAsync(-2, 0, 0, 2)

def fly_left():
    client.moveByVelocityAsync(0, -2, 0, 2)

def fly_right():
    client.moveByVelocityAsync(0, 2, 0, 2)

def rotate_left():
    client.rotateByYawRateAsync(-10, 2)

def rotate_right():
    client.rotateByYawRateAsync(10, 2)

def scan():
    save_path = os.path.join(os.getcwd(), "scan_image.png")
    raw = client.simGetImage("0", airsim.ImageType.Scene)
    if raw:
        with open(save_path, "wb") as f:
            f.write(bytearray(raw))
        print(f"Image saved to {save_path}")

def analyse():
    print("Analysing environment (stubbed ML call).")

def stop():
    client.moveByVelocityAsync(0, 0, 0, 1)
    print("Drone stopped.")

# Map intents to functions
INTENT_ACTIONS = {
    "takeoff": takeoff,
    "land": land,
    "hover": hover,
    "fly_forward": fly_forward,
    "fly_backward": fly_backward,
    "fly_left": fly_left,
    "fly_right": fly_right,
    "rotate_left": rotate_left,
    "rotate_right": rotate_right,
    "scan": scan,
    "analyse": analyse,
    "stop": stop,
}

# ==========================================
# 7. Main Execution Loop
# ==========================================
def main_loop():
    while True:
        command_text = command_queue.get()  # blocking until a command is spoken
        print(f"🗣️ Heard: {command_text}")

        # Translate if not English
        try:
            translated = translator(command_text, max_length=40)[0]["translation_text"]
            print(f"🌍 Translated: {translated}")
        except:
            translated = command_text

        intent = detect_intent(translated.lower())
        if intent:
            print(f"✅ Detected Intent: {intent}")
            INTENT_ACTIONS[intent]()
        else:
            print("❌ No matching intent found.")

# ==========================================
# Start
# ==========================================
if __name__ == "__main__":
    main_loop()
