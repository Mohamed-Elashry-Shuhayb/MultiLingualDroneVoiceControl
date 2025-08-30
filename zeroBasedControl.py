"""
Production-ready threaded voice-controlled AirSim controller with:
 - zero-shot classification (facebook/bart-large-mnli)
 - HuggingFace translation (Helsinki multilingual -> en) with fallback
 - negation detection
 - robust task management (cancelable AirSim futures)
 - safer rotation, stop, and task cancellation behavior
"""

import os
import re
import time
import math
import threading
import queue
from typing import Optional, Callable, Any, Tuple

import airsim
import speech_recognition as sr
from transformers import pipeline
from googletrans import Translator as _GoogleTrans  # fallback

# ---------------------------
# Configuration
# ---------------------------
ZERO_SHOT_MODEL = "facebook/bart-large-mnli"
TRANSLATE_MODEL = "Helsinki-NLP/opus-mt-mul-en"
# Labels for zero-shot classification (adjust wording to match your earlier labels)
ZS_LABELS = [
    "take off", "land", "up", "down", "forward", "backward",
    "left", "right", "rotate left", "rotate right", "stop", "scan", "analyse"
]
ZS_CONFIDENCE_THRESHOLD = 0.55  # stricter than 0.2

# Movement parameters
MOVE_VELOCITY = 1.0
MOVE_SCALE = 5.0  # used to scale dx/dy in translate_to_position_local
ROTATE_DEG_PER_SEC = 30.0  # yaw rate for rotateByYawRateAsync
ROTATE_DURATION = 2.0      # bounded rotation duration (seconds)
TRANSLATOR_TIMEOUT = 5.0   # seconds to attempt HF translation

# Image save folder
IMAGE_SAVE_DIR = os.path.join(os.getcwd(), "collected_photos")
os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)

# ---------------------------
# Utilities: Timer
# ---------------------------
class Timer:
    def __init__(self):
        self._start = None

    def start(self):
        self._start = time.perf_counter()

    def end(self) -> float:
        if self._start is None:
            raise RuntimeError("Timer not started. Call start() before end().")
        elapsed = time.perf_counter() - self._start
        self._start = None
        return elapsed

# ---------------------------
# Translation + Zero-shot setup
# ---------------------------
print("[INIT] Loading models... (this may take a moment)")
try:
    translator_pipe = pipeline("translation", model=TRANSLATE_MODEL)
    hf_translation_available = True
    print("[INIT] HuggingFace translator loaded.")
except Exception as e:
    print(f"[INIT] HF translator failed to load ({e}), falling back to googletrans.")
    translator_pipe = None
    hf_translation_available = False

try:
    zero_shot = pipeline("zero-shot-classification", model=ZERO_SHOT_MODEL)
    print("[INIT] Zero-shot model loaded.")
except Exception as e:
    print(f"[INIT] Zero-shot model load failed: {e}")
    raise

# fallback translator (online, unreliable depending on network/API changes)
googletrans = _GoogleTrans()

# ---------------------------
# AirSim connection & controller
# ---------------------------
print("[INIT] Connecting to AirSim...")
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)
print("[INIT] AirSim connected and armed.")

class DroneController:
    """
    Wraps AirSim calls and stores/cancels the last async task safely.
    All long-running commands should register tasks via _set_last_task.
    """
    def __init__(self, client: airsim.MultirotorClient):
        self.client = client
        self._last_task = None
        self._task_lock = threading.Lock()

    def _set_last_task(self, task):
        with self._task_lock:
            self._last_task = task

    def _cancel_last_task(self):
        with self._task_lock:
            if self._last_task is not None:
                try:
                    self._last_task.cancel()
                except Exception:
                    pass
                self._last_task = None

    def takeoff(self):
        print("[DRONE] takeoff")
        self._cancel_last_task()
        try:
            t = self.client.takeoffAsync()
            self._set_last_task(t)
            t.join()
        finally:
            self._set_last_task(None)

    def land(self):
        print("[DRONE] land")
        self._cancel_last_task()
        try:
            t = self.client.landAsync()
            self._set_last_task(t)
            t.join()
        finally:
            self._set_last_task(None)

    def hover(self):
        print("[DRONE] hover")
        self._cancel_last_task()
        try:
            t = self.client.hoverAsync()
            self._set_last_task(t)
            t.join()
        finally:
            self._set_last_task(None)

    def stop(self):
        print("[DRONE] stop - set velocity 0 and cancel previous task")
        # cancel any high-level task and issue immediate stop
        self._cancel_last_task()
        try:
            self.client.moveByVelocityAsync(0, 0, 0, duration=1).join()
        except Exception:
            pass
        # ensure hover after stopping
        try:
            self.hover()
        except Exception:
            pass

    def translate_to_position_local(self, cancel_event: threading.Event, dx: float, dy: float, dz: float):
        """
        Move relative to current orientation. Polls cancel_event and cancels the AirSim task if requested.
        """
        print(f"[DRONE] translate local dx={dx}, dy={dy}, dz={dz}")
        self._cancel_last_task()
        try:
            state = self.client.getMultirotorState()
            pos = state.kinematics_estimated.position
            yaw = self.get_yaw()
            dx_world = dx * math.cos(yaw) - dy * math.sin(yaw)
            dy_world = dx * math.sin(yaw) + dy * math.cos(yaw)

            target = airsim.Vector3r(
                pos.x_val + MOVE_SCALE * dx_world,
                pos.y_val + MOVE_SCALE * dy_world,
                pos.z_val + dz
            )
            task = self.client.moveToPositionAsync(target.x_val, target.y_val, target.z_val, MOVE_VELOCITY)
            self._set_last_task(task)

            # Wait and poll for cancellation
            while not cancel_event.is_set():
                # Short sleep to avoid busy loop
                time.sleep(0.1)
                # Try to detect completion; join will return if done.
                try:
                    task.join(timeout=0.1)
                    break
                except TypeError:
                    # Some AirSim join implementations may not accept timeout;
                    # we ignore and continue.
                    pass
            # If cancellation requested, attempt to cancel the AirSim task
            if cancel_event.is_set():
                try:
                    task.cancel()
                except Exception:
                    pass
            # ensure finalization
            try:
                task.join()
            except Exception:
                pass
        finally:
            self._set_last_task(None)
            print("[DRONE] translate finished/cancelled")

    def rotate_by_rate(self, cancel_event: threading.Event, direction: str):
        """
        Rotate at yaw rate for a bounded duration, or until canceled.
        direction: 'left' or 'right'
        """
        print(f"[DRONE] rotate_by_rate {direction}")
        self._cancel_last_task()
        rate = -ROTATE_DEG_PER_SEC if direction.lower().startswith("left") else ROTATE_DEG_PER_SEC
        try:
            task = self.client.rotateByYawRateAsync(rate, duration=ROTATE_DURATION)
            self._set_last_task(task)
            # wait for the bounded rotation to complete or for cancel_event
            start = time.time()
            while time.time() - start < ROTATE_DURATION and not cancel_event.is_set():
                time.sleep(0.05)
            if cancel_event.is_set():
                try:
                    task.cancel()
                except Exception:
                    pass
            try:
                task.join()
            except Exception:
                pass
        finally:
            self._set_last_task(None)
            print("[DRONE] rotate finished/cancelled")

    def scan(self, save_path: Optional[str] = None):
        if save_path is None:
            save_path = os.path.join(IMAGE_SAVE_DIR, "scan_image.png")
        print(f"[DRONE] scan -> {save_path}")
        try:
            responses = self.client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])
            if responses and len(responses) > 0:
                resp = responses[0]
                with open(save_path, "wb") as f:
                    f.write(resp.image_data_uint8)
                print("[DRONE] scan saved.")
            else:
                print("[DRONE] scan got no response.")
        except Exception as e:
            print(f"[DRONE] scan exception: {e}")

    def analyse(self, save_path: Optional[str] = None):
        if save_path is None:
            save_path = os.path.join(IMAGE_SAVE_DIR, "analyse_image.png")
        print(f"[DRONE] analyse -> {save_path}")
        try:
            responses = self.client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Segmentation, False, False)])
            if responses and len(responses) > 0:
                resp = responses[0]
                with open(save_path, "wb") as f:
                    f.write(resp.image_data_uint8)
                print("[DRONE] analyse saved.")
            else:
                print("[DRONE] analyse got no response.")
        except Exception as e:
            print(f"[DRONE] analyse exception: {e}")

    def get_yaw(self) -> float:
        orientation = self.client.getMultirotorState().kinematics_estimated.orientation
        _, _, yaw = airsim.to_eularian_angles(orientation)
        return yaw

# ---------------------------
# Task Manager
# ---------------------------
class TaskManager:
    """
    Serializes tasks. New tasks cancel the previous by setting its cancel_event.
    Each task runs in its own thread; task functions that accept cancel_event should support cancellation.
    """
    def __init__(self):
        self._thread: Optional[threading.Thread] = None
        self._cancel_event: Optional[threading.Event] = None
        self._lock = threading.Lock()

    def execute(self, func: Callable[..., Any], *args):
        with self._lock:
            # cancel previous
            if self._cancel_event is not None:
                self._cancel_event.set()
            if self._thread is not None and self._thread.is_alive():
                # allow the previous thread to finalize
                self._thread.join(timeout=1.0)

            cancel_event = threading.Event()
            self._cancel_event = cancel_event

            def wrapper():
                try:
                    # try calling with cancel_event first
                    try:
                        func(cancel_event, *args)
                    except TypeError:
                        # fallback if func does not accept cancel_event
                        func(*args)
                except Exception as e:
                    print(f"[TaskManager] task error: {e}")

            thread = threading.Thread(target=wrapper, daemon=True)
            self._thread = thread
            thread.start()

    def cancel_current(self):
        with self._lock:
            if self._cancel_event:
                self._cancel_event.set()

# ---------------------------
# Voice recognition (background)
# ---------------------------
command_queue: "queue.Queue[str]" = queue.Queue()
recognizer = sr.Recognizer()

def voice_listener(language_code: str, shutdown_event: threading.Event):
    """
    Background thread: listen and put transcribed texts into command_queue.
    """
    try:
        mic = sr.Microphone()
    except OSError as e:
        print("[VOICE] Microphone not available:", e)
        return

    with mic as source:
        recognizer.adjust_for_ambient_noise(source, duration=1.0)
        print("[VOICE] Microphone calibrated.")
        while not shutdown_event.is_set():
            try:
                audio = recognizer.listen(source, phrase_time_limit=6)
            except Exception as e:
                print("[VOICE] listen error:", e)
                continue

            try:
                text = recognizer.recognize_google(audio, language=language_code)
                text = text.strip()
                if text:
                    print("[VOICE] Heard:", text)
                    command_queue.put(text)
            except sr.UnknownValueError:
                print("[VOICE] Could not understand audio.")
            except sr.RequestError as e:
                print("[VOICE] SR request error:", e)
                time.sleep(1.0)

# ---------------------------
# Helpers: translation, negation, classify
# ---------------------------
def translate_to_english(text: str) -> str:
    """Prefer HuggingFace translator; fallback to googletrans if HF not available."""
    if not text:
        return text
    if hf_translation_available and translator_pipe is not None:
        try:
            out = translator_pipe(text, max_length=200)
            if isinstance(out, list) and out:
                # pipeline returns list of dicts with "translation_text"
                return out[0].get("translation_text", text).lower()
            return str(out).lower()
        except Exception as e:
            print("[TRANS] HF translate failed:", e)
    # fallback
    try:
        return googletrans.translate(text, src="auto", dest="en").text.lower()
    except Exception as e:
        print("[TRANS] googletrans fallback failed:", e)
        return text.lower()

def detect_negation(text: str) -> bool:
    """Simple negation detection. Expand pattern as needed."""
    return bool(re.search(r"\b(don't|do not|not|never|dont)\b", text, flags=re.I))

def classify_zero_shot(text: str) -> Optional[Tuple[str, float]]:
    """Classify using zero-shot pipeline. Returns (label, score) if above threshold."""
    try:
        res = zero_shot(text, ZS_LABELS, multi_label=False)
        label = res["labels"][0]
        score = float(res["scores"][0])
        print(f"[ZS] label='{label}', score={score:.3f}")
        if score >= ZS_CONFIDENCE_THRESHOLD:
            return label, score
        else:
            print("[ZS] below confidence threshold")
            return None
    except Exception as e:
        print("[ZS] classification error:", e)
        return None

# ---------------------------
# High-level command execution mapping
# ---------------------------
drone = DroneController(client)
tasks = TaskManager()
timer = Timer()

def execute_mapped_command(label: str, raw_text: str):
    """
    Map the zero-shot label string to concrete drone actions.
    Label strings are expected in ZS_LABELS (e.g., "take off", "rotate left" etc.)
    """
    # negation guard
    if detect_negation(raw_text):
        print("[EXEC] Negation detected in original phrase - skipping execution.")
        return

    label_normalized = label.strip().lower()

    # direct mappings
    if label_normalized == "take off" or label_normalized == "takeoff" or label_normalized == "take off":
        tasks.execute(drone.takeoff)
    elif label_normalized == "land":
        tasks.execute(drone.land)
    elif label_normalized == "up":
        # up -> negative z in AirSim (climb)
        tasks.execute(drone.translate_to_position_local, 0, 0, -10)
    elif label_normalized == "down":
        tasks.execute(drone.translate_to_position_local, 0, 0, 10)
    elif label_normalized == "forward":
        tasks.execute(drone.translate_to_position_local, 10, 0, 0)
    elif label_normalized == "backward":
        tasks.execute(drone.translate_to_position_local, -10, 0, 0)
    elif label_normalized == "left":
        tasks.execute(drone.translate_to_position_local, 0, -10, 0)
    elif label_normalized == "right":
        tasks.execute(drone.translate_to_position_local, 0, 10, 0)
    elif label_normalized == "rotate left":
        tasks.execute(drone.rotate_by_rate, "left")
    elif label_normalized == "rotate right":
        tasks.execute(drone.rotate_by_rate, "right")
    elif label_normalized == "stop":
        # cancel current and then stop explicitly
        tasks.cancel_current()
        tasks.execute(drone.stop)
    elif label_normalized == "scan":
        tasks.execute(drone.scan)
    elif label_normalized == "analyse":
        tasks.execute(drone.analyse)
    else:
        print(f"[EXEC] No mapping for label '{label_normalized}'")

# ---------------------------
# Main loop
# ---------------------------
def main(language_code: str = "en"):
    print("[MAIN] Starting voice loop.")
    shutdown = threading.Event()
    listener = threading.Thread(target=voice_listener, args=(language_code, shutdown), daemon=True)
    listener.start()

    try:
        while True:
            try:
                raw = command_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            timer.start()
            print(f"[MAIN] Raw captured: {raw}")

            # Translate to English for classification (zero-shot labels are English)
            translated = translate_to_english(raw)
            print(f"[MAIN] Translated: {translated}")

            # classify zero-shot
            classification = classify_zero_shot(translated)
            if classification:
                label, score = classification
                print(f"[MAIN] Classified as '{label}' (score={score:.3f}). Executing...")
                execute_mapped_command(label, raw)
            else:
                # fallback: try a substring match (simple heuristic)
                lowered = translated.lower()
                fallback_label = None
                for lbl in ZS_LABELS:
                    if re.search(r"\b" + re.escape(lbl) + r"\b", lowered):
                        fallback_label = lbl
                        break
                if fallback_label:
                    print(f"[MAIN] Fallback substring matched: {fallback_label}")
                    execute_mapped_command(fallback_label, raw)
                else:
                    print("[MAIN] Could not recognize a confident command.")

            elapsed = timer.end()
            print(f"[MAIN] Command round-trip time: {elapsed:.3f}s")

    except KeyboardInterrupt:
        print("[MAIN] KeyboardInterrupt, shutting down...")
    finally:
        shutdown.set()
        # Cancel tasks and land safely
        print("[MAIN] Cancelling tasks and landing.")
        tasks.cancel_current()
        time.sleep(0.5)
        try:
            drone.stop()
            drone.land()
        except Exception:
            pass
        client.armDisarm(False)
        client.enableApiControl(False)
        print("[MAIN] Shutdown complete.")

# ---------------------------
# Entrypoint
# ---------------------------
if __name__ == "__main__":
    # Ask user for language (keeps backwards compatibility)
    lang = input("Enter language code you will speak (e.g., 'hi' for Hindi, 'en' for English): ").strip() or "en"
    main(language_code=lang)
