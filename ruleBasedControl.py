import airsim
import time
import threading
import math
import speech_recognition as sr
from googletrans import Translator
from queue import Queue, Empty


# ----------------------------- Utilities -----------------------------

class Timer:
    def __init__(self):
        self.start_time = None

    def start(self):
        self.start_time = time.perf_counter()

    def end(self):
        if self.start_time is None:
            raise ValueError("Timer was not started. Call start() before end().")
        elapsed_time = time.perf_counter() - self.start_time
        self.start_time = None
        return elapsed_time


timer = Timer()


# ----------------------------- Voice I/O -----------------------------

recognizer = sr.Recognizer()

def get_voice_command(source_language: str):
    """Capture voice input and transcribe it (blocking)."""
    with sr.Microphone() as source:
        print("🎤 Listening for command...")
        audio = recognizer.listen(source)
    try:
        command_text = recognizer.recognize_google(audio, language=source_language)
        print(f"🗣️  Transcribed: {command_text}")
        return command_text.lower()
    except sr.UnknownValueError:
        print("🤷 Could not understand the audio.")
        return None
    except sr.RequestError:
        print("⚠️  Could not request results from the speech recognition service.")
        return None


# ----------------------------- NLP / Commands -----------------------------

class DroneCommandProcessor:
    def __init__(self):
        self.commands = {
            'takeoff': ['take off', 'takeoff', 'launch', 'start', 'begin', 'lift'],
            'land': ['land', 'touchdown', 'come down', 'descend ground'],
            'stop': ['stop', 'halt', 'pause', 'freeze', 'stay', 'hover'],
            'up': ['up', 'higher', 'ascend', 'upward', 'upar'],
            'down': ['down', 'lower', 'descend', 'downward', 'niche'],
            'left': ['left', 'leftward', 'baye'],
            'right': ['right', 'rightward', 'daye'],
            'forward': ['forward', 'ahead', 'straight', 'front', 'age'],
            'backward': ['backward', 'back', 'reverse', 'backwards', 'piche'],
            'rotate_left': ['rotate left', 'turn left', 'spin left', 'spin counterclockwise'],
            'rotate_right': ['rotate right', 'turn right', 'spin right', 'spin clockwise'],
            'quit': ['quit', 'exit', 'stop listening', 'shutdown', 'shut down']
        }

    def process_command(self, text: str):
        """Process the command text and return the corresponding action."""
        text = text.lower().strip()

        # Immediate stop if any 'stop' word present
        if any(stop_word in text for stop_word in self.commands['stop']):
            return {'action': 'stop'}

        # Negations like "don't", "do not" result in stop (conservative behavior)
        parts = text.split()
        if any(neg in parts for neg in ['dont', "don't", 'not']):
            return {'action': 'stop'}

        # Rotations have two-stage logic
        if any(rot in text for rot in ['rotate', 'turn', 'spin']):
            if any(left in text for left in ['left', 'counterclockwise']):
                return {'action': 'rotate_left'}
            if any(right in text for right in ['right', 'clockwise']):
                return {'action': 'rotate_right'}

        # Generic match
        for command, variations in self.commands.items():
            if any(variation in text for variation in variations):
                # If a rotate keyword wasn’t spoken, ignore rotate_* accidental matches
                if command in ['rotate_left', 'rotate_right'] and not any(
                    rot in text for rot in ['rotate', 'turn', 'spin']
                ):
                    continue
                return {'action': command}

        return None


# ----------------------------- Drone Control -----------------------------

class DroneController:
    def __init__(self, client: airsim.MultirotorClient):
        self.client = client
        self.is_moving = False
        self._movement_lock = threading.Lock()

    def takeoff(self):
        print("🛫 Taking off...")
        self.client.takeoffAsync().join()

    def land(self):
        print("🛬 Landing...")
        self.client.landAsync().join()

    def hover(self):
        """Make the drone hover at the current position."""
        try:
            self.client.hoverAsync().join()
        except RuntimeError as e:
            if "IOLoop is already running" not in str(e):
                raise

    def stop(self):
        """Stop the drone and maintain its position."""
        print("⛔ Stopping and hovering...")
        with self._movement_lock:
            self.is_moving = False
        self.hover()

    def translate_to_position_local(self, dx: float, dy: float, dz: float):
        """Move the drone relative to its current position."""
        try:
            with self._movement_lock:
                if self.is_moving:
                    print("↪️ Movement skipped; another movement is in progress.")
                    return
                self.is_moving = True

            current_state = self.client.getMultirotorState()
            current_position = current_state.kinematics_estimated.position
            yaw = self.get_yaw()

            dx_world = dx * math.cos(yaw) - dy * math.sin(yaw)
            dy_world = dx * math.sin(yaw) + dy * math.cos(yaw)

            target_position = airsim.Vector3r(
                current_position.x_val + dx_world,
                current_position.y_val + dy_world,
                current_position.z_val + dz
            )

            self.client.moveToPositionAsync(
                target_position.x_val,
                target_position.y_val,
                target_position.z_val,
                velocity=2
            ).join()
        except Exception as e:
            print(f"⚠️ Error in movement: {str(e)}")
            # Recovery to safer state
            self.hover()
        finally:
            with self._movement_lock:
                self.is_moving = False

    def rotate(self, direction: str):
        """Rotate the drone in the specified direction ('left'|'right')."""
        try:
            with self._movement_lock:
                if self.is_moving:
                    print("↪️ Rotation skipped; another command is in progress.")
                    return
                self.is_moving = True

            rate = 10 if direction == 'right' else -10
            self.client.rotateByYawRateAsync(rate, duration=3).join()
        except Exception as e:
            print(f"⚠️ Error in rotation: {str(e)}")
            self.hover()
        finally:
            with self._movement_lock:
                self.is_moving = False

    def get_yaw(self) -> float:
        """Get the drone's current yaw angle."""
        orientation = self.client.getMultirotorState().kinematics_estimated.orientation
        _, _, yaw = airsim.to_eularian_angles(orientation)
        return yaw


# ----------------------------- Task Manager -----------------------------

class TaskManager:
    """Serialize drone commands in their own threads."""
    def __init__(self):
        self.current_task_thread: threading.Thread | None = None
        self.lock = threading.Lock()

    def execute(self, func, *args):
        """Run a command in a separate thread, waiting for any in-progress command to finish."""
        with self.lock:
            if self.current_task_thread and self.current_task_thread.is_alive():
                # Wait for previous task to finish to avoid overlapping motions
                self.current_task_thread.join()

            self.current_task_thread = threading.Thread(target=self._run, args=(func, *args), daemon=True)
            self.current_task_thread.start()

    @staticmethod
    def _run(func, *args):
        try:
            func(*args)
        except Exception as e:
            print(f"⚠️ Command error: {e}")


# ----------------------------- Orchestration -----------------------------

def translate_to_english(translator: Translator, text: str, source_language: str):
    """Translate recognized text to English using Google Translate."""
    try:
        translated = translator.translate(text, src=source_language, dest='en')
        print(f"🌐 Translated: {translated.text}")
        return translated.text.lower()
    except Exception as e:
        print(f"⚠️ Translation error: {e}")
        return None


def voice_listener_loop(source_language: str, out_queue: Queue, shutdown_event: threading.Event):
    """Continuously listen for voice commands and put them into a queue."""
    while not shutdown_event.is_set():
        text = get_voice_command(source_language)
        if text:
            out_queue.put(text)


def main():
    # Language selection
    translator = Translator()
    source_language = input(
        "Enter the language code you will speak (e.g., 'hi' for Hindi, 'es' for Spanish, 'en' for English): "
    ).strip()

    # AirSim client setup
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)

    controller = DroneController(client)
    processor = DroneCommandProcessor()
    tasks = TaskManager()
    timer = Timer()

    # Voice thread + queue
    shutdown_event = threading.Event()
    voice_queue: Queue[str] = Queue()
    listener_thread = threading.Thread(
        target=voice_listener_loop, args=(source_language, voice_queue, shutdown_event), daemon=True
    )
    listener_thread.start()
    print("✅ Voice listener started. Say 'quit' or 'exit' to stop.")

    try:
        while True:
            try:
                raw_text = voice_queue.get(timeout=0.1)  # non-blocking-ish
            except Empty:
                continue

            timer.start()
            translated = translate_to_english(translator, raw_text, source_language)
            if not translated:
                print(f"⏱️ Time taken: {timer.end():.3f}s")
                continue

            print(f"🧭 Processing command: {translated}")
            command_info = processor.process_command(translated)
            if not command_info:
                print("❓ Command not recognized.")
                print(f"⏱️ Time taken: {timer.end():.3f}s")
                continue

            action = command_info['action']
            print(f"▶️  Executing action: {action}")

            movement_distance = 4  # meters
            if action == 'quit':
                print("👋 Shutting down on user request...")
                break

            # Dispatch to TaskManager
            if action == 'stop':
                tasks.execute(controller.stop)
            elif action == 'takeoff':
                tasks.execute(controller.takeoff)
            elif action == 'land':
                tasks.execute(controller.land)
            elif action == 'up':
                tasks.execute(controller.translate_to_position_local, 0, 0, -movement_distance)
            elif action == 'down':
                tasks.execute(controller.translate_to_position_local, 0, 0, movement_distance)
            elif action == 'forward':
                tasks.execute(controller.translate_to_position_local, movement_distance, 0, 0)
            elif action == 'backward':
                tasks.execute(controller.translate_to_position_local, -movement_distance, 0, 0)
            elif action == 'left':
                tasks.execute(controller.translate_to_position_local, 0, -movement_distance, 0)
            elif action == 'right':
                tasks.execute(controller.translate_to_position_local, 0, movement_distance, 0)
            elif action == 'rotate_left':
                tasks.execute(controller.rotate, 'left')
            elif action == 'rotate_right':
                tasks.execute(controller.rotate, 'right')
            else:
                print("❓ Command not recognized.")

            print(f"⏱️ Time taken: {timer.end():.3f}s")

    except KeyboardInterrupt:
        print("\n🔻 Program interrupted by user.")
    finally:
        # Graceful shutdown
        shutdown_event.set()
        try:
            listener_thread.join(timeout=1.0)
        except RuntimeError:
            pass

        try:
            controller.stop()
        except Exception:
            pass

        client.armDisarm(False)
        client.enableApiControl(False)
        print("✅ Drone disarmed and API control released.")


if __name__ == "__main__":
    main()
