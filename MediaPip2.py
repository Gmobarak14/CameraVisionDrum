import cv2
import mediapipe as mp
import numpy as np
import pygame
import threading
import time
import math


# ============================================================
# SETTINGS
# ============================================================

SMOOTHING = 0.35

DOWN_VELOCITY_THRESHOLD = 0.0025
UP_VELOCITY_THRESHOLD = 0.0015
MIN_TRAVEL = 0.015

TAP_COOLDOWN = 0.15

# Tap grading thresholds
ON_BEAT_THRESHOLD_MS = 60
CLOSE_THRESHOLD_MS = 120


# MediaPipe fingertip + MCP landmarks
FINGERS = {
    "INDEX": (8, 5),
    "MIDDLE": (12, 9),
    "RING": (16, 13),
    "PINKY": (20, 17),
}


# ============================================================
# METRONOME CLICK SOUND
# ============================================================

pygame.mixer.pre_init(
    frequency=44100,
    size=-16,
    channels=1,
    buffer=256
)

pygame.init()


def make_click():
    sample_rate = 44100
    duration = 0.04
    frequency = 1000

    t = np.linspace(
        0,
        duration,
        int(sample_rate * duration),
        False
    )

    envelope = np.exp(-t * 60)

    wave = (
        np.sin(2 * np.pi * frequency * t)
        * envelope
    )

    audio = (wave * 32767 * 0.5).astype(np.int16)

    return pygame.sndarray.make_sound(audio)


CLICK_SOUND = make_click()


# ============================================================
# METRONOME
# ============================================================

class Metronome:

    def __init__(self, bpm=80):
        self.bpm = bpm
        self.running = False

        self.start_time = None
        self.last_beat_time = 0

        self.thread = None
        self.lock = threading.Lock()

    def start(self):

        with self.lock:
            if self.running:
                return

            self.running = True
            self.start_time = time.perf_counter()

        self.thread = threading.Thread(
            target=self._run,
            daemon=True
        )

        self.thread.start()

    def stop(self):

        with self.lock:
            self.running = False

    def toggle(self):

        if self.running:
            self.stop()
        else:
            self.start()

    def set_bpm(self, bpm):

        bpm = max(30, min(240, bpm))

        with self.lock:

            if bpm != self.bpm:

                self.bpm = bpm

                # Restart timing grid when tempo changes
                if self.running:
                    self.start_time = time.perf_counter()

    def _run(self):

        beat_number = 0

        while True:

            with self.lock:

                if not self.running:
                    break

                bpm = self.bpm
                start_time = self.start_time

            beat_interval = 60.0 / bpm

            next_beat = (
                start_time
                + beat_number * beat_interval
            )

            now = time.perf_counter()

            if now >= next_beat:

                CLICK_SOUND.play()

                self.last_beat_time = next_beat

                beat_number += 1

            else:

                time.sleep(
                    min(
                        0.002,
                        next_beat - now
                    )
                )

    def get_timing_error(self, tap_time):
        """
        Returns timing error relative to nearest beat.

        Negative = early
        Positive = late
        """

        with self.lock:

            if not self.running:
                return None

            start_time = self.start_time
            bpm = self.bpm

        beat_interval = 60.0 / bpm

        elapsed = tap_time - start_time

        nearest_beat_number = round(
            elapsed / beat_interval
        )

        nearest_beat_time = (
            start_time
            + nearest_beat_number * beat_interval
        )

        error_seconds = (
            tap_time - nearest_beat_time
        )

        return error_seconds * 1000


# ============================================================
# TAP DETECTOR
# ============================================================

class TapDetector:

    def __init__(self):
        self.states = {}

    def update(self, key, depth):

        if key not in self.states:

            self.states[key] = {

                "smooth_depth": depth,

                "previous_depth": depth,

                "velocity": 0,

                "state": "idle",

                "start_depth": depth,

                "peak_depth": depth,

                "last_tap": 0
            }

        data = self.states[key]

        # Smooth noisy MediaPipe depth
        smooth_depth = (
            SMOOTHING * depth
            + (1 - SMOOTHING)
            * data["smooth_depth"]
        )

        velocity = (
            smooth_depth
            - data["previous_depth"]
        )

        data["smooth_depth"] = smooth_depth
        data["previous_depth"] = smooth_depth
        data["velocity"] = velocity

        tap = False
        travel = 0

        # ------------------------------------
        # Finger begins moving toward table
        # ------------------------------------

        if data["state"] == "idle":

            if velocity > DOWN_VELOCITY_THRESHOLD:

                data["state"] = "moving_down"

                data["start_depth"] = smooth_depth

                data["peak_depth"] = smooth_depth

        # ------------------------------------
        # Finger is moving toward table
        # ------------------------------------

        elif data["state"] == "moving_down":

            if smooth_depth > data["peak_depth"]:

                data["peak_depth"] = smooth_depth

            # Direction reversed
            if velocity < -UP_VELOCITY_THRESHOLD:

                travel = (
                    data["peak_depth"]
                    - data["start_depth"]
                )

                current_time = time.perf_counter()

                if (
                    travel > MIN_TRAVEL
                    and
                    current_time
                    - data["last_tap"]
                    > TAP_COOLDOWN
                ):

                    tap = True

                    data["last_tap"] = (
                        current_time
                    )

                data["state"] = "idle"

        return tap, smooth_depth, velocity, travel


# ============================================================
# TAP SCORING
# ============================================================

def score_tap(error_ms):

    absolute_error = abs(error_ms)

    if absolute_error <= ON_BEAT_THRESHOLD_MS:

        return "ON BEAT"

    elif absolute_error <= CLOSE_THRESHOLD_MS:

        if error_ms < 0:
            return "SLIGHTLY EARLY"
        else:
            return "SLIGHTLY LATE"

    else:

        if error_ms < 0:
            return "EARLY"
        else:
            return "LATE"


# ============================================================
# MEDIAPIPE
# ============================================================

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


hands = mp_hands.Hands(

    static_image_mode=False,

    max_num_hands=2,

    model_complexity=1,

    min_detection_confidence=0.6,

    min_tracking_confidence=0.6
)


# ============================================================
# OBJECTS
# ============================================================

tap_detector = TapDetector()

metronome = Metronome(
    bpm=80
)


# ============================================================
# CAMERA
# ============================================================

camera = cv2.VideoCapture(0)

camera.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    1280
)

camera.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    720
)


# ============================================================
# WINDOW + TEMPO SLIDER
# ============================================================

WINDOW_NAME = "Rehab Rhythm Trainer"

cv2.namedWindow(WINDOW_NAME)


def tempo_changed(value):

    # Prevent zero BPM
    bpm = max(30, value)

    metronome.set_bpm(bpm)


cv2.createTrackbar(

    "Tempo BPM",

    WINDOW_NAME,

    80,

    200,

    tempo_changed
)


# ============================================================
# PERFORMANCE DATA
# ============================================================

last_feedback = ""

last_feedback_time = 0

last_error = 0

tap_count = 0

on_beat_count = 0


print("Controls:")
print("SPACE = Start / Stop")
print("Q = Quit")


# ============================================================
# MAIN LOOP
# ============================================================

while camera.isOpened():

    success, frame = camera.read()

    if not success:
        break


    frame = cv2.flip(
        frame,
        1
    )


    height, width, _ = frame.shape


    # ========================================================
    # TEMPO
    # ========================================================

    bpm = cv2.getTrackbarPos(
        "Tempo BPM",
        WINDOW_NAME
    )

    bpm = max(30, bpm)

    metronome.set_bpm(bpm)


    # ========================================================
    # MEDIAPIPE
    # ========================================================

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results = hands.process(
        rgb_frame
    )


    # ========================================================
    # HAND DETECTION
    # ========================================================

    if results.multi_hand_landmarks:

        for hand_index, hand_landmarks in enumerate(
            results.multi_hand_landmarks
        ):

            handedness = (
                results
                .multi_handedness[
                    hand_index
                ]
                .classification[0]
                .label
            )


            landmarks = (
                hand_landmarks.landmark
            )


            mp_drawing.draw_landmarks(

                frame,

                hand_landmarks,

                mp_hands.HAND_CONNECTIONS
            )


            # =================================================
            # FINGERS
            # =================================================

            for finger_name, (
                tip_id,
                mcp_id
            ) in FINGERS.items():


                tip = landmarks[
                    tip_id
                ]

                mcp = landmarks[
                    mcp_id
                ]


                # Relative depth
                depth = (
                    tip.z
                    - mcp.z
                )


                key = (
                    handedness,
                    finger_name
                )


                tap, depth, velocity, travel = (
                    tap_detector.update(
                        key,
                        depth
                    )
                )


                # Screen location
                x = int(
                    tip.x * width
                )

                y = int(
                    tip.y * height
                )


                cv2.circle(

                    frame,

                    (x, y),

                    7,

                    (255, 255, 255),

                    -1
                )


                # =================================================
                # TAP DETECTED
                # =================================================

                if tap:

                    tap_time = (
                        time.perf_counter()
                    )


                    if metronome.running:

                        error_ms = (
                            metronome
                            .get_timing_error(
                                tap_time
                            )
                        )


                        result = score_tap(
                            error_ms
                        )


                        tap_count += 1


                        if (
                            abs(error_ms)
                            <= ON_BEAT_THRESHOLD_MS
                        ):

                            on_beat_count += 1


                        last_error = error_ms


                        last_feedback = (
                            f"{handedness} "
                            f"{finger_name}: "
                            f"{result}"
                        )


                        last_feedback_time = (
                            time.time()
                        )


                        print(
                            last_feedback,
                            f"{error_ms:+.1f} ms"
                        )


                    else:

                        last_feedback = (
                            f"{handedness} "
                            f"{finger_name} TAP"
                        )

                        last_feedback_time = (
                            time.time()
                        )


                    # Show tap location
                    cv2.circle(

                        frame,

                        (x, y),

                        35,

                        (255, 255, 255),

                        4
                    )


    # ========================================================
    # BEAT FLASH
    # ========================================================

    if metronome.running:

        time_since_beat = (
            time.perf_counter()
            - metronome.last_beat_time
        )

        if (
            0
            <= time_since_beat
            < 0.08
        ):

            cv2.circle(

                frame,

                (
                    width // 2,
                    100
                ),

                35,

                (255, 255, 255),

                -1
            )


    # ========================================================
    # UI
    # ========================================================

    status = (
        "PLAYING"
        if metronome.running
        else
        "STOPPED"
    )


    cv2.putText(

        frame,

        f"{bpm} BPM",

        (30, 60),

        cv2.FONT_HERSHEY_SIMPLEX,

        1,

        (255, 255, 255),

        2
    )


    cv2.putText(

        frame,

        status,

        (30, 105),

        cv2.FONT_HERSHEY_SIMPLEX,

        1,

        (255, 255, 255),

        2
    )


    # ========================================================
    # TAP FEEDBACK
    # ========================================================

    if (
        time.time()
        - last_feedback_time
        < 0.7
    ):

        cv2.putText(

            frame,

            last_feedback,

            (
                30,
                height - 110
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            1,

            (255, 255, 255),

            3
        )


        if metronome.running:

            cv2.putText(

                frame,

                f"{last_error:+.1f} ms",

                (
                    30,
                    height - 65
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.9,

                (255, 255, 255),

                2
            )


    # ========================================================
    # ACCURACY
    # ========================================================

    if tap_count > 0:

        accuracy = (
            on_beat_count
            / tap_count
            * 100
        )

    else:

        accuracy = 0


    cv2.putText(

        frame,

        f"On Beat: {accuracy:.1f}%",

        (
            width - 260,
            60
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.8,

        (255, 255, 255),

        2
    )


    cv2.putText(

        frame,

        "SPACE: Start/Stop   Q: Quit",

        (
            30,
            height - 20
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.6,

        (255, 255, 255),

        1
    )


    cv2.imshow(
        WINDOW_NAME,
        frame
    )


    # ========================================================
    # KEYBOARD
    # ========================================================

    key = (
        cv2.waitKey(1)
        & 0xFF
    )


    if key == ord("q"):

        break


    elif key == 32:  # SPACE

        metronome.toggle()

        # Reset score when starting
        if metronome.running:

            tap_count = 0
            on_beat_count = 0


# ============================================================
# CLEANUP
# ============================================================

metronome.stop()

camera.release()

cv2.destroyAllWindows()

hands.close()

pygame.quit()