import cv2
import mediapipe as mp
import time

# -----------------------------
# SETTINGS TO TUNE
# -----------------------------

SMOOTHING = 0.35

# How fast the finger must move toward the table
DOWN_VELOCITY_THRESHOLD = 0.0025

# How fast it must move back upward
UP_VELOCITY_THRESHOLD = 0.0015

# Minimum depth movement before we count it as a tap
MIN_TRAVEL = 0.015

# Prevent one tap from being counted multiple times
TAP_COOLDOWN = 0.15


# MediaPipe landmark numbers
FINGERS = {
    "INDEX": (8, 5),    # fingertip, MCP joint
    "MIDDLE": (12, 9),
    "RING": (16, 13),
    "PINKY": (20, 17),
}


# -----------------------------
# TAP DETECTOR
# -----------------------------

class TapDetector:
    def __init__(self):
        self.states = {}

    def update(self, key, depth):
        """
        key:
            e.g. ("Left", "INDEX")

        depth:
            fingertip depth relative to its MCP joint
        """

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

        # -----------------------------
        # Smooth depth
        # -----------------------------
        smooth_depth = (
            SMOOTHING * depth
            + (1 - SMOOTHING) * data["smooth_depth"]
        )

        velocity = smooth_depth - data["previous_depth"]

        data["smooth_depth"] = smooth_depth
        data["previous_depth"] = smooth_depth
        data["velocity"] = velocity

        tap = False

        # -----------------------------
        # Start downward movement
        # -----------------------------
        if data["state"] == "idle":

            if velocity > DOWN_VELOCITY_THRESHOLD:

                data["state"] = "moving_down"
                data["start_depth"] = smooth_depth
                data["peak_depth"] = smooth_depth

        # -----------------------------
        # Finger currently moving down
        # -----------------------------
        elif data["state"] == "moving_down":

            if smooth_depth > data["peak_depth"]:
                data["peak_depth"] = smooth_depth

            # Movement has reversed upward
            if velocity < -UP_VELOCITY_THRESHOLD:

                travel = (
                    data["peak_depth"]
                    - data["start_depth"]
                )

                current_time = time.time()

                if (
                    travel > MIN_TRAVEL
                    and current_time - data["last_tap"] > TAP_COOLDOWN
                ):

                    tap = True
                    data["last_tap"] = current_time

                data["state"] = "idle"

        return tap, smooth_depth, velocity


# -----------------------------
# MEDIAPIPE SETUP
# -----------------------------

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

tap_detector = TapDetector()


# -----------------------------
# CAMERA
# -----------------------------

camera = cv2.VideoCapture(0)

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

tap_message = ""
tap_message_time = 0


print("Starting hand tracking...")
print("Press Q to quit.")


while camera.isOpened():

    success, frame = camera.read()

    if not success:
        print("Could not read camera.")
        break

    # Mirror image so left/right feels natural
    frame = cv2.flip(frame, 1)

    height, width, _ = frame.shape

    # OpenCV uses BGR, MediaPipe uses RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb_frame)

    # -----------------------------
    # HAND DETECTION
    # -----------------------------

    if results.multi_hand_landmarks:

        for hand_index, hand_landmarks in enumerate(
            results.multi_hand_landmarks
        ):

            # Left / Right hand
            handedness = (
                results.multi_handedness[hand_index]
                .classification[0]
                .label
            )

            landmarks = hand_landmarks.landmark

            # Draw MediaPipe skeleton
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # -----------------------------
            # CHECK EACH FINGER
            # -----------------------------

            for finger_name, (tip_id, mcp_id) in FINGERS.items():

                tip = landmarks[tip_id]
                mcp = landmarks[mcp_id]

                # ---------------------------------------
                # IMPORTANT
                #
                # We use fingertip depth relative to MCP.
                # This reduces false taps caused by the
                # entire hand moving up/down.
                # ---------------------------------------

                depth = tip.z - mcp.z

                key = (handedness, finger_name)

                tap, smooth_depth, velocity = (
                    tap_detector.update(key, depth)
                )

                # Convert tip location to pixels
                x = int(tip.x * width)
                y = int(tip.y * height)

                # Finger marker
                cv2.circle(
                    frame,
                    (x, y),
                    8,
                    (255, 255, 255),
                    -1
                )

                # -----------------------------
                # TAP!
                # -----------------------------

                if tap:

                    tap_message = (
                        f"{handedness.upper()} "
                        f"{finger_name} TAP"
                    )

                    tap_message_time = time.time()

                    print(tap_message)

                    # Large circle where tap happened
                    cv2.circle(
                        frame,
                        (x, y),
                        35,
                        (255, 255, 255),
                        4
                    )

            # Display hand label near wrist
            wrist = landmarks[0]

            wrist_x = int(wrist.x * width)
            wrist_y = int(wrist.y * height)

            cv2.putText(
                frame,
                handedness,
                (wrist_x, wrist_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

    # -----------------------------
    # SHOW TAP MESSAGE
    # -----------------------------

    if time.time() - tap_message_time < 0.4:

        cv2.putText(
            frame,
            tap_message,
            (40, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.4,
            (255, 255, 255),
            3
        )

    cv2.putText(
        frame,
        "Press Q to quit",
        (30, height - 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.imshow("MediaPipe Tap Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# -----------------------------
# CLEANUP
# -----------------------------

camera.release()
cv2.destroyAllWindows()
hands.close()