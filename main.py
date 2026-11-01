"""
Virtual Steering Wheel
Controleaza un joc de curse cu doua maini in fata webcam-ului.

Taste:  c = calibrare (tine mainile in pozitia "volan drept")
        g = pornit/oprit trimiterea catre controller
        q = iesire
"""
import math
import time

import cv2
import mediapipe as mp
import numpy as np

try:
    import vgamepad as vg
    HAS_GAMEPAD = True
except Exception:  # driverul ViGEmBus lipseste sau vgamepad nu e instalat
    HAS_GAMEPAD = False

# ---------------- Configurare ----------------
CAM_INDEX = 0
FRAME_W, FRAME_H = 640, 480
MAX_ANGLE_DEG = 45.0     # unghiul la care volanul e dat "la maxim"
DEADZONE = 0.05          # zona moarta in jurul centrului (0..1)
MIN_CUTOFF = 1.0         # One Euro: mai mic = mai neted, dar mai lent
BETA = 0.02              # One Euro: mai mare = reactioneaza mai repede la miscari rapide
AUTO_THROTTLE = 0.6      # acceleratie constanta cand nu frani (0..1)
MAX_READ_FAILURES = 30   # cate cadre ratate la rand toleram inainte sa iesim
WINDOW_NAME = "Virtual Steering Wheel"

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


# ---------------- Filtru One Euro ----------------
class OneEuroFilter:
    """Filtru adaptiv: neteda zgomotul cand miscarea e lenta,
    dar ramane reactiv cand miscarea e rapida."""

    def __init__(self, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        if self.x_prev is None:
            self.x_prev, self.t_prev = x, t
            return x
        dt = max(t - self.t_prev, 1e-6)
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev, self.dx_prev, self.t_prev = x_hat, dx_hat, t
        return x_hat


# ---------------- Logica volanului ----------------
def palm_center(hand, w, h):
    """Folosim baza degetului mijlociu (landmark 9): stabil si usor de urmarit."""
    lm = hand.landmark[9]
    return np.array([lm.x * w, lm.y * h])


def steering_angle_deg(left_pt, right_pt):
    """Unghiul liniei dintre maini fata de orizontala.
    Pozitiv = mana dreapta mai jos = virezi la dreapta."""
    dx = right_pt[0] - left_pt[0]
    dy = right_pt[1] - left_pt[1]
    return math.degrees(math.atan2(dy, dx))


def apply_deadzone(value, dz):
    if abs(value) < dz:
        return 0.0
    # rescalare, ca sa nu existe "salt" la marginea zonei moarte
    return math.copysign((abs(value) - dz) / (1 - dz), value)


def send_neutral(pad):
    """Volan drept, fara acceleratie, fara frana. Apelat la orice oprire."""
    if pad is None:
        return
    pad.reset()  # aduce toate butoanele/axele la zero
    pad.update()


def create_gamepad():
    """Incearca sa creeze controllerul virtual; intoarce None daca nu merge."""
    if not HAS_GAMEPAD:
        print("[!] vgamepad nu este instalat: pip install vgamepad")
        return None
    try:
        return vg.VX360Gamepad()
    except Exception as e:  # tipic: driverul ViGEmBus lipseste
        print(f"[!] Nu pot crea controllerul virtual ({e}).")
        print("    Instaleaza driverul ViGEmBus: https://github.com/nefarius/ViGEmBus/releases")
        return None


def is_fist(hand):
    """Pumn = cel mult un deget intins. Un deget e intins daca varful
    e mai departe de incheietura decat articulatia PIP."""
    wrist = np.array([hand.landmark[0].x, hand.landmark[0].y])
    tips, pips = [8, 12, 16, 20], [6, 10, 14, 18]
    extended = 0
    for tip, pip_ in zip(tips, pips):
        d_tip = np.linalg.norm(np.array([hand.landmark[tip].x, hand.landmark[tip].y]) - wrist)
        d_pip = np.linalg.norm(np.array([hand.landmark[pip_].x, hand.landmark[pip_].y]) - wrist)
        if d_tip > d_pip:
            extended += 1
    return extended <= 1


# ---------------- Overlay ----------------
def draw_wheel(frame, steer, brake, throttle, fps, latency_ms, sending, has_pad):
    h, w = frame.shape[:2]
    cx, cy, r = w // 2, h - 90, 60
    cv2.circle(frame, (cx, cy), r, (255, 255, 255), 3)
    ang = math.radians(steer * MAX_ANGLE_DEG * 2 - 90)
    cv2.line(frame, (cx, cy), (int(cx + r * math.cos(ang)), int(cy + r * math.sin(ang))),
             (0, 255, 255), 4)
    # bare pentru acceleratie / frana
    cv2.rectangle(frame, (20, h - 120), (40, h - 20), (255, 255, 255), 1)
    cv2.rectangle(frame, (20, int(h - 20 - 100 * throttle)), (40, h - 20), (0, 200, 0), -1)
    cv2.rectangle(frame, (w - 40, h - 120), (w - 20, h - 20), (255, 255, 255), 1)
    cv2.rectangle(frame, (w - 40, int(h - 20 - 100 * brake)), (w - 20, h - 20), (0, 0, 220), -1)
    status = "GAMEPAD ON" if sending else ("GAMEPAD OFF" if has_pad else "NO GAMEPAD DRIVER")
    # "proc" = doar timpul de procesare; latenta end-to-end reala vine la metrics (Task 6)
    cv2.putText(frame, f"FPS {fps:4.1f} | proc {latency_ms:4.1f} ms | steer {steer:+.2f} | {status}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)


# ---------------- Main ----------------
def main():
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"[!] Nu pot deschide camera {CAM_INDEX}. Incearca alt CAM_INDEX "
              "sau inchide aplicatiile care folosesc camera.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    # buffer de 1 cadru: altfel DirectShow poate tine cadre vechi si adauga latenta
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    pad = create_gamepad()
    try:
        run_loop(cap, pad)
    except KeyboardInterrupt:
        pass  # Ctrl+C: iesim curat, finally se ocupa de rest
    finally:
        # orice iesire (q, Ctrl+C, exceptie) lasa controllerul in neutru
        send_neutral(pad)
        cap.release()
        cv2.destroyAllWindows()


def run_loop(cap, pad):
    sending = pad is not None
    steer_filter = OneEuroFilter(MIN_CUTOFF, BETA)
    neutral_offset = 0.0
    fps, last = 0.0, time.perf_counter()
    steer, brake, throttle = 0.0, 0.0, 0.0
    read_failures = 0

    with mp_hands.Hands(max_num_hands=2, model_complexity=0,
                        min_detection_confidence=0.6, min_tracking_confidence=0.5) as hands:
        while True:
            ok, frame = cap.read()
            if not ok:
                # unele camere dau cateva cadre goale la pornire (mai ales cu DirectShow)
                read_failures += 1
                if read_failures >= MAX_READ_FAILURES:
                    print("[!] Camera nu mai trimite cadre.")
                    break
                continue
            read_failures = 0
            t0 = time.perf_counter()
            frame = cv2.flip(frame, 1)  # oglinda, ca sa fie intuitiv
            h, w = frame.shape[:2]
            res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if res.multi_hand_landmarks and len(res.multi_hand_landmarks) == 2:
                hs = sorted(res.multi_hand_landmarks, key=lambda x: x.landmark[9].x)
                left, right = hs[0], hs[1]  # stanga/dreapta pe ecran
                lp, rp = palm_center(left, w, h), palm_center(right, w, h)
                cv2.line(frame, tuple(lp.astype(int)), tuple(rp.astype(int)), (0, 255, 255), 3)
                for hand in hs:
                    mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

                raw_angle = steering_angle_deg(lp, rp)
                self_angle = raw_angle - neutral_offset
                value = float(np.clip(self_angle / MAX_ANGLE_DEG, -1, 1))
                value = steer_filter(value, time.perf_counter())
                steer = apply_deadzone(value, DEADZONE)

                braking = is_fist(left) or is_fist(right)
                brake = 1.0 if braking else 0.0
                throttle = 0.0 if braking else AUTO_THROTTLE
            else:
                # mainile nu sunt vizibile: opreste in siguranta
                steer, brake, throttle = 0.0, 0.0, 0.0
                cv2.putText(frame, "Arata ambele maini", (10, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            if pad and sending:
                pad.left_joystick_float(x_value_float=steer, y_value_float=0.0)
                pad.right_trigger_float(value_float=throttle)
                pad.left_trigger_float(value_float=brake)
                pad.update()

            now = time.perf_counter()
            latency_ms = (now - t0) * 1000
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - last, 1e-6))
            last = now
            draw_wheel(frame, steer, brake, throttle, fps, latency_ms, sending, pad is not None)
            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            # butonul X al ferestrei: altfel imshow ar redeschide fereastra
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord("g") and pad is not None:
                sending = not sending
                if not sending:
                    send_neutral(pad)  # nu lasam ultima comanda "blocata" in joc
            if key == ord("c") and res.multi_hand_landmarks and len(res.multi_hand_landmarks) == 2:
                neutral_offset = raw_angle
                steer_filter = OneEuroFilter(MIN_CUTOFF, BETA)


if __name__ == "__main__":
    main()
