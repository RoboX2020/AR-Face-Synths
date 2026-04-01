import numpy as np
import sounddevice as sd
import cv2
import mediapipe as mp
import math
import time
import os
from collections import deque

MODEL_PATH = "face_landmarker.task"

# --- AUDIO SFX SYNTHESIZER ---
class SFXEngine:
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
        self.sounds = []
        self.play_smile_beat = False
        self.beat_phase = 0
        self.beat_samples_per_measure = int(self.sample_rate * (60.0 / 130.0) * 4) # 130 BPM, 4 beats
        
        self.stream = sd.OutputStream(channels=1, samplerate=self.sample_rate, callback=self._audio_callback)
        self.stream.start()

    def play(self, sfx_type):
        """Add a sound effect to the active playing list."""
        if sfx_type == "wink":
            # Coin DING: Two quick high frequency sine bursts
            self.sounds.append({"type": "coin", "phase": 0, "time": 0.0, "duration": 0.4})
        elif sfx_type == "pout":
            # Squeak/Boop: Fast pitch slide up
            self.sounds.append({"type": "squeak", "phase": 0, "time": 0.0, "duration": 0.3})
        elif sfx_type == "brows":
            # Slide Whistle: Slow pitch slide up
            self.sounds.append({"type": "slide", "phase": 0, "time": 0.0, "duration": 0.8})
        elif sfx_type == "surprise":
            # Boing: FM quickly oscillating
            self.sounds.append({"type": "boing", "phase": 0, "time": 0.0, "duration": 0.6})
        elif sfx_type == "shake":
            # Scratch: Filtered noise
            self.sounds.append({"type": "scratch", "phase": 0, "time": 0.0, "duration": 0.4})

    def _audio_callback(self, outdata, frames, time_info, status):
        mixed = np.zeros(frames)
        t_arr = np.linspace(0, frames / self.sample_rate, frames, endpoint=False)
        
        active_sounds = []
        for s in self.sounds:
            t = s["time"] + t_arr
            s["time"] += frames / self.sample_rate
            
            wave = np.zeros(frames)
            if s["type"] == "coin":
                # Freq jumps from B5 to E6
                freqs = np.where(t < 0.1, 987.77, 1318.51)
                env = np.exp(-10.0 * t)
                phase_inc = 2.0 * np.pi * freqs / self.sample_rate
                phases = s["phase"] + np.cumsum(phase_inc)
                s["phase"] = phases[-1] % (2.0 * np.pi)
                wave = np.sin(phases) * env * 0.5
                
            elif s["type"] == "squeak":
                # Freq slides sharply up 400 -> 1200
                freqs = 400.0 + 800.0 * (1.0 - np.exp(-15.0 * t))
                env = np.exp(-8.0 * t)
                phase_inc = 2.0 * np.pi * freqs / self.sample_rate
                phases = s["phase"] + np.cumsum(phase_inc)
                s["phase"] = phases[-1] % (2.0 * np.pi)
                wave = (np.mod(phases, 2*np.pi) / np.pi - 1.0) * env * 0.4 # Sawtooth
                
            elif s["type"] == "slide":
                # Freq slides slowly up 300 -> 800
                freqs = 300.0 + 500.0 * (t / s["duration"])
                # Parabolic envelope (fade in and out)
                env = np.sin(np.pi * np.clip(t / s["duration"], 0, 1))
                phase_inc = 2.0 * np.pi * freqs / self.sample_rate
                phases = s["phase"] + np.cumsum(phase_inc)
                s["phase"] = phases[-1] % (2.0 * np.pi)
                wave = np.sin(phases) * env * 0.5
                
            elif s["type"] == "boing":
                # FM synthesis for "boi-oi-oing"
                mod_freq = 15.0 * np.exp(-2.0 * t)
                mod_idx = 400.0 * np.exp(-3.0 * t)
                phases = s["phase"] + 2.0 * np.pi * 300.0 * t_arr
                s["phase"] = phases[-1] % (2.0 * np.pi)
                wave = np.sin(phases + mod_idx/300.0 * np.sin(2.0*np.pi*mod_freq*t)) * np.exp(-4.0 * t) * 0.6
                
            elif s["type"] == "scratch":
                # Noise burst swept by LPF
                noise = np.random.uniform(-1, 1, frames)
                env = np.exp(-10.0 * t)
                wave = noise * env * 0.4 # Simplified filter as volume drop
            
            mixed += wave
            
            if s["time"] < s["duration"]:
                active_sounds.append(s)
        self.sounds = active_sounds

        # Background Beat (Smile)
        if self.play_smile_beat:
            beat_dur = self.sample_rate * (60.0 / 130.0)
            for i in range(frames):
                self.beat_phase = (self.beat_phase + 1) % self.beat_samples_per_measure
                pos = self.beat_phase % beat_dur
                val = 0.0
                # Simple Kick pattern every beat
                if pos < 4000:
                    val += np.sin(2.0 * np.pi * (100.0 * np.exp(-0.003 * pos)) * pos/self.sample_rate) * np.exp(-0.001 * pos)
                # Hi-hat on offbeats
                if pos > beat_dur/2 and pos < beat_dur/2 + 2000:
                    val += np.random.uniform(-1,1) * np.exp(-0.005 * (pos - beat_dur/2)) * 0.3
                mixed[i] += val * 0.6
                
        # Hard limit
        outdata[:, 0] = np.clip(mixed, -1.0, 1.0)

    def close(self):
        self.stream.stop()
        self.stream.close()

# Geometric Helpers
def dist(p1, p2, w, h):
    return math.sqrt(((p1.x - p2.x)*w)**2 + ((p1.y - p2.y)*h)**2)

class FloatingText:
    def __init__(self, text, x, y, color):
        self.text = text
        self.x = x
        self.y = y
        self.color = color
        self.life = 1.0

    def update(self):
        self.y -= 3  # Float up
        self.life -= 0.03

    def draw(self, img):
        if self.life > 0:
            scale = 1.5 + (1.0 - self.life)
            alpha = max(0, min(255, int(self.life * 255)))
            c = (self.color[0], self.color[1], self.color[2]) # Ideally alpha blended, but we'll fade color briefly
            cv2.putText(img, self.text, (int(self.x), int(self.y)), cv2.FONT_HERSHEY_DUPLEX, 
                        scale, (0,0,0), 6, cv2.LINE_AA)
            cv2.putText(img, self.text, (int(self.x), int(self.y)), cv2.FONT_HERSHEY_DUPLEX, 
                        scale, c, 3, cv2.LINE_AA)

def pt(lm, w, h): return (int(lm.x * w), int(lm.y * h))

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: {MODEL_PATH} not found!")
        return

    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    landmarker = FaceLandmarker.create_from_options(options)
    sfx = SFXEngine()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Cooldown latches
    latch_wink = False
    latch_pout = False
    latch_brows = False
    latch_surprise = False
    
    # Shake history
    nose_history = deque(maxlen=10)

    floating_texts = []

    print("Face SFX Soundboard Started. Press 'q' to quit.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success: break

        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape
        viz = frame.copy()
        
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        
        results = landmarker.detect(mp_image)

        if results.face_landmarks:
            landmarks = results.face_landmarks[0]
            face_w = dist(landmarks[234], landmarks[454], w, h)
            
            # --- 1. Wink Detection (Coin) ---
            eye_l_top, eye_l_bot = landmarks[159], landmarks[145]
            eye_r_top, eye_r_bot = landmarks[386], landmarks[374]
            open_l = dist(eye_l_top, eye_l_bot, w, h) / face_w
            open_r = dist(eye_r_top, eye_r_bot, w, h) / face_w
            
            # If difference in openness is large, one eye is closed while other open
            is_wink = abs(open_l - open_r) > 0.018 and (open_l < 0.01 or open_r < 0.01)
            
            if is_wink and not latch_wink:
                sfx.play("wink")
                wx, wy = (pt(eye_l_top,w,h) if open_l < 0.01 else pt(eye_r_top,w,h))
                floating_texts.append(FloatingText("*DING!*", wx-50, wy-30, (0,255,255)))
                latch_wink = True
            elif not is_wink:
                latch_wink = False

            # --- 2. Eyebrow Raise (Slide Whistle) ---
            l_brow, r_brow = landmarks[105], landmarks[334]
            brow_h = ((dist(eye_l_top, l_brow, w, h) + dist(eye_r_top, r_brow, w, h)) / 2.0) / face_w
            is_brows_up = brow_h > 0.16
            
            # --- 3. Mouth Open ---
            mouth_h = dist(landmarks[13], landmarks[14], w, h) / face_w
            is_mouth_open = mouth_h > 0.1
            
            # --- 4. Surprise (Boing!) ---
            is_surprise = is_brows_up and is_mouth_open
            
            if is_surprise and not latch_surprise:
                sfx.play("surprise")
                floating_texts.append(FloatingText("*BOING!*", int(landmarks[10].x*w)-80, int(landmarks[10].y*h)-50, (0,100,255)))
                latch_surprise = True
                latch_brows = True # Prevent slide whistle from firing at same time
            elif not is_surprise:
                latch_surprise = False
                
            # If brows up BUT mouth closed -> Slide whistle
            if is_brows_up and not is_mouth_open and not latch_brows and not latch_surprise:
                sfx.play("brows")
                floating_texts.append(FloatingText("*swooop*", int(landmarks[10].x*w)-50, int(landmarks[10].y*h)-20, (255,100,255)))
                latch_brows = True
            elif not is_brows_up:
                latch_brows = False

            # --- 5. Pout / Kiss (Squeak) ---
            m_left, m_right = landmarks[61], landmarks[291]
            mouth_w = dist(m_left, m_right, w, h) / face_w
            # Normal width ~0.4. Pout makes it squeeze ~0.25
            is_pout = mouth_w < 0.32 and mouth_h < 0.05
            
            if is_pout and not latch_pout:
                sfx.play("pout")
                floating_texts.append(FloatingText("*SQUEAK!*", int(landmarks[13].x*w)-60, int(landmarks[13].y*h)+40, (0,255,0)))
                latch_pout = True
            elif not is_pout:
                latch_pout = False
                
            # --- 6. Wide Smile (Beat Loop) ---
            is_smile = mouth_w > 0.45
            sfx.play_smile_beat = is_smile
            if is_smile and len(floating_texts) < 2:
                # Add occasional text
                if np.random.rand() < 0.05:
                    floating_texts.append(FloatingText("♪ BEATS ♪", int(landmarks[13].x*w)-50, int(landmarks[13].y*h)+50, (255,0,0)))

            # --- 7. Head Shake (DJ Scratch) ---
            nose = landmarks[1]
            nose_history.append(nose.x)
            if len(nose_history) == 10:
                # Calculate total horizontal travel back and forth over 10 frames
                travel = sum(abs(nose_history[i] - nose_history[i-1]) for i in range(1, 10))
                if travel > 0.08:
                    sfx.play("shake")
                    floating_texts.append(FloatingText("*WIKI-WIKI*", pt(nose,w,h)[0]-80, pt(nose,w,h)[1], (200,200,200)))
                    nose_history.clear() # Reset to prevent spam

        # Render floating texts
        for ft in floating_texts:
            ft.update()
            ft.draw(viz)
        floating_texts = [ft for ft in floating_texts if ft.life > 0]
        
        # Draw UI
        cv2.putText(viz, "ARCADE SOUNDBOARD", (20, 40), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0,0,0), 4)
        cv2.putText(viz, "ARCADE SOUNDBOARD", (20, 40), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255,255,0), 2)
        
        cv2.imshow("Face SFX", viz)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    sfx.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
