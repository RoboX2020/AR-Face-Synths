import numpy as np
import sounddevice as sd
import cv2
import mediapipe as mp
import math
import os
import time
from mediapipe.framework.formats import landmark_pb2

MODEL_PATH = "face_landmarker.task"

# Frequencies for Arpeggios
BASE_FREQ = 130.81 # C3

def note_freq(semitones):
    return BASE_FREQ * (2 ** (semitones / 12.0))

MAJOR_ARP = [note_freq(0), note_freq(4), note_freq(7), note_freq(12), note_freq(7), note_freq(4)]
MINOR_ARP = [note_freq(0), note_freq(3), note_freq(7), note_freq(12), note_freq(7), note_freq(3)]

class ArpSynth:
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
        
        # Core synth params
        self.tempo_bpm = 120.0 
        self.pitch_shift = 1.0 
        self.is_minor = False
        
        # New FX params
        self.pan = 0.5 # 0.0 (Left) to 1.0 (Right)
        self.filter_cutoff = 1.0 # 0.0 to 1.0 (alpha for LPF)
        self.trigger_kick = False
        
        self.phase = 0.0
        self.sample_count = 0
        self.arp_index = 0
        self.volume = 0.8
        
        self.current_freq = MAJOR_ARP[0]
        self.target_freq = MAJOR_ARP[0]
        
        # Audio state
        self.lpf_state_left = 0.0
        self.lpf_state_right = 0.0
        self.kick_phase = 0.0
        self.kick_env = 0.0
        
        self.stream = sd.OutputStream(
            channels=2, # Stereo upgraded
            samplerate=self.sample_rate,
            callback=self._audio_callback
        )
        self.stream.start()

    def set_params(self, tempo, pitch, minor, pan, filter_c):
        self.tempo_bpm = max(60.0, min(800.0, tempo))
        self.pitch_shift = pitch
        self.is_minor = minor
        self.pan = max(0.0, min(1.0, pan))
        # Map filter control to LPF alpha [0.05 (muffled) to 1.0 (bright/wah open)]
        self.filter_cutoff = max(0.05, min(1.0, filter_c))

    def play_kick(self):
        self.trigger_kick = True

    def _audio_callback(self, outdata, frames, time_info, status):
        samples_per_step = int(self.sample_rate * 60.0 / self.tempo_bpm)
        
        wave_l = np.zeros(frames)
        wave_r = np.zeros(frames)
        
        arp_seq = MINOR_ARP if self.is_minor else MAJOR_ARP
        
        if self.trigger_kick:
            self.kick_env = 1.0
            self.kick_phase = 0.0
            self.trigger_kick = False
            
        for i in range(frames):
            if self.sample_count >= samples_per_step:
                self.sample_count = 0
                self.arp_index = (self.arp_index + 1) % len(arp_seq)
            
            # --- SYNTH VOICE ---
            base_f = arp_seq[self.arp_index] * self.pitch_shift
            # Portamento glide
            self.current_freq += (base_f - self.current_freq) * 0.05
            
            dPhase = 2.0 * np.pi * self.current_freq / self.sample_rate
            self.phase = (self.phase + dPhase) % (2.0 * np.pi)
            
            # Arpeggiator Envelope
            x = self.sample_count / samples_per_step
            env = np.exp(-5.0 * x) * self.volume
            
            # Waveform: Saw + Square
            raw_wave = np.sin(self.phase) * 0.6 + np.sin(self.phase * 2) * 0.3 + np.sin(self.phase * 3) * 0.1
            synth_sample = raw_wave * env
            
            # --- LOW PASS FILTER (Wah-wah on jaw drop) ---
            self.lpf_state_left += self.filter_cutoff * (synth_sample - self.lpf_state_left)
            filt_sample = self.lpf_state_left
            
            # --- DRUM KICK (Blink) ---
            kick_sample = 0.0
            if self.kick_env > 0.001:
                # Fast decaying pitch and amp
                k_freq = 50.0 + 300.0 * self.kick_env
                dkPhase = 2.0 * np.pi * k_freq / self.sample_rate
                self.kick_phase = (self.kick_phase + dkPhase) % (2.0 * np.pi)
                kick_sample = np.sin(self.kick_phase) * self.kick_env
                self.kick_env *= 0.999 # Smooth decay
            
            mixed = filt_sample + kick_sample * 0.8
            
            # --- PANNING ---
            # Equal power panning
            pan_left = math.cos(self.pan * math.pi / 2.0)
            pan_right = math.sin(self.pan * math.pi / 2.0)
            
            wave_l[i] = mixed * pan_left
            wave_r[i] = mixed * pan_right
            
            self.sample_count += 1
            
        outdata[:, 0] = wave_l
        outdata[:, 1] = wave_r

    def close(self):
        self.stream.stop()
        self.stream.close()

# Geometric Helpers
def dist(p1, p2, w, h):
    return math.sqrt(((p1.x - p2.x)*w)**2 + ((p1.y - p2.y)*h)**2)

def draw_rounded_rect(img, pt1, pt2, color, thickness=cv2.FILLED, r=10):
    x1, y1 = pt1
    x2, y2 = pt2
    if thickness == cv2.FILLED:
        cv2.rectangle(img, (x1+r, y1), (x2-r, y2), color, cv2.FILLED)
        cv2.rectangle(img, (x1, y1+r), (x2, y2-r), color, cv2.FILLED)
        cv2.circle(img, (x1+r, y1+r), r, color, cv2.FILLED)
        cv2.circle(img, (x2-r, y1+r), r, color, cv2.FILLED)
        cv2.circle(img, (x1+r, y2-r), r, color, cv2.FILLED)
        cv2.circle(img, (x2-r, y2-r), r, color, cv2.FILLED)
    else:
        cv2.line(img, (x1+r, y1), (x2-r, y1), color, thickness)
        cv2.line(img, (x1+r, y2), (x2-r, y2), color, thickness)
        cv2.line(img, (x1, y1+r), (x1, y2-r), color, thickness)
        cv2.line(img, (x2, y1+r), (x2, y2-r), color, thickness)
        cv2.ellipse(img, (x1+r, y1+r), (r, r), 180, 0, 90, color, thickness)
        cv2.ellipse(img, (x2-r, y1+r), (r, r), 270, 0, 90, color, thickness)
        cv2.ellipse(img, (x1+r, y2-r), (r, r), 90, 0, 90, color, thickness)
        cv2.ellipse(img, (x2-r, y2-r), (r, r), 0, 0, 90, color, thickness)

def draw_bar(img, x, y, w, h, val, min_v, max_v, title, color):
    val = max(min_v, min(max_v, val))
    pct = (val - min_v) / (max_v - min_v) if max_v > min_v else 0
    draw_rounded_rect(img, (x, y), (x+w, y+h), (40, 40, 40), cv2.FILLED, 5)
    fill_w = int(w * pct)
    if fill_w > 0:
        draw_rounded_rect(img, (x, y), (x+fill_w, y+h), color, cv2.FILLED, 5)
    cv2.putText(img, f"{title}: {val:.2f}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

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
    synth = ArpSynth()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    mp_drawing = mp.solutions.drawing_utils
    mp_face_mesh = mp.solutions.face_mesh

    # Smoothing states
    smooth_smile = 0.0
    smooth_brow = 0.0
    smooth_frown = 0.0
    smooth_jaw = 0.0
    smooth_roll = 0.0
    
    blink_latch = False

    print("Face Music Synth FULL ENGAGED. Press 'q' to quit.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        # Cyberpunk dark vignette
        frame = cv2.convertScaleAbs(frame, alpha=0.6, beta=10)
        h, w, c = frame.shape
        
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        
        results = landmarker.detect(mp_image)
        overlay = np.zeros_like(frame, dtype=np.uint8)

        if results.face_landmarks:
            landmarks = results.face_landmarks[0]
            
            # --- FEATURE EXTRACTION ---
            # Distance scale constraint
            face_width = dist(landmarks[234], landmarks[454], w, h)
            
            # 1. Smile (Tempo)
            m_left, m_right = landmarks[61], landmarks[291]
            norm_smile = dist(m_left, m_right, w, h) / face_width
            
            # 2. EYebrows (Pitch)
            l_eye, l_brow = landmarks[159], landmarks[105]
            r_eye, r_brow = landmarks[386], landmarks[334]
            norm_brow = ((dist(l_eye, l_brow, w, h) + dist(r_eye, r_brow, w, h)) / 2.0) / face_width
            
            # 3. Frown / Furrow (Minor Mode)
            in_brow_l, in_brow_r = landmarks[107], landmarks[336]
            norm_frown = dist(in_brow_l, in_brow_r, w, h) / face_width
            
            # 4. NEW: Jaw Drop (Wah-Wah Filter)
            lip_top, lip_bot = landmarks[13], landmarks[14]
            norm_jaw = dist(lip_top, lip_bot, w, h) / face_width
            
            # 5. NEW: Head Tilt Roll (Panning)
            # Roll angle calculated from eyeline (Left Eye Outer 33 vs Right Eye Outer 263)
            eye_l_out, eye_r_out = landmarks[33], landmarks[263]
            # y differs -> roll. dy/dx
            dx = (eye_r_out.x - eye_l_out.x) * w
            dy = (eye_r_out.y - eye_l_out.y) * h
            roll_angle = math.atan2(dy, dx) # Approx radians
            
            # 6. NEW: Eye Blink (Percussion)
            # Vertical eye distance vs horizontal
            eye_l_top, eye_l_bot = landmarks[159], landmarks[145]
            eye_r_top, eye_r_bot = landmarks[386], landmarks[374]
            eye_openness = (dist(eye_l_top, eye_l_bot, w, h) + dist(eye_r_top, eye_r_bot, w, h)) / (2.0 * face_width)
            
            is_blinking = eye_openness < 0.02
            if is_blinking and not blink_latch:
                synth.play_kick()
                blink_latch = True
            elif not is_blinking:
                blink_latch = False
            
            # Smoothing (EMA)
            smooth_smile = smooth_smile * 0.8 + norm_smile * 0.2
            smooth_brow = smooth_brow * 0.8 + norm_brow * 0.2
            smooth_frown = smooth_frown * 0.8 + norm_frown * 0.2
            smooth_jaw = smooth_jaw * 0.8 + norm_jaw * 0.2
            smooth_roll = smooth_roll * 0.8 + roll_angle * 0.2
            
            # --- MAPPING CALCULATIONS ---
            smile_t = max(0.0, min(1.0, (smooth_smile - 0.35) / 0.15))
            bpm = 100.0 + smile_t * 300.0
            
            brow_t = max(0.0, min(1.0, (smooth_brow - 0.10) / 0.08))
            pitch = 1.0 + brow_t * 1.5
            
            is_minor = smooth_frown < 0.11
            
            jaw_t = max(0.0, min(1.0, (smooth_jaw - 0.02) / 0.15)) # 0.0 (closed) to 1.0 (wide open)
            filter_cutoff = 0.05 + jaw_t * 0.95 # Muffled to Bright
            
            # Roll angle mapping [-0.5 rad to 0.5 rad] -> [0.0 (Left) to 1.0 (Right)]
            # Actually if head tilts right, dy is positive, angle positive.
            roll_t = max(-0.5, min(0.5, smooth_roll))
            pan = 0.5 + roll_t # 0.5 is center
            
            synth.set_params(bpm, pitch, is_minor, pan, filter_cutoff)
            
            # --- VISUALS ---
            # Dynamic Pulse from Arpeggiator Extracted from Synth Time
            samples_per_step = int(synth.sample_rate * 60.0 / synth.tempo_bpm)
            pulse_pct = max(0.0, min(1.0, (samples_per_step - synth.sample_count) / samples_per_step))
            
            color_pulse = (int(0 * pulse_pct), int(255 * pulse_pct), int(255 * pulse_pct))
            if is_minor:
                color_pulse = (int(0 * pulse_pct), int(0 * pulse_pct), int(255 * pulse_pct))
                
            # Construct proto list strictly for drawing tesselation
            face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            face_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) for lm in landmarks
            ])
            
            drawing_spec = mp_drawing.DrawingSpec(color=color_pulse, thickness=1, circle_radius=0)
            
            mp_drawing.draw_landmarks(
                image=overlay,
                landmark_list=face_landmarks_proto,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=drawing_spec
            )
            
            # Draw Glassmorphic UI Info Panel
            panel_x, panel_y = 20, 20
            panel_w, panel_h = 320, 320
            
            draw_rounded_rect(overlay, (panel_x, panel_y), (panel_x+panel_w, panel_y+panel_h), (30, 30, 40), cv2.FILLED, 15)
            draw_rounded_rect(overlay, (panel_x, panel_y), (panel_x+panel_w, panel_y+panel_h), (200, 200, 200), 2, 15)
            
            cv2.putText(overlay, "FACE SYNTH X-TREME", (panel_x+20, panel_y+30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            draw_bar(overlay, panel_x+20, panel_y+50, 280, 15, smile_t, 0.0, 1.0, f"Tempo (SMILE) BPM=[{bpm:.0f}]", (0, 255, 255))
            draw_bar(overlay, panel_x+20, panel_y+90, 280, 15, brow_t, 0.0, 1.0, f"Pitch (BROWS) Mult=[{pitch:.1f}x]", (255, 100, 255))
            draw_bar(overlay, panel_x+20, panel_y+130, 280, 15, filter_cutoff, 0.0, 1.0, f"Wah-Wah Filter (JAW)", (0, 100, 255))
            draw_bar(overlay, panel_x+20, panel_y+170, 280, 15, pan, 0.0, 1.0, f"Stereo Pan (HEAD TILT)", (255, 255, 0))
            
            mood_text = "DARK (MINOR SCALE)" if is_minor else "HAPPY (MAJOR SCALE)"
            mood_color = (0, 0, 255) if is_minor else (0, 255, 0)
            cv2.putText(overlay, f"Scale  : {mood_text}", (panel_x+20, panel_y+220), cv2.FONT_HERSHEY_SIMPLEX, 0.6, mood_color, 2)
            
            kick_msg = ">>> KICK <<<" if synth.kick_env > 0.05 else ""
            cv2.putText(overlay, kick_msg, (panel_x+20, panel_y+260), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Master pulse circle
            r = int(50 * pulse_pct * synth.volume)
            cv2.circle(overlay, (w - 100, h - 100), max(1, r), mood_color, -1, cv2.LINE_AA)
            cv2.circle(overlay, (w - 100, h - 100), 50 + int(synth.kick_env*20), (255, 255, 255), 2, cv2.LINE_AA)

        # Blend UI over frame
        final_frame = cv2.addWeighted(overlay, 0.9, frame, 1.0, 0)

        cv2.imshow("Face Music Synth Phase 2", final_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    synth.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
