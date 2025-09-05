#!/usr/bin/env python3
import argparse
import os
import time
import math
from datetime import datetime
from pathlib import Path
import subprocess
import sys
import re
import threading
import tempfile

import cv2
import numpy as np
import mediapipe as mp

# ----------------------------
# iPhone camera auto-detect via ffmpeg
# ----------------------------
def find_iphone_camera_index_via_ffmpeg():
    """
    Uses ffmpeg to list AVFoundation devices and returns the index of 'iPhone' camera if found.
    Requires a system ffmpeg (brew install ffmpeg).
    """
    try:
        cmd = ["ffmpeg", "-f", "avfoundation", "-list_devices", "true", "-i", ""]
        proc = subprocess.run(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        text = proc.stderr + proc.stdout
        matches = re.findall(r"\[(\d+)\]\s+(.*iPhone.*Camera.*)", text, flags=re.IGNORECASE)
        if matches:
            return int(matches[0][0])
    except Exception as e:
        print(f"[Warn] Could not query FFmpeg devices: {e}", file=sys.stderr)
    return None

# ----------------------------
# Output path helper
# ----------------------------
def make_output_path(custom_path: str) -> str:
    """
    Decide where to save recordings:
    - If --outfile is passed, use that.
    - Otherwise, save into the current working directory with a timestamped filename.
    """
    if custom_path:
        base = os.path.expanduser(custom_path)
        folder = os.path.dirname(base) or "."
        os.makedirs(folder, exist_ok=True)
        return base if base.lower().endswith(".mp4") else base + ".mp4"
    cwd = os.getcwd()
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return os.path.join(cwd, f"iphone_record_{ts}.mp4")
# ----------------------------
# HUD / overlay helpers
# ----------------------------
def draw_recording_status(frame, recording: bool):
    h, w = frame.shape[:2]
    label = "REC" if recording else "PAUSED"
    color = (0, 0, 255) if recording else (200, 200, 200)
    cv2.circle(frame, (24, 24), 10, color, thickness=-1, lineType=cv2.LINE_AA)
    cv2.putText(frame, label, (44, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

def draw_caption_bar(frame, text, pad=8):
    """Draw a semi-transparent bar at the bottom with white text."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.7
    thickness = 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    bar_h = th + pad * 2
    y0 = h - bar_h

    cv2.rectangle(overlay, (0, y0), (w, h), (0, 0, 0), -1)
    alpha = 0.45
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    x = max(10, (w - tw) // 2)
    y = y0 + pad + th
    cv2.putText(frame, text, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

# ----------------------------
# Gesture classification (lightweight heuristics)
# ----------------------------
def classify_hand_gesture(hand_landmarks, handedness_label=""):
    """
    Heuristic hand gesture classifier:
    - Open Palm: many fingers extended + far from wrist
    - Fist: fingers curled + close to wrist
    - Pointing: index extended, others curled
    - Thumbs Up: thumb extended, others curled
    """
    lm = hand_landmarks.landmark
    WRIST = lm[0]
    tips = [lm[4], lm[8], lm[12], lm[16], lm[20]]           # thumb/index/middle/ring/pinky tips
    pipj = [lm[3], lm[6], lm[10], lm[14], lm[18]]           # joints before tips

    dists = [math.hypot(t.x - WRIST.x, t.y - WRIST.y) for t in tips]
    avg_dist = sum(dists) / len(dists)

    curls = []
    for t, p in zip(tips, pipj):
        dt = math.hypot(t.x - WRIST.x, t.y - WRIST.y)
        dp = math.hypot(p.x - WRIST.x, p.y - WRIST.y)
        curls.append(dt < dp)
    extended = [not c for c in curls]
    thumb_ext, idx_ext, mid_ext, ring_ext, pink_ext = extended

    if idx_ext and not (mid_ext or ring_ext or pink_ext) and not thumb_ext:
        return f"{handedness_label}Pointing"
    if thumb_ext and not (idx_ext or mid_ext or ring_ext or pink_ext):
        return f"{handedness_label}Thumbs Up"
    if sum(extended) >= 4 and avg_dist > 0.12:
        return f"{handedness_label}Open Palm"
    if sum(extended) <= 1 and avg_dist < 0.10:
        return f"{handedness_label}Fist"
    return f"{handedness_label}Neutral"

# ----------------------------
# Transcription helpers (optional)
# ----------------------------
def seconds_to_srt(ts: float) -> str:
    h = int(ts // 3600)
    m = int((ts % 3600) // 60)
    s = int(ts % 60)
    ms = int(round((ts - int(ts)) * 1000))
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

def transcribe_with_mlx_whisper(video_path: str, model="mlx-community/whisper-small", make_srt=True):
    """
    Transcribe an MP4 using MLX Whisper (Apple Silicon). Requires:
      brew install ffmpeg
      pip install mlx-whisper
    Writes `*.txt` and (optionally) `*.srt`.
    """
    try:
        import mlx_whisper
    except Exception as e:
        raise RuntimeError("mlx-whisper not installed. Run: pip install mlx-whisper") from e

    base = Path(video_path).with_suffix("")
    out_txt = str(base) + ".txt"
    out_srt = str(base) + ".srt"

    print(f"[Transcribe] Loading model: {model}")
    result = mlx_whisper.transcribe(video_path, path_or_hf_repo=model, word_timestamps=False)

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(result.get("text", "").strip() + "\n")
    print(f"[Transcribe] Wrote transcript: {out_txt}")

    if make_srt and "segments" in result:
        with open(out_srt, "w", encoding="utf-8") as f:
            for i, seg in enumerate(result["segments"], start=1):
                start = seconds_to_srt(seg["start"])
                end = seconds_to_srt(seg["end"])
                text = seg["text"].strip()
                f.write(f"{i}\n{start} --> {end}\n{text}\n\n")
        print(f"[Transcribe] Wrote captions:   {out_srt}")

def auto_transcribe(video_path: str, model="mlx-community/whisper-small"):
    try:
        transcribe_with_mlx_whisper(video_path, model=model, make_srt=True)
    except Exception as e:
        print(f"[Transcribe] Failed: {e}\n"
              f"Tip: ensure ffmpeg is installed and `pip install mlx-whisper` in your env.")

# ----------------------------
# FastVLM continuous caption worker (no fixed interval)
# ----------------------------
class FastVLMCaptioner:
    def __init__(self, model_path: str, predict_py: str, prompt: str | None = None,
                 max_side: int = 512):
        """
        Runs apple/ml-fastvlm predict.py in a loop; always uses the latest frame.
        For maximum throughput you could import the model in-process, but this
        subprocess approach stays closest to the official repo entry point.
        """
        self.model_path = model_path
        self.predict_py = predict_py
        self.prompt = prompt or "Describe what is happening in this video frame in one short sentence."
        self.max_side = max_side
        self._latest_text = ""
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread = None

    def start(self, get_bgr_frame_callable):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, args=(get_bgr_frame_callable,), daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)

    def latest_text(self) -> str:
        with self._lock:
            return self._latest_text

    def _resize_for_model(self, img_bgr):
        h, w = img_bgr.shape[:2]
        scale = self.max_side / max(h, w)
        if scale < 1.0:
            nh, nw = int(h * scale), int(w * scale)
            return cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
        return img_bgr

    def _run(self, get_frame):
        # Tight loop: fetch latest frame, run predict.py, update text, repeat.
        while not self._stop.is_set():
            try:
                frame = get_frame()
                if frame is None:
                    time.sleep(0.005)  # yield briefly if no frame yet
                    continue

                small = self._resize_for_model(frame)

                # Write temp PNG for predict.py
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    tmp_path = tmp.name
                    cv2.imwrite(tmp_path, small)

                # Call Apple’s predict.py
                cmd = [
                    sys.executable, self.predict_py,
                    "--model-path", self.model_path,
                    "--image-file", tmp_path,
                    "--prompt", self.prompt
                ]
                proc = subprocess.run(cmd, capture_output=True, text=True)
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

                text = (proc.stdout or "").strip() or (proc.stderr or "").strip()
                if len(text) > 160:
                    text = text[:157] + "..."
                with self._lock:
                    self._latest_text = text

            except Exception as e:
                with self._lock:
                    self._latest_text = f"(FastVLM error: {e})"
            # No sleep -> model latency sets the cadence

# ----------------------------
# CLI
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="iPhone camera + MediaPipe overlay + continuous VLM captions + recorder. "
                    "SPACE to start/stop recording, Q to quit."
    )
    p.add_argument("--cam-index", type=int, default=None, help="Camera index (try 0/1/2/3).")
    p.add_argument("--auto-iphone", action="store_true", help="Auto-select 'iPhone Camera' via ffmpeg.")
    p.add_argument("--list-cams", action="store_true", help="Probe indices 0..5 with AVFoundation and exit.")
    p.add_argument("--width", type=int, default=1280, help="Capture width.")
    p.add_argument("--height", type=int, default=720, help="Capture height.")
    p.add_argument("--fps", type=int, default=30, help="Capture FPS.")
    p.add_argument("--outfile", type=str, default="", help="Output mp4 path (default: current working directory).")
    p.add_argument("--show-fps", action="store_true", help="Show live FPS overlay.")
    p.add_argument("--face-mesh", action="store_true", help="Use Face Mesh landmarks instead of face boxes.")
    p.add_argument("--auto-transcribe", action="store_true", help="Auto-transcribe video after stopping recording.")

    # FastVLM
    p.add_argument("--fastvlm-model", type=str, default="",
                   help="Path to FastVLM MLX checkpoint dir (or set FASTVLM_MODEL env var)")
    p.add_argument("--fastvlm-repo", type=str, default="",
                   help="Path to the cloned apple/ml-fastvlm repo (or set FASTVLM_REPO env var)")
    p.add_argument("--vlm-prompt", type=str, default="",
                   help="Custom prompt for FastVLM video description")
    p.add_argument("--vlm-maxside", type=int, default=512,
                   help="Resize longest image side before VLM to this many pixels (speed/quality tradeoff).")

    p.add_argument("--rotate90", action="store_true", help="Rotate frame 90° clockwise (if Continuity is rotated).")
    return p.parse_args()

# ----------------------------
# Main
# ----------------------------
def main():
    args = parse_args()

    # Optional: list cams quickly and exit
    if args.list_cams:
        print("Probing camera indices 0..5 via AVFoundation...")
        for i in range(6):
            cap = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)
            ok = cap.isOpened()
            ok2 = False
            if ok:
                ok2, _ = cap.read()
            print(f"idx {i}: opened={ok} frame={ok2}")
            cap.release()
        return

    # Choose camera index
    cam_index = args.cam_index
    if cam_index is None and args.auto_iphone:
        cam_index = find_iphone_camera_index_via_ffmpeg()
        if cam_index is None:
            print("[Info] Could not auto-detect iPhone camera via ffmpeg. Falling back to index 0.")
            cam_index = 0
    if cam_index is None:
        cam_index = 0

    # Open camera (prefer AVFoundation on macOS)
    cap = cv2.VideoCapture(cam_index, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera at index {cam_index}. Try a different --cam-index.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS) or args.fps
    print(f"[Camera] Using index={cam_index} at {actual_w}x{actual_h} @ {actual_fps:.1f} fps")

    # MediaPipe setup
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands
    mp_face = mp.solutions.face_detection
    mp_face_mesh = mp.solutions.face_mesh

    hands = mp_hands.Hands(
        model_complexity=1,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    face_detector = None
    face_mesh = None
    if args.face_mesh:
        face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=2,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    else:
        face_detector = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)

    out_path = make_output_path(args.outfile)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = None
    recording = False

    prev_t = time.time()
    fps_disp = 0.0
    fps_alpha = 0.05

    window_name = "iPhone + MediaPipe + FastVLM (SPACE: start/stop, Q: quit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # Set up FastVLM continuous captioner (optional)
    fastvlm = None
    last_frame_lock = threading.Lock()
    last_frame = {"bgr": None}

    # Allow env vars as fallback to CLI
    fastvlm_repo_arg = args.fastvlm_repo or os.environ.get("FASTVLM_REPO", "")
    fastvlm_model_arg = args.fastvlm_model or os.environ.get("FASTVLM_MODEL", "")

    if fastvlm_model_arg:
        repo = os.path.expanduser(fastvlm_repo_arg)
        model_path = os.path.expanduser(fastvlm_model_arg)
        predict_py = os.path.join(repo, "predict.py")
        if not os.path.isfile(predict_py):
            print(f"[FastVLM] predict.py not found at {predict_py}. Set --fastvlm-repo correctly.")
        elif not os.path.isdir(model_path):
            print(f"[FastVLM] Model path not found: {model_path}. Set --fastvlm-model to the checkpoint dir.")
        else:
            def _get_latest_frame():
                with last_frame_lock:
                    return None if last_frame["bgr"] is None else last_frame["bgr"].copy()

            fastvlm = FastVLMCaptioner(
                model_path=model_path,
                predict_py=predict_py,
                prompt=(args.vlm_prompt or "Describe what is happening in this video frame in one short sentence."),
                max_side=int(args.vlm_maxside)
            )
            fastvlm.start(_get_latest_frame)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Frame grab failed. Trying again...")
                continue

            if args.rotate90:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

            # stash frame for VLM worker
            if fastvlm is not None:
                with last_frame_lock:
                    last_frame["bgr"] = frame

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # --- Hands ---
            hand_results = hands.process(rgb)
            if hand_results.multi_hand_landmarks:
                for hls in hand_results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, hls, mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )

            # --- Faces ---
            face_count = 0
            if face_mesh is not None:
                mesh_results = face_mesh.process(rgb)
                if mesh_results.multi_face_landmarks:
                    face_count = len(mesh_results.multi_face_landmarks)
                    for fl in mesh_results.multi_face_landmarks:
                        mp_drawing.draw_landmarks(
                            image=frame, landmark_list=fl,
                            connections=mp_face_mesh.FACEMESH_TESSELATION,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                        )
                        mp_drawing.draw_landmarks(
                            image=frame, landmark_list=fl,
                            connections=mp_face_mesh.FACEMESH_CONTOURS,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
                        )
            else:
                face_results = face_detector.process(rgb)
                if face_results.detections:
                    face_count = len(face_results.detections)
                    for det in face_results.detections:
                        mp_drawing.draw_detection(frame, det)

            # --- Gesture labels for caption ---
            hand_labels = []
            if hand_results and hand_results.multi_hand_landmarks:
                handed = []
                try:
                    handed = [h.classification[0].label for h in hand_results.multi_handedness]
                except Exception:
                    handed = [""] * len(hand_results.multi_hand_landmarks)

                for hlm, hand_lab in zip(hand_results.multi_hand_landmarks, handed):
                    label_prefix = f"{hand_lab} " if hand_lab else ""
                    hand_labels.append(classify_hand_gesture(hlm, handedness_label=label_prefix))

            gestures = ", ".join(hand_labels[:2]) if hand_labels else "None"
            caption = f"Faces: {face_count} | Hands: {len(hand_labels)} | Gestures: {gestures}"

            # Append FastVLM text continuously
            if fastvlm is not None:
                vlm_text = fastvlm.latest_text()
                if vlm_text:
                    caption = caption + "  •  " + vlm_text

            draw_caption_bar(frame, caption)

            # HUD: recording + FPS
            draw_recording_status(frame, recording)
            now = time.time()
            inst_fps = 1.0 / max(now - prev_t, 1e-6)
            prev_t = now
            fps_disp = (1 - fps_alpha) * fps_disp + fps_alpha * inst_fps if fps_disp > 0 else inst_fps
            if args.show_fps:
                cv2.putText(frame, f"{fps_disp:.1f} FPS", (10, frame.shape[0] - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

            # Start writer lazily on first record
            if recording:
                if writer is None:
                    writer = cv2.VideoWriter(
                        out_path, fourcc, actual_fps if actual_fps > 0 else args.fps,
                        (frame.shape[1], frame.shape[0])
                    )
                    if not writer.isOpened():
                        raise SystemExit(f"Could not open VideoWriter at: {out_path}")
                    print(f"[Recording] Writing to {out_path}")
                writer.write(frame)

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF

            # SPACE toggles recording
            if key == 32:  # space
                recording = not recording
                if not recording:
                    if writer is not None:
                        writer.release()
                        writer = None
                        print(f"[Recording] Stopped. Saved: {out_path}")
                        if args.auto_transcribe:
                            auto_transcribe(out_path)
                    # New filename if we start again later in same run
                    if not args.outfile:
                        out_path = make_output_path(args.outfile)
                else:
                    print("[Recording] Started.")

            # Q or ESC quits
            if key in (ord('q'), 27):
                break

    finally:
        if writer is not None:
            writer.release()
            print(f"[Recording] Finalized: {out_path}")
            if args.auto_transcribe:
                auto_transcribe(out_path)
        cap.release()
        cv2.destroyAllWindows()
        # stop VLM worker
        if fastvlm is not None:
            fastvlm.stop()

if __name__ == "__main__":
    main()
