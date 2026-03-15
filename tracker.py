import cv2
import os
import sys
from pathlib import Path
from datetime import datetime
from datetime import timedelta
from ultralytics import YOLO


class Tracker:
    def __init__(self, model_path='yolov8n.pt'):
        # Handle PyInstaller path resolution
        if hasattr(sys, '_MEIPASS'):
            model_path = os.path.join(sys._MEIPASS, model_path)
        self.model = YOLO(model_path)

    # ── Helpers ───────────────────────────────────────────────────────────

    def extract_first_frame(self, video_path):
        """Returns the first frame as RGB, or (None, error_str)."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None, "Error: Could not open video file."
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return None, "Error: Could not read first frame."
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), None

    def detect_objects(self, frame):
        """Plain YOLO detection (no tracking state). Returns list of dicts."""
        results = self.model(frame, verbose=False)
        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls_id = int(box.cls[0].item())
                detections.append({
                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                    'confidence': box.conf[0].item(),
                    'class': cls_id,
                    'class_name': r.names[cls_id],
                })
        return detections

    def calculate_iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
        inter = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        aA = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        aB = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        return inter / float(aA + aB - inter)

    # ── Main tracking ─────────────────────────────────────────────────────

    def process_video(self, video_path, target_bbox,
                      progress_callback=None, frame_callback=None,
                      stop_event=None, frame_skip=3, start_frame=0):
        """
        Tracks target_bbox across the video using YOLO ByteTrack + frame skipping.

        frame_skip  : Run YOLO only every N frames. Skipped frames reuse the
                      last known / linearly-extrapolated bbox — making processing
                      ~N× faster while keeping accuracy high for slow/normal
                      moving objects (default = 3).

        MISS_PATIENCE : Number of *processed* (YOLO-checked) frames with no
                        detection before declaring disappearance.
        EDGE_PATIENCE : Processed frames where bbox is only at edge before exit.
        """

        MISS_PATIENCE = 30    # processed frames (actual YOLO calls, not raw frames)
        EDGE_PATIENCE = 5

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"error": "Could not open video."}

        fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        orig_w       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        max_dim = 640
        scale   = (min(max_dim / orig_w, max_dim / orig_h)
                   if (orig_w > max_dim or orig_h > max_dim) else 1.0)
        proc_w  = int(orig_w * scale)
        proc_h  = int(orig_h * scale)

        proc_h  = int(orig_h * scale)

        # Scale user bbox to processing resolution
        ux1, uy1, ux2, uy2 = target_bbox
        sx1, sy1 = int(ux1 * scale), int(uy1 * scale)
        sx2, sy2 = int(ux2 * scale), int(uy2 * scale)
        
        # Seek to starting frame if not 0
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # ── Frame 0 / Initial Frame : lock onto a ByteTrack track ID ──────
        ret, frame = cap.read()
        if not ret:
            cap.release()
            return {"error": "Could not read video."}

        first_frame_bgr = frame.copy()
        pf0 = cv2.resize(frame, (proc_w, proc_h))

        res0 = self.model.track(pf0, persist=True, verbose=False)

        target_track_id = None
        target_class_id = None
        current_xyxy    = (sx1, sy1, sx2, sy2)

        if res0 and res0[0].boxes is not None and len(res0[0].boxes):
            user_cx = (sx1 + sx2) // 2
            user_cy = (sy1 + sy2) // 2
            best_iou   = -1.0
            best_cdist = float('inf')
            best_box   = None

            for box in res0[0].boxes:
                bx1, by1, bx2, by2 = [int(v) for v in box.xyxy[0].tolist()]
                iou = self.calculate_iou((sx1, sy1, sx2, sy2),
                                         (bx1, by1, bx2, by2))
                cd  = abs((bx1 + bx2) // 2 - user_cx) + \
                      abs((by1 + by2) // 2 - user_cy)
                if iou > best_iou or (iou == best_iou and cd < best_cdist):
                    best_iou, best_cdist, best_box = iou, cd, box

            if best_box is not None:
                bx1, by1, bx2, by2 = [int(v) for v in best_box.xyxy[0].tolist()]
                target_class_id = int(best_box.cls[0].item())
                if best_box.id is not None:
                    target_track_id = int(best_box.id[0].item())
                current_xyxy = (bx1, by1, bx2, by2)
                print(f"[Frame 0] Locked → track_id={target_track_id}, "
                      f"class={res0[0].names[target_class_id]} (IoU={best_iou:.2f})")

        edge_margin = max(5, int(min(proc_w, proc_h) * 0.015))

        def _at_edge(b):
            return (b[0] <= edge_margin or b[1] <= edge_margin or
                    b[2] >= proc_w - edge_margin or
                    b[3] >= proc_h - edge_margin)

        def _cdist(b1, b2):
            return (abs((b1[0] + b1[2]) // 2 - (b2[0] + b2[2]) // 2) +
                    abs((b1[1] + b1[3]) // 2 - (b2[1] + b2[3]) // 2))

        # Velocity tracking for extrapolation during skipped frames
        prev_xyxy = current_xyxy
        vel_x = 0
        vel_y = 0

        frame_idx               = start_frame + 1
        disappeared             = False
        disappearance_frame_idx = 0
        miss_streak             = 0
        edge_streak             = 0

        # ── Per-frame loop ─────────────────────────────────────────────────
        while cap.isOpened():
            if stop_event is not None and stop_event.is_set():
                cap.release()
                return {'stopped': True, 'last_frame_idx': frame_idx - 1}

            ret, frame = cap.read()
            if not ret:
                break

            current_bgr = frame.copy()

            run_yolo = (frame_idx % frame_skip == 0)

            if run_yolo:
                pf = cv2.resize(frame, (proc_w, proc_h))
                res = self.model.track(pf, persist=True, verbose=False)

                found_xyxy      = None
                same_class_dets = []

                if res and res[0].boxes is not None and len(res[0].boxes):
                    for box in res[0].boxes:
                        cls_id = int(box.cls[0].item())
                        bx1, by1, bx2, by2 = [int(v) for v in box.xyxy[0].tolist()]
                        det = (bx1, by1, bx2, by2)

                        if target_class_id is None or cls_id == target_class_id:
                            same_class_dets.append(det)

                        if (found_xyxy is None and
                                target_track_id is not None and
                                box.id is not None and
                                int(box.id[0].item()) == target_track_id):
                            found_xyxy = det

                    if found_xyxy is None and same_class_dets:
                        found_xyxy = min(same_class_dets,
                                         key=lambda b: _cdist(b, current_xyxy))

                # ── Disappearance / edge logic (only on YOLO frames) ───────
                if found_xyxy is not None:
                    miss_streak = 0
                    # Update velocity for extrapolation
                    vel_x = ((found_xyxy[0] + found_xyxy[2]) // 2 -
                             (prev_xyxy[0] + prev_xyxy[2]) // 2) // frame_skip
                    vel_y = ((found_xyxy[1] + found_xyxy[3]) // 2 -
                             (prev_xyxy[1] + prev_xyxy[3]) // 2) // frame_skip
                    prev_xyxy    = current_xyxy
                    current_xyxy = found_xyxy

                    bx1, by1, bx2, by2 = found_xyxy

                    if _at_edge(found_xyxy):
                        interior = [b for b in same_class_dets if not _at_edge(b)]
                        if interior:
                            found_xyxy = min(interior,
                                             key=lambda b: _cdist(b, current_xyxy))
                            current_xyxy = found_xyxy
                            bx1, by1, bx2, by2 = found_xyxy
                            edge_streak = 0
                        else:
                            edge_streak += 1
                            if edge_streak >= EDGE_PATIENCE:
                                disappeared             = True
                                disappearance_frame_idx = frame_idx
                                break
                    else:
                        edge_streak = 0

                else:
                    miss_streak += 1
                    edge_streak  = 0
                    # Keep extrapolating — don't move current_xyxy
                    if miss_streak >= MISS_PATIENCE:
                        disappeared             = True
                        disappearance_frame_idx = frame_idx
                        break

            else:
                # Skipped frame: only extrapolate if we currently have a good lock
                if miss_streak == 0:
                    w = current_xyxy[2] - current_xyxy[0]
                    h = current_xyxy[3] - current_xyxy[1]
                    cx = (current_xyxy[0] + current_xyxy[2]) // 2 + vel_x
                    cy = (current_xyxy[1] + current_xyxy[3]) // 2 + vel_y
                    cx = max(w // 2, min(proc_w - w // 2, cx))
                    cy = max(h // 2, min(proc_h - h // 2, cy))
                    current_xyxy = (cx - w // 2, cy - h // 2,
                                    cx + w // 2, cy + h // 2)

            # Live feed — skip drawing intermediate frames if frame_skip > 1 to save overhead
            if frame_callback and run_yolo:
                frame_rgb = cv2.cvtColor(current_bgr, cv2.COLOR_BGR2RGB)
                if miss_streak > 0:
                    # Object not found yet — show frame with no bbox
                    frame_callback(frame_rgb, None)
                else:
                    bx1, by1, bx2, by2 = current_xyxy
                    ox1, oy1 = int(bx1 / scale), int(by1 / scale)
                    ox2, oy2 = int(bx2 / scale), int(by2 / scale)
                    frame_callback(frame_rgb, (ox1, oy1, ox2, oy2))

            frame_idx += 1
            if progress_callback and (frame_idx % 10 == 0 or run_yolo):
                progress_callback(frame_idx, total_frames)

        # ── Build result dict ──────────────────────────────────────────────
        res_dict = {'disappeared': disappeared, 'last_frame_idx': frame_idx - 1}

        if disappeared:
            seconds               = int(disappearance_frame_idx / fps)
            res_dict['timestamp'] = str(timedelta(seconds=seconds))

            try:
                import easyocr, warnings
                warnings.filterwarnings("ignore", category=UserWarning,
                                        module="torch.utils.data.dataloader")
                h, w, _ = current_bgr.shape
                crop    = current_bgr[0:int(h * 0.2), int(w * 0.5):w]
                reader  = easyocr.Reader(['en'], gpu=False, verbose=False)
                ocr_txt = " ".join(reader.readtext(crop, detail=0)).strip()
                if ocr_txt:
                    res_dict['timestamp_ocr'] = ocr_txt
            except Exception as e:
                print(f"OCR skipped: {e}")

            res_dict['frame_before'] = first_frame_bgr
            res_dict['frame_after']  = current_bgr

        cap.release()
        return res_dict
