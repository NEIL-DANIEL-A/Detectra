import cv2
import os
import sys
from pathlib import Path
from datetime import datetime
from datetime import timedelta
from ultralytics import YOLO

# ─────────────────────────────────────────
#  MODEL PATHS (loaded from AppData)
# ─────────────────────────────────────────
APP_DATA_DIR = Path(os.getenv("LOCALAPPDATA")) / "Detectra"
MODELS_DIR   = APP_DATA_DIR / "models"
EASYOCR_DIR  = APP_DATA_DIR / "easyocr"


class Tracker:
    def __init__(self):
        model_path = str(MODELS_DIR / "yolov8n.pt")
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
                    'bbox'      : (int(x1), int(y1), int(x2), int(y2)),
                    'confidence': box.conf[0].item(),
                    'class'     : cls_id,
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

    def _init_csrt(self, frame, bbox):
        """
        Initialize CSRT tracker on given frame and bbox.
        bbox: (x1, y1, x2, y2) → converted to (x, y, w, h) for OpenCV.
        """
        tracker = cv2.TrackerCSRT_create()
        x1, y1, x2, y2 = bbox
        tracker.init(frame, (x1, y1, x2 - x1, y2 - y1))
        return tracker

    def _csrt_update(self, csrt, frame):
        """
        Update CSRT tracker.
        Returns (x1, y1, x2, y2) if tracking, or None if lost.
        """
        ok, box = csrt.update(frame)
        if not ok:
            return None
        x, y, w, h = [int(v) for v in box]
        return (x, y, x + w, y + h)

    def _csrt_still_valid(self, csrt_bbox, last_yolo_bbox,
                           frames_since_yolo, max_allowed_frames=10):
        """
        Validate whether CSRT result should be trusted.

        CSRT is trusted ONLY when:
        1. It hasn't been too long since YOLO last confirmed the object
        2. CSRT bbox hasn't moved too far from last known YOLO position

        This prevents CSRT from falsely tracking background after
        the object has disappeared.

        Returns True if CSRT is trustworthy, False if it should be ignored.
        """
        # ── Rule 1: Don't trust CSRT if YOLO hasn't confirmed
        #            the object for too many frames ──────────────────────
        if frames_since_yolo > max_allowed_frames:
            return False

        # ── Rule 2: Don't trust CSRT if it has drifted too far
        #            from last known YOLO position ─────────────────────
        if last_yolo_bbox is not None:
            cx_csrt  = (csrt_bbox[0] + csrt_bbox[2]) // 2
            cy_csrt  = (csrt_bbox[1] + csrt_bbox[3]) // 2
            cx_yolo  = (last_yolo_bbox[0] + last_yolo_bbox[2]) // 2
            cy_yolo  = (last_yolo_bbox[1] + last_yolo_bbox[3]) // 2

            # Max allowed drift = 1.5x the object's width
            obj_w    = last_yolo_bbox[2] - last_yolo_bbox[0]
            obj_h    = last_yolo_bbox[3] - last_yolo_bbox[1]
            max_drift = max(obj_w, obj_h) * 1.5

            drift = ((cx_csrt - cx_yolo) ** 2 + (cy_csrt - cy_yolo) ** 2) ** 0.5
            if drift > max_drift:
                return False

        return True

    # ── Main tracking ─────────────────────────────────────────────────────

    def process_video(self, video_path, target_bbox,
                      progress_callback=None, frame_callback=None,
                      stop_event=None, frame_skip=3, start_frame=0,
                      disappearance_callback=None):
        """
        YOLO + CSRT Hybrid Tracker (Validated):

        - CSRT runs every frame for smooth tracking
        - YOLO runs every N frames as ground truth verifier
        - CSRT is only trusted when:
            a) YOLO confirmed object recently (within frame_skip frames)
            b) CSRT hasn't drifted far from last YOLO position
        - If CSRT fails validation → treated same as CSRT lost
        - Both YOLO and validated CSRT lost → miss streak increments
        - miss streak >= MISS_PATIENCE → disappearance declared

        MISS_PATIENCE   : YOLO frames with no detection before disappearance
        EDGE_PATIENCE   : Frames at edge before declaring exit
        CSRT_REINIT_IOU : Min IoU between CSRT and YOLO before reinit
        CSRT_MAX_SOLO   : Max frames CSRT can track alone without YOLO confirmation
        """

        # MISS_PATIENCE is in sampled-frame units (one per seek step).
        # We want roughly the same real-frame tolerance regardless of speed,
        # so divide the raw-frame budget (30) by frame_skip.
        MISS_PATIENCE   = max(5, 30 // frame_skip)
        EDGE_PATIENCE   = 5
        CSRT_REINIT_IOU = 0.3
        CSRT_MAX_SOLO   = 2   # sampled frames: CSRT trusted for 2 steps after YOLO

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"error": "Could not open video."}

        fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
        self._last_fps = fps  # exposed so disappearance_callback can use it
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        orig_w       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        max_dim = 640
        scale   = (min(max_dim / orig_w, max_dim / orig_h)
                   if (orig_w > max_dim or orig_h > max_dim) else 1.0)
        proc_w  = int(orig_w * scale)
        proc_h  = int(orig_h * scale)

        proc_h  = int(orig_h * scale)

        ux1, uy1, ux2, uy2 = target_bbox
        sx1, sy1 = int(ux1 * scale), int(uy1 * scale)
        sx2, sy2 = int(ux2 * scale), int(uy2 * scale)

        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # ── Frame 0: Lock onto YOLO track ID + init CSRT ─────────────────
        ret, frame = cap.read()
        if not ret:
            cap.release()
            return {"error": "Could not read video."}

        first_frame_bgr = frame.copy()
        pf0             = cv2.resize(frame, (proc_w, proc_h))
        res0            = self.model.track(pf0, persist=True, verbose=False)

        target_track_id = None
        target_class_id = None
        current_xyxy    = (sx1, sy1, sx2, sy2)

        if res0 and res0[0].boxes is not None and len(res0[0].boxes):
            user_cx    = (sx1 + sx2) // 2
            user_cy    = (sy1 + sy2) // 2
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
                print(f"[Frame 0] YOLO locked → track_id={target_track_id}, "
                      f"class={res0[0].names[target_class_id]} (IoU={best_iou:.2f})")

        # ── Init CSRT on frame 0 ──────────────────────────────────────────
        csrt              = self._init_csrt(pf0, current_xyxy)
        csrt_bbox         = current_xyxy
        last_yolo_bbox    = current_xyxy   # last position YOLO confirmed
        frames_since_yolo = 0              # frames since YOLO last confirmed

        print(f"[Frame 0] CSRT initialized on bbox={current_xyxy}")

        edge_margin = max(5, int(min(proc_w, proc_h) * 0.015))

        def _at_edge(b):
            return (b[0] <= edge_margin or b[1] <= edge_margin or
                    b[2] >= proc_w - edge_margin or
                    b[3] >= proc_h - edge_margin)

        def _cdist(b1, b2):
            return (abs((b1[0] + b1[2]) // 2 - (b2[0] + b2[2]) // 2) +
                    abs((b1[1] + b1[3]) // 2 - (b2[1] + b2[3]) // 2))

        prev_xyxy = current_xyxy
        vel_x     = 0
        vel_y     = 0

        frame_idx               = start_frame + 1
        disappeared             = False
        disappearance_frame_idx = 0
        miss_streak             = 0
        edge_streak             = 0

        # ── Per-frame loop ────────────────────────────────────────────────
        # Speed is implemented by physically seeking ahead frame_skip frames
        # each iteration so the decoder never touches skipped frames.
        # CSRT also runs only on the sampled frame — since YOLO reinits CSRT
        # whenever it drifts, single-frame CSRT updates are still valid.
        while cap.isOpened():
            if stop_event is not None and stop_event.is_set():
                cap.release()
                return {'stopped': True, 'last_frame_idx': frame_idx - 1}

            ret, frame = cap.read()
            if not ret:
                break

            current_bgr = frame.copy()
            pf          = cv2.resize(frame, (proc_w, proc_h))
            run_yolo    = True   # every sampled frame runs YOLO

            # ── CSRT: update on this sampled frame ───────────────────────
            csrt_result = self._csrt_update(csrt, pf)

            if csrt_result is not None:
                csrt_bbox = csrt_result

            # ── Validate CSRT before trusting it ─────────────────────────
            frames_since_yolo += 1
            csrt_trusted = (
                csrt_result is not None and
                self._csrt_still_valid(
                    csrt_bbox, last_yolo_bbox,
                    frames_since_yolo, CSRT_MAX_SOLO
                )
            )

            # ── YOLO: runs every N frames ─────────────────────────────────
            if run_yolo:
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

                # ── YOLO found object ─────────────────────────────────────
                if found_xyxy is not None:
                    miss_streak       = 0
                    last_yolo_bbox    = found_xyxy  # update last confirmed YOLO pos
                    frames_since_yolo = 0           # reset counter

                    # Check if CSRT drifted from YOLO
                    iou_check = self.calculate_iou(csrt_bbox, found_xyxy)
                    if iou_check < CSRT_REINIT_IOU:
                        print(f"[Frame {frame_idx}] CSRT drifted "
                              f"(IoU={iou_check:.2f}), reinit on YOLO bbox")
                        csrt      = self._init_csrt(pf, found_xyxy)
                        csrt_bbox = found_xyxy

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
                            found_xyxy   = min(interior,
                                               key=lambda b: _cdist(b, current_xyxy))
                            current_xyxy = found_xyxy
                            bx1, by1, bx2, by2 = found_xyxy
                            csrt      = self._init_csrt(pf, found_xyxy)
                            csrt_bbox = found_xyxy
                            edge_streak = 0
                        else:
                            edge_streak += 1
                            if edge_streak >= EDGE_PATIENCE:
                                disappeared             = True
                                disappearance_frame_idx = frame_idx
                                if disappearance_callback:
                                    disappearance_callback(first_frame_bgr, current_bgr, frame_idx)
                                break
                    else:
                        edge_streak = 0

                # ── YOLO lost object ──────────────────────────────────────
                else:
                    if csrt_trusted:
                        # CSRT validated — trust it, don't count as miss
                        print(f"[Frame {frame_idx}] YOLO lost, "
                              f"CSRT validated at {csrt_bbox}")
                        current_xyxy = csrt_bbox
                    else:
                        # YOLO lost + CSRT not trusted = real miss
                        miss_streak += 1
                        edge_streak  = 0
                        if miss_streak >= MISS_PATIENCE:
                            disappeared             = True
                            disappearance_frame_idx = frame_idx
                            if disappearance_callback:
                                disappearance_callback(first_frame_bgr, current_bgr, frame_idx)
                            break

            # (Non-YOLO frame branch removed — every sampled frame now runs
            # YOLO. Speed is controlled by seeking, not by skipping YOLO.)

            # ── Live feed callback ────────────────────────────────────────
            if frame_callback:
                frame_rgb = cv2.cvtColor(current_bgr, cv2.COLOR_BGR2RGB)
                if miss_streak > 0 and not csrt_trusted:
                    frame_callback(frame_rgb, None)
                else:
                    bx1, by1, bx2, by2 = current_xyxy
                    ox1, oy1 = int(bx1 / scale), int(by1 / scale)
                    ox2, oy2 = int(bx2 / scale), int(by2 / scale)
                    frame_callback(frame_rgb, (ox1, oy1, ox2, oy2))

            frame_idx += frame_skip   # advance logical frame counter by skip amount

            # ── Progress callback: fire at most every 1% of total frames ──
            # Using a fixed modulo on frame_idx caused a backlog in root.after()
            # at high speeds (frame_skip % frame_skip == 0 is always true).
            # Instead we compute a step size = 1% of total and only fire when
            # we cross a new 1% boundary.
            if progress_callback:
                _step = max(frame_skip, int(total_frames * 0.01))
                if frame_idx % _step < frame_skip:
                    progress_callback(frame_idx, total_frames)

            # ── Physical seek: jump ahead frame_skip frames in the video ─
            # This is what actually makes higher speeds faster — the decoder
            # skips frame_skip-1 frames entirely instead of decoding them.
            if frame_skip > 1:
                next_pos = frame_idx
                if next_pos < total_frames:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, next_pos)

        # ── Handle disappearance at video end ─────────────────────────────
        MIN_END_MISS = 5
        if not disappeared and miss_streak >= MIN_END_MISS:
            disappeared             = True
            disappearance_frame_idx = frame_idx - (miss_streak * frame_skip)

        # ── Build result dict ─────────────────────────────────────────────
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
                reader  = easyocr.Reader(
                    ['en'],
                    gpu=False,
                    verbose=False,
                    model_storage_directory=str(EASYOCR_DIR)
                )
                ocr_txt = " ".join(reader.readtext(crop, detail=0)).strip()
                if ocr_txt:
                    res_dict['timestamp_ocr'] = ocr_txt
            except Exception as e:
                print(f"OCR skipped: {e}")

            res_dict['frame_before'] = first_frame_bgr
            res_dict['frame_after']  = current_bgr

        cap.release()
        return res_dict