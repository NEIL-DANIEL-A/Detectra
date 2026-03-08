import cv2
import os
import shutil
from datetime import timedelta
from ultralytics import YOLO

class Tracker:
    def __init__(self, model_path='yolov8n.pt'):
        # Initialize YOLO model
        # Using yolov8n.pt (nano) for speed. 
        self.model = YOLO(model_path)
    
    def extract_first_frame(self, video_path):
        """
        Reads the video file and returns the first valid frame.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None, "Error: Could not open video file."
        
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return None, "Error: Could not read first frame."
        
        # Convert BGR (OpenCV) to RGB for UI
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame_rgb, None

    def detect_objects(self, frame_rgb):
        """
        Runs YOLO detection on a single frame. 
        Returns a list of dicts: {'bbox': (x1,y1,x2,y2), 'confidence': conf, 'class': cls_id}
        """
        # YOLO accepts BGR or RGB, but we must be consistent. Since we already loaded RGB:
        results = self.model(frame_rgb, verbose=False)
        detections = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # bounding box
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf[0].item()
                cls_id = int(box.cls[0].item())
                
                detections.append({
                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                    'confidence': conf,
                    'class': cls_id,
                    'class_name': result.names[cls_id]
                })
        return detections

    def calculate_iou(self, boxA, boxB):
        # Determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # Compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        # Compute the area of both the prediction and ground-truth rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

        # Compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the intersection area
        iou = interArea / float(boxAArea + boxBArea - interArea)

        return iou

    def process_video(self, video_path, target_bbox, progress_callback=None, frame_callback=None):
        """
        Processes the video frame by frame. Tracks the target_bbox using OpenCV KCF tracker (fastest).
        Flags disappearance if the object is missing (tracker failure) or leaves the frame boundaries.
        
        target_bbox is (x1, y1, x2, y2)
        progress_callback is a function(current_frame, total_frames) to update the GUI
        frame_callback is a function(frame_rgb, bbox) to update the GUI canvas with tracking view
        
        Returns:
            dict with:
                'disappeared': bool
                'timestamp': str (HH:MM:SS)
                'frame_before_path': str
                'frame_after_path': str
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"error": "Could not open video."}

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Determine scale down factor for speed. Max dimension 640.
        max_dim = 640
        scale = 1.0
        if orig_width > max_dim or orig_height > max_dim:
            scale = min(max_dim / float(orig_width), max_dim / float(orig_height))
            
        process_width = int(orig_width * scale)
        process_height = int(orig_height * scale)
        
        # Prepare output directory
        results_dir = 'results'
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        # Initialize OpenCV Tracker (using MIL as it is built-in and stable across versions)
        try:
            tracker = cv2.TrackerMIL_create()
        except AttributeError:
            # Fallback
            tracker = cv2.legacy.TrackerMIL_create()
            
        # Convert (x1, y1, x2, y2) to scaled (x, y, w, h)
        x1, y1, x2, y2 = target_bbox
        init_bbox = (
            int(x1 * scale), 
            int(y1 * scale), 
            int((x2 - x1) * scale), 
            int((y2 - y1) * scale)
        )
        
        ret, frame = cap.read()
        if not ret:
            return {"error": "Could not read video."}
            
        # Resize frame for tracker
        process_frame = cv2.resize(frame, (process_width, process_height))
        
        # --- Dynamic YOLO Initialization ---
        target_class_id = None
        current_bbox = init_bbox
        
        # Run YOLO on the first scaled frame
        detections = self.detect_objects(process_frame)
        box_A = (init_bbox[0], init_bbox[1], init_bbox[0]+init_bbox[2], init_bbox[1]+init_bbox[3])
        
        best_iou = 0.0
        best_det = None
        for det in detections:
            iou = self.calculate_iou(box_A, det['bbox'])
            if iou > best_iou:
                best_iou = iou
                best_det = det
                
        # Snap to YOLO object if there's a good overlap
        if best_iou > 0.3 and best_det is not None:
            target_class_id = best_det['class']
            bx1, by1, bx2, by2 = best_det['bbox']
            current_bbox = (bx1, by1, bx2 - bx1, by2 - by1)
            print(f"Target locked to YOLO class {target_class_id} ({best_det['class_name']}) with IoU {best_iou:.2f}")

        # Initialize OpenCV Tracker (MIL)
        try:
            tracker = cv2.TrackerMIL_create()
        except AttributeError:
            tracker = cv2.legacy.TrackerMIL_create()
            
        tracker.init(process_frame, current_bbox)

        first_frame_bgr = frame.copy()
        frame_idx = 1
        
        disappeared = False
        disappearance_frame_idx = 0
        
        # Edge margin conceptually scaled
        edge_margin = max(5, int(min(process_width, process_height) * 0.01)) 

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            current_bgr_frame = frame.copy()
            process_frame = cv2.resize(frame, (process_width, process_height))
            
            # 1. Update MIL tracker first as a baseline
            kcf_success, kcf_bbox = tracker.update(process_frame)
            
            # 2. Try to find the object with YOLO if we locked onto a class
            yolo_match_found = False
            if target_class_id is not None:
                detections = self.detect_objects(process_frame)
                
                # Compare against expected position
                pred_bbox = kcf_bbox if kcf_success else current_bbox
                box_pred = (pred_bbox[0], pred_bbox[1], pred_bbox[0]+pred_bbox[2], pred_bbox[1]+pred_bbox[3])
                
                # Prevent YOLO from finding a new object if the current object is already touching or very close to the edge.
                pred_tx, pred_ty, pred_tw, pred_th = [int(v) for v in pred_bbox]
                is_near_edge = (
                    pred_tx <= edge_margin * 3 or 
                    pred_ty <= edge_margin * 3 or 
                    (pred_tx + pred_tw) >= (process_width - edge_margin * 3) or 
                    (pred_ty + pred_th) >= (process_height - edge_margin * 3)
                )

                best_iou = 0.0
                best_det = None
                
                if not is_near_edge:
                    for det in detections:
                        if det['class'] == target_class_id:
                            iou = self.calculate_iou(box_pred, det['bbox'])
                            if iou > best_iou:
                                best_iou = iou
                                best_det = det
                                
                # If YOLO found a good match and we are not near the edge, snap to it and re-init MIL
                if best_iou > 0.4 and best_det is not None:
                    bx1, by1, bx2, by2 = best_det['bbox']
                    current_bbox = (bx1, by1, bx2 - bx1, by2 - by1)
                    yolo_match_found = True
                    
                    try:
                        tracker = cv2.TrackerMIL_create()
                    except AttributeError:
                        tracker = cv2.legacy.TrackerMIL_create()
                    tracker.init(process_frame, current_bbox)
            
            # 3. Resolve current bbox state
            if yolo_match_found:
                success = True
                bbox = current_bbox
            else:
                success = kcf_success
                bbox = kcf_bbox
                if success:
                    current_bbox = kcf_bbox
            
            if success:
                # Bbox is (x, y, w, h) on scaled frame
                tx, ty, tw, th = [int(v) for v in bbox]
                
                # Check if it touches the borders
                if (tx <= edge_margin or 
                    ty <= edge_margin or 
                    (tx + tw) >= (process_width - edge_margin) or 
                    (ty + th) >= (process_height - edge_margin)):
                    
                    disappeared = True
                    disappearance_frame_idx = frame_idx
                    break
                    
                # Callback for live tracking
                if frame_callback:
                    ux, uy, uw, uh = tx/scale, ty/scale, tw/scale, th/scale
                    unscaled_bbox = (int(ux), int(uy), int(ux+uw), int(uy+uh))
                    
                    frame_rgb = cv2.cvtColor(current_bgr_frame, cv2.COLOR_BGR2RGB)
                    frame_callback(frame_rgb, unscaled_bbox)
            else:
                # Tracker lost the object
                disappeared = True
                disappearance_frame_idx = frame_idx
                break
                
            frame_idx += 1
            if progress_callback and frame_idx % 5 == 0:
                progress_callback(frame_idx, total_frames)

        # Build results
        res = {'disappeared': disappeared}
        if disappeared:
            seconds = int(disappearance_frame_idx / fps)
            res['timestamp'] = str(timedelta(seconds=seconds))
            
            # Extract OCR Timestamp from the target frame
            try:
                import easyocr
                h, w, _ = current_bgr_frame.shape
                # Crop top right region (top 20%, right 50%)
                crop = current_bgr_frame[0:int(h*0.2), int(w*0.5):w]
                reader = easyocr.Reader(['en'], gpu=False)
                ocr_result = reader.readtext(crop, detail=0)
                timestamp_ocr = " ".join(ocr_result)
                if timestamp_ocr.strip():
                    res['timestamp_ocr'] = timestamp_ocr.strip()
            except Exception as e:
                print(f"OCR failed or EasyOCR not installed: {e}")
            
            before_path = os.path.join(results_dir, "before_disappearance.jpg")
            after_path = os.path.join(results_dir, "after_disappearance.jpg")
            
            cv2.imwrite(before_path, first_frame_bgr)
            cv2.imwrite(after_path, current_bgr_frame)
            
            res['frame_before_path'] = before_path
            res['frame_after_path'] = after_path
        
        cap.release()
        return res

