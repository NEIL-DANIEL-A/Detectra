import cv2
import numpy as np

def create_mock_video(output_path='mock.mp4', num_frames=100, width=640, height=480, fps=30):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Box properties
    box_size = 50
    x, y = 100, 100
    color = (0, 0, 255) # Red

    disappearance_frame = 70 # Disappears at frame 70

    for i in range(num_frames):
        # Create a light gray background
        frame = np.ones((height, width, 3), dtype=np.uint8) * 200

        # Move the box diagonally
        x += 2
        y += 1

        # Draw the box until it disappears
        if i < disappearance_frame:
            cv2.rectangle(frame, (x, y), (x + box_size, y + box_size), color, -1)
            
            # To make it realistic for YOLO, let's draw something that looks sort of like a person or a car, 
            # or just rely on YOLO picking up anything but usually YOLO won't detect a simple red square.
            # To be safe for testing, let's draw an image of a person or make it complex enough, 
            # Or we can just use the YOLO 'person' class if we overlay a face/body.
            # actually YOLOv8 can detect simple shapes sometimes if they look like a class, like a 'stop sign'.
            # Let's draw a more complex shape (a person-like figure).
            
            # Head
            cv2.circle(frame, (x + 25, y + 10), 10, (0,0,0), -1)
            # Body
            cv2.rectangle(frame, (x + 15, y + 20), (x + 35, y + box_size), (0,0,0), -1)

        # Draw some static background objects so YOLO has other things to detect (e.g., a "car" shape)
        cv2.rectangle(frame, (400, 300), (500, 350), (255, 0, 0), -1) # Blue box
        cv2.circle(frame, (420, 350), 15, (0, 0, 0), -1) # wheels
        cv2.circle(frame, (480, 350), 15, (0, 0, 0), -1)
        
        out.write(frame)

    out.release()
    print(f"Mock video saved to {output_path}")

if __name__ == "__main__":
    create_mock_video()
