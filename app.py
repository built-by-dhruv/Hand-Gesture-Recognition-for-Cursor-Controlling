import os
import warnings
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
logging.getLogger('mediapipe').setLevel(logging.ERROR)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

try:
    import absl.logging
    absl.logging.set_verbosity(absl.logging.ERROR)
except ImportError:
    pass

import cv2
import mediapipe as mp
import time
from controller import Controller, Config , initialize_controller

class HandTrackingApp:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        if not self.cap.isOpened():
            raise Exception("Error: Could not open camera")
        
        self.mpHands = mp.solutions.hands # type: ignore
        self.hands = self.mpHands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mpDraw = mp.solutions.drawing_utils # type: ignore
        
        self.prev_time = 0
        self.fps_counter = 0
        self.fps_display = 0
        
        self.hand_detected = False
        self.frames_without_hand = 0
        self.max_frames_without_hand = 1
        
        print("Hand Tracking Controller initialized successfully!")
        print("Controls:")
        print("- Move wrist to control cursor")
        print("- All fingers up + thumb down = freeze cursor")
        print("- ESC key to exit")
        print("Adjust parameters in Config class (controller.py) to test cursor behavior.")
        
    def calculate_fps(self):
        current_time = time.time()
        fps = 1 / (current_time - self.prev_time) if self.prev_time != 0 else 0
        self.prev_time = current_time
        
        self.fps_counter += 1
        if self.fps_counter >= 10:
            self.fps_display = int(fps)
            self.fps_counter = 0
            
        return self.fps_display
    
    def draw_info_overlay(self, img):
        height, width = img.shape[:2]
        
        fps = self.calculate_fps()
        cv2.putText(img, f'FPS: {fps}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        status_color = (0, 255, 0) if self.hand_detected else (0, 0, 255)
        status_text = "Hand: Detected" if self.hand_detected else "Hand: Not Detected"
        cv2.putText(img, status_text, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        if self.hand_detected and hasattr(Controller, 'all_fingers_up'):
            cursor_frozen = Controller.all_fingers_up and Controller.thumb_finger_down
            cursor_status = "Cursor: FROZEN" if cursor_frozen else "Cursor: ACTIVE"
            cursor_color = (0, 255, 255) if cursor_frozen else (255, 255, 0)
            cv2.putText(img, cursor_status, (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, cursor_color, 2)
        
        cv2.putText(img, "ESC: Exit", (width - 100, height - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def process_hand_landmarks(self, results, img):
        if results.multi_hand_landmarks:
            self.hand_detected = True
            self.frames_without_hand = 0
            
            Controller.hand_Landmarks = results.multi_hand_landmarks[0]
            
            self.mpDraw.draw_landmarks(
                img, 
                Controller.hand_Landmarks, 
                self.mpHands.HAND_CONNECTIONS,
                landmark_drawing_spec=self.mpDraw.DrawingSpec(
                    color=(0, 0, 255), thickness=2, circle_radius=2
                ),
                connection_drawing_spec=self.mpDraw.DrawingSpec(
                    color=(0, 255, 0), thickness=2
                )
            )
            
            try:
                Controller.update_fingers_status()
                Controller.cursor_moving()
                
                # if hasattr(Controller, 'detect_scrolling'):
                #     Controller.detect_scrolling()
                # if hasattr(Controller, 'detect_zoomming'):
                #     Controller.detect_zoomming()
                # if hasattr(Controller, 'detect_clicking'):
                #     Controller.detect_clicking()
                # if hasattr(Controller, 'detect_dragging'):
                #     Controller.detect_dragging()
                    
            except Exception as e:
                print(f"Error in controller methods: {e}")
        else:
            self.frames_without_hand += 1
            if self.frames_without_hand >= self.max_frames_without_hand:
                self.hand_detected = False
                Controller.hand_Landmarks = None
                if hasattr(Controller, 'reset_smoothing'):
                    Controller.reset_smoothing()
    
    def run(self):
        try:
            while True:
                success, img = self.cap.read()
                
                if not success:
                    print("Error: Failed to read from camera")
                    break
                
                img = cv2.flip(img, 1)
                
                imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                results = self.hands.process(imgRGB)
                
                self.process_hand_landmarks(results, img)
                
                self.draw_info_overlay(img)
                
                cv2.imshow('Hand Gesture Controller', img)
                
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC key
                    print("Exiting application...")
                    break
                elif key == ord('r'):
                    print("Resetting controller...")
                    if hasattr(Controller, 'reset_smoothing'):
                        Controller.reset_smoothing()
                        
        except KeyboardInterrupt:
            print("\nApplication interrupted by user")
        except Exception as e:
            print(f"Unexpected error: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        print("Cleaning up...")
        self.cap.release()
        cv2.destroyAllWindows()
        print("Application closed successfully")

def main():
    """Main function"""
    Controller.set_sensitivity(base_sensitivity=10, max_multiplier=5.0)  # Optimized for wrist tracking
    initialize_controller()

    try:
        app = HandTrackingApp()
        app.run()
    except Exception as e:
        print(f"Failed to initialize application: {e}")
        print("Please check your camera connection and try again")

if __name__ == "__main__":
    main()