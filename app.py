# Suppress warnings before importing ML libraries
import os
import warnings
import logging

# Suppress TensorFlow and MediaPipe warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
logging.getLogger('mediapipe').setLevel(logging.ERROR)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Suppress ABSL logging
try:
    import absl.logging
    absl.logging.set_verbosity(absl.logging.ERROR)
except ImportError:
    pass

# Main imports
import cv2
import mediapipe as mp
import time
from controller import Controller

class HandTrackingApp:
    def __init__(self):
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Check if camera opened successfully
        if not self.cap.isOpened():
            raise Exception("Error: Could not open camera")
        
        # Initialize MediaPipe
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=False,
            max_num_hands=1,  # Only track one hand for better performance
            min_detection_confidence=0.7,  # Increased for better detection
            min_tracking_confidence=0.5
        )
        self.mpDraw = mp.solutions.drawing_utils
        
        # Performance tracking
        self.prev_time = 0
        self.fps_counter = 0
        self.fps_display = 0
        
        # Hand tracking state
        self.hand_detected = False
        self.frames_without_hand = 0
        self.max_frames_without_hand = 10  # Reset after 10 frames without detection
        
        print("Hand Tracking Controller initialized successfully!")
        print("Controls:")
        print("- Move index finger to control cursor")
        print("- All fingers up + thumb down = freeze cursor")
        print("- ESC key to exit")
        
    def calculate_fps(self):
        """Calculate and display FPS"""
        current_time = time.time()
        fps = 1 / (current_time - self.prev_time) if self.prev_time != 0 else 0
        self.prev_time = current_time
        
        # Update FPS display every 10 frames for smoother display
        self.fps_counter += 1
        if self.fps_counter >= 10:
            self.fps_display = int(fps)
            self.fps_counter = 0
            
        return self.fps_display
    
    def draw_info_overlay(self, img):
        """Draw information overlay on the image"""
        height, width = img.shape[:2]
        
        # Draw FPS
        fps = self.calculate_fps()
        cv2.putText(img, f'FPS: {fps}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw hand detection status
        status_color = (0, 255, 0) if self.hand_detected else (0, 0, 255)
        status_text = "Hand: Detected" if self.hand_detected else "Hand: Not Detected"
        cv2.putText(img, status_text, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        # Draw cursor status if hand is detected
        if self.hand_detected and hasattr(Controller, 'all_fingers_up'):
            cursor_frozen = Controller.all_fingers_up and Controller.thumb_finger_down
            cursor_status = "Cursor: FROZEN" if cursor_frozen else "Cursor: ACTIVE"
            cursor_color = (0, 255, 255) if cursor_frozen else (255, 255, 0)
            cv2.putText(img, cursor_status, (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, cursor_color, 2)
        
        # Draw instructions
        cv2.putText(img, "ESC: Exit", (width - 100, height - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def process_hand_landmarks(self, results, img):
        """Process hand landmarks and update controller"""
        if results.multi_hand_landmarks:
            # Hand detected
            self.hand_detected = True
            self.frames_without_hand = 0
            
            # Use the first (and only) hand
            Controller.hand_Landmarks = results.multi_hand_landmarks[0]
            
            # Draw hand landmarks
            self.mpDraw.draw_landmarks(
                img, 
                Controller.hand_Landmarks, 
                self.mpHands.HAND_CONNECTIONS,
                landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                    color=(0, 0, 255), thickness=2, circle_radius=2
                ),
                connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                    color=(0, 255, 0), thickness=2
                )
            )
            
            # Update controller
            try:
                Controller.update_fingers_status()
                Controller.cursor_moving()
                
                # Call other detection methods if they exist
                if hasattr(Controller, 'detect_scrolling'):
                    Controller.detect_scrolling()
                if hasattr(Controller, 'detect_zoomming'):
                    Controller.detect_zoomming()
                if hasattr(Controller, 'detect_clicking'):
                    Controller.detect_clicking()
                if hasattr(Controller, 'detect_dragging'):
                    Controller.detect_dragging()
                    
            except Exception as e:
                print(f"Error in controller methods: {e}")
        else:
            # No hand detected
            self.frames_without_hand += 1
            if self.frames_without_hand >= self.max_frames_without_hand:
                self.hand_detected = False
                Controller.hand_Landmarks = None
                # Reset smoothing when hand is lost
                if hasattr(Controller, 'reset_smoothing'):
                    Controller.reset_smoothing()
    
    def run(self):
        """Main application loop"""
        try:
            while True:
                success, img = self.cap.read()
                
                if not success:
                    print("Error: Failed to read from camera")
                    break
                
                # Flip image horizontally for mirror effect
                img = cv2.flip(img, 1)
                
                # Convert BGR to RGB for MediaPipe
                imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Process hand detection
                results = self.hands.process(imgRGB)
                
                # Process hand landmarks and update controller
                self.process_hand_landmarks(results, img)
                
                # Draw information overlay
                self.draw_info_overlay(img)
                
                # Display the image
                cv2.imshow('Hand Gesture Controller', img)
                
                # Check for exit key (ESC)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC key
                    print("Exiting application...")
                    break
                elif key == ord('r'):  # 'R' key to reset
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
        """Clean up resources"""
        print("Cleaning up...")
        self.cap.release()
        cv2.destroyAllWindows()
        print("Application closed successfully")

def main():
   """Main function"""
   # For even more aggressive movement:
   Controller.set_sensitivity(base_sensitivity=1, max_multiplier=1.0)

   try:
      app = HandTrackingApp()
      app.run()
   except Exception as e:
      print(f"Failed to initialize application: {e}")
      print("Please check your camera connection and try again")

if __name__ == "__main__":
   main()