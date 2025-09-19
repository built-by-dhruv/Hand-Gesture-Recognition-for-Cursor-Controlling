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
from utils.fps_meter import FPSMeter
from video.capture_manager import CaptureManager

def reload_config():
    """
    Reload config values from config.json and update the Config class and controller runtime.
    """
    import importlib
    import config as config_module
    importlib.reload(config_module)
    global Config
    Config = config_module.Config
    initialize_controller()

class HandTrackingApp:
    def __init__(self):
        """
        Initialize the Hand Tracking Application, setting up camera, Mediapipe hands module,
        and other necessary parameters. Print control instructions for the user.
        """
        self.capture = CaptureManager(device_index=0, width=640, height=480, target_fps=Config.TARGET_FPS)
        
        self.mpHands = mp.solutions.hands # type: ignore
        self.hands = self.mpHands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mpDraw = mp.solutions.drawing_utils # type: ignore
        
        self.fps_meter = FPSMeter(window=30, ema_alpha=0.9)
        self.fps_display = 0
        
        self.hand_detected = False
        self.frames_without_hand = 0
        self.max_frames_without_hand = 1
        
    print("Hand Tracking Controller initialized successfully!")
    print("Controls:")
    print("- RIGHT hand: pinch (thumb + index) to ACTIVATE movement")
    print("- Move ring/little finger midpoint to move cursor")
    print("- LEFT hand: index+thumb = hold LEFT click; middle+thumb = RIGHT click")
    print("- All fingers up + thumb down = FREEZE cursor")
    print("- ESC key to exit")
    print("Adjust parameters in the Gradio UI for real-time tuning.")
        
    def calculate_fps(self):
        """Update and return smoothed FPS using FPSMeter."""
        self.fps_display = self.fps_meter.get_int()
        return self.fps_display
    
    def draw_info_overlay(self, img):
        """
        Draw overlay information on the image, including FPS, hand detection status,
        and cursor status (active or frozen). This information helps in understanding
        the current state of the application and the detected hand.
        """
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
        """
        Process the hand landmarks detected in the current frame, updating the controller
        state and drawing the landmarks and connections on the image. This method also
        handles the detection of various hand gestures and updates the cursor movement
        accordingly.
        """
        if results.multi_hand_landmarks:
            self.hand_detected = True
            self.frames_without_hand = 0

            hands = results.multi_hand_landmarks
            handedness_list = getattr(results, 'multi_handedness', None)
            left_hand = None
            right_hand = None
            if handedness_list:
                for i, lm in enumerate(hands):
                    try:
                        label = handedness_list[i].classification[0].label  # 'Left' or 'Right'
                        # We flipped the image horizontally, invert labels to match user's perspective
                        if label == 'Left':
                            label = 'Right'
                        elif label == 'Right':
                            label = 'Left'
                    except Exception:
                        label = None
                    if label == 'Right' and right_hand is None:
                        right_hand = lm
                    elif label == 'Left' and left_hand is None:
                        left_hand = lm
            else:
                # Fallback heuristic based on x position after flip (rightmost = right hand)
                if len(hands) == 1:
                    right_hand = hands[0]
                elif len(hands) >= 2:
                    x0 = hands[0].landmark[0].x
                    x1 = hands[1].landmark[0].x
                    if x0 >= x1:
                        right_hand, left_hand = hands[0], hands[1]
                    else:
                        right_hand, left_hand = hands[1], hands[0]

            # Optionally invert roles based on config
            if getattr(Config, 'INVERT_HANDS', False):
                right_hand, left_hand = left_hand, right_hand

            # Draw landmarks for all hands
            for lm in hands:
                self.mpDraw.draw_landmarks(
                    img,
                    lm,
                    self.mpHands.HAND_CONNECTIONS,
                    landmark_drawing_spec=self.mpDraw.DrawingSpec(
                        color=(0, 0, 255), thickness=2, circle_radius=2
                    ),
                    connection_drawing_spec=self.mpDraw.DrawingSpec(
                        color=(0, 255, 0), thickness=2
                    )
                )
            
            try:
                # Movement with right hand
                if right_hand is not None:
                    Controller.hand_Landmarks = right_hand
                    Controller.update_fingers_status()
                    Controller.cursor_moving()
                # Clicks with left hand
                if left_hand is not None:
                    Controller.hand_Landmarks = left_hand
                    Controller.update_fingers_status()
                    # Left click hold: index+thumb
                    Controller.handle_left_click_hold(bool(Controller.index_finger_within_thumb_finger))
                    # Right click single: middle+thumb
                    Controller.handle_right_click(bool(Controller.middle_finger_within_thumb_finger))
                else:
                    # No left hand -> ensure click states are reset
                    if hasattr(Controller, 'release_left_hold'):
                        Controller.release_left_hold()
                # Restore right hand as default context
                if right_hand is not None:
                    Controller.hand_Landmarks = right_hand
                
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
                if hasattr(Controller, 'release_left_hold'):
                    Controller.release_left_hold()
                # No right-hold; right click is stateless single click
    
    def run(self):
        """
        Main application loop. Processes webcam frames, updates controller, and checks running flag.
        """
        try:
            while Config.running:
                # keep capture target fps in sync with config
                self.capture.set_target_fps(Config.TARGET_FPS)
                success, img = self.capture.read()
                
                if not success:
                    print("Error: Failed to read from camera")
                    break
                
                img = cv2.flip(img, 1)
                
                imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                results = self.hands.process(imgRGB)
                
                self.process_hand_landmarks(results, img)
                # tick FPS after processing a frame
                self.fps_meter.tick()
                self.draw_info_overlay(img)
                
                cv2.imshow('Hand Gesture Controller', img)
                
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC key
                    print("Exiting application...")
                    break
        except KeyboardInterrupt:
            print("\nApplication interrupted by user")
        except Exception as e:
            print(f"Unexpected error: {e}")
        finally:
            self.cleanup()

    def cleanup(self):
        """
        Release resources and close all windows on exit.
        """
        print("Cleaning up...")
        self.capture.release()
        cv2.destroyAllWindows()
        print("Application closed successfully")

def main():
    """
    Main function to run the hand tracking app.
    """
    try:
        app = HandTrackingApp()
        app.run()
    except Exception as e:
        print(f"Failed to initialize application: {e}")
        print("Please check your camera connection and try again")

if __name__ == "__main__":
    main()