import pyautogui
import math
import time

class Controller:
    # Existing state variables
    prev_hand = None
    right_clicked = False
    left_clicked = False
    double_clicked = False
    dragging = False
    hand_Landmarks = None
    
    # Fixed naming and added missing variables
    little_finger_down = None
    little_finger_up = None
    index_finger_down = None
    index_finger_up = None
    middle_finger_down = None
    middle_finger_up = None
    ring_finger_down = None
    ring_finger_up = None
    thumb_finger_down = None  # Fixed typo: Thump -> thumb
    thumb_finger_up = None    # Fixed typo: Thump -> thumb
    all_fingers_down = None
    all_fingers_up = None
    
    # Gesture detection variables
    index_finger_within_thumb_finger = None
    middle_finger_within_thumb_finger = None
    little_finger_within_thumb_finger = None
    ring_finger_within_thumb_finger = None
    
    # Screen dimensions
    screen_width, screen_height = pyautogui.size()
    
    # Movement control variables
    _smoothing_factor = 0.5  # Reduced for more responsive movement
    # _smoothing_factor = 0.15  # Reduced for more responsive movement
    _min_movement_threshold = 10  # Reduced threshold
    _prev_smooth_x = None
    _prev_smooth_y = None
    _prev_time = None  # For velocity calculation
    _movement_mode = "dynamic"  # "dynamic" or "linear"

    @staticmethod
    def update_fingers_status():
        """
        Updates the status of all fingers based on hand landmarks
        Improved with better error handling and clearer logic
        """
        if not Controller.hand_Landmarks or not Controller.hand_Landmarks.landmark:
            return
            
        try:
            landmarks = Controller.hand_Landmarks.landmark
            
            # Finger tip and base landmark indices
            # Using more accurate finger detection based on MediaPipe hand landmarks
            Controller.little_finger_down = landmarks[20].y > landmarks[17].y
            Controller.little_finger_up = landmarks[20].y < landmarks[17].y
            
            Controller.index_finger_down = landmarks[8].y > landmarks[5].y
            Controller.index_finger_up = landmarks[8].y < landmarks[5].y
            
            Controller.middle_finger_down = landmarks[12].y > landmarks[9].y
            Controller.middle_finger_up = landmarks[12].y < landmarks[9].y
            
            Controller.ring_finger_down = landmarks[16].y > landmarks[13].y
            Controller.ring_finger_up = landmarks[16].y < landmarks[13].y
            
            # Improved thumb detection using wrist as reference point
            Controller.thumb_finger_down = landmarks[4].y > landmarks[3].y
            Controller.thumb_finger_up = landmarks[4].y < landmarks[3].y
            
            # All fingers status
            Controller.all_fingers_down = (Controller.index_finger_down and 
                                         Controller.middle_finger_down and 
                                         Controller.ring_finger_down and 
                                         Controller.little_finger_down)
            
            Controller.all_fingers_up = (Controller.index_finger_up and 
                                       Controller.middle_finger_up and 
                                       Controller.ring_finger_up and 
                                       Controller.little_finger_up)
            
            # Finger-thumb proximity detection (for pinch gestures)
            thumb_tip = landmarks[4]
            
            # Using distance-based detection instead of just y-coordinate
            Controller.index_finger_within_thumb_finger = Controller._is_finger_near_thumb(landmarks[8], thumb_tip)
            Controller.middle_finger_within_thumb_finger = Controller._is_finger_near_thumb(landmarks[12], thumb_tip)
            Controller.ring_finger_within_thumb_finger = Controller._is_finger_near_thumb(landmarks[16], thumb_tip)
            Controller.little_finger_within_thumb_finger = Controller._is_finger_near_thumb(landmarks[20], thumb_tip)
            
        except (IndexError, AttributeError) as e:
            print(f"Error updating finger status: {e}")

    @staticmethod
    def _is_finger_near_thumb(finger_tip, thumb_tip, threshold=0.05):
        """
        Check if finger tip is close to thumb tip using Euclidean distance
        """
        distance = math.sqrt((finger_tip.x - thumb_tip.x)**2 + 
                           (finger_tip.y - thumb_tip.y)**2)
        return distance < threshold

    @staticmethod
    def get_position(hand_x_position, hand_y_position):
        """
        Dynamic position calculation with velocity-based movement
        """
        try:
            old_x, old_y = pyautogui.position()
            current_x = int(hand_x_position * Controller.screen_width)
            current_y = int(hand_y_position * Controller.screen_height)

            # Initialize previous hand position
            if Controller.prev_hand is None:
                Controller.prev_hand = (current_x, current_y)
                Controller._prev_time = pyautogui.time.time()
                return (old_x, old_y)  # Don't move on first detection
            
            # Calculate movement delta and velocity
            delta_x = current_x - Controller.prev_hand[0]
            delta_y = current_y - Controller.prev_hand[1]
            
            # Calculate time delta for velocity
            current_time = time.time()
            time_delta = current_time - getattr(Controller, '_prev_time', current_time)
            time_delta = max(time_delta, 0.001)  # Prevent division by zero
            
            # Calculate velocity (pixels per second)
            velocity_x = abs(delta_x) / time_delta
            velocity_y = abs(delta_y) / time_delta
            velocity = math.sqrt(velocity_x**2 + velocity_y**2)
            
            # Dynamic sensitivity based on velocity
            base_ratio = 1.5  # Base sensitivity
            velocity_multiplier = 1.0
            
            # Increase multiplier for faster movements
            if velocity > 100:  # Slow movement
                velocity_multiplier = 1.2
            elif velocity > 300:  # Medium movement
                velocity_multiplier = 2.0
            elif velocity > 600:  # Fast movement
                velocity_multiplier = 3.5
            elif velocity > 1000:  # Very fast movement
                velocity_multiplier = 5.0
            
            # Apply dynamic sensitivity
            dynamic_ratio = base_ratio * velocity_multiplier
            new_x = old_x + delta_x * dynamic_ratio
            new_y = old_y + delta_y * dynamic_ratio
            
            # Update previous position and time
            Controller.prev_hand = (current_x, current_y)
            Controller._prev_time = current_time

            # Improved boundary clamping
            threshold = 5
            new_x = max(threshold, min(new_x, Controller.screen_width - threshold))
            new_y = max(threshold, min(new_y, Controller.screen_height - threshold))

            return (new_x, new_y)
            
        except Exception as e:
            print(f"Error in get_position: {e}")
            return pyautogui.position()  # Return current position on error

    @staticmethod
    def cursor_moving():
        """
        Improved cursor movement with velocity-based acceleration, tracking the wrist for stability
        """
        if not Controller.hand_Landmarks or not Controller.hand_Landmarks.landmark:
            return
            
        try:
            # Use wrist (point 0) for more stable control
            TRACKING_POINT = 0  # Wrist landmark
            
            landmark = Controller.hand_Landmarks.landmark[TRACKING_POINT]
            current_x, current_y = landmark.x, landmark.y
            target_x, target_y = Controller.get_position(current_x, current_y)
            
            # Check if cursor should be frozen
            cursor_frozen = Controller.all_fingers_up and Controller.thumb_finger_down
            
            if cursor_frozen:
                return
                
            # Get current cursor position
            current_cursor_pos = pyautogui.position()
            
            # Calculate movement distance and velocity
            distance = math.sqrt((target_x - current_cursor_pos.x)**2 + 
                            (target_y - current_cursor_pos.y)**2)
            
            # Apply minimal smoothing only for very small movements
            if distance < 10 and Controller._prev_smooth_x is not None:
                target_x = (Controller._prev_smooth_x + 
                        (target_x - Controller._prev_smooth_x) * 0.7)
                target_y = (Controller._prev_smooth_y + 
                        (target_y - Controller._prev_smooth_y) * 0.7)
            
            # Move cursor with equation-based duration
            if distance > Controller._min_movement_threshold:
                duration = max(0.001, 0.01 / (1 + 0.05 * distance))  # Smooth duration curve
                pyautogui.moveTo(target_x, target_y, duration=duration)
            
            # Update smoothing history
            Controller._prev_smooth_x = target_x
            Controller._prev_smooth_y = target_y
                
        except pyautogui.FailSafeException:
            print("PyAutoGUI fail-safe triggered - move mouse to corner to stop")
        except Exception as e:
            print(f"Cursor movement error: {e}")

        
    @staticmethod
    def reset_smoothing():
        """
        Reset smoothing variables - call this when hand tracking is lost
        """
        Controller._prev_smooth_x = None
        Controller._prev_smooth_y = None
        Controller.prev_hand = None
        Controller._prev_time = None

    @staticmethod
    def set_movement_mode(mode="dynamic"):
        """
        Set movement mode: "dynamic" for velocity-based or "linear" for constant sensitivity
        """
        if mode in ["dynamic", "linear"]:
            Controller._movement_mode = mode
            print(f"Movement mode set to: {mode}")
        else:
            print("Invalid mode. Use 'dynamic' or 'linear'")

    @staticmethod
    def set_sensitivity(base_sensitivity=1.5, max_multiplier=5.0):
        """
        Adjust cursor movement sensitivity
        """
        Controller._base_sensitivity = max(0.5, min(base_sensitivity, 3.0))
        Controller._max_velocity_multiplier = max(1.0, min(max_multiplier, 10.0))
        print(f"Sensitivity set to: base={Controller._base_sensitivity}, max_multiplier={Controller._max_velocity_multiplier}")

    @staticmethod
    def set_smoothing(factor=0.15):
        """
        Adjust movement smoothing (0 = no smoothing, 0.9 = maximum smoothing)
        """
        Controller._smoothing_factor = max(0.0, min(factor, 0.9))
        print(f"Smoothing set to: {Controller._smoothing_factor}")

# Additional utility functions
def initialize_controller():
    """
    Initialize the controller with optimal settings
    """
    # Disable PyAutoGUI fail-safe if needed (be careful with this)
    # pyautogui.FAILSAFE = False
    
    # Set reasonable pause between PyAutoGUI commands
    pyautogui.PAUSE = 0.01
    
    print("Controller initialized successfully")

# Example usage:
# initialize_controller()
# Controller.set_sensitivity(1.2)  # Slightly more sensitive
# Controller.set_smoothing(0.4)    # Moderate smoothing