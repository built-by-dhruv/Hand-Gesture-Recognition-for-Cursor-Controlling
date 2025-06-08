from config import Config
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
    thumb_finger_down = None
    thumb_finger_up = None
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
    _smoothing_factor = Config.SMOOTHING_FACTOR
    _min_movement_threshold = Config.MIN_MOVEMENT_THRESHOLD
    _prev_smooth_x = None
    _prev_smooth_y = None
    _prev_time = None
    _movement_mode = "dynamic"

    @staticmethod
    def update_fingers_status():
        """
        Updates the status of all fingers based on hand landmarks
        """
        if not Controller.hand_Landmarks or not Controller.hand_Landmarks.landmark:
            return
            
        try:
            landmarks = Controller.hand_Landmarks.landmark
            
            Controller.little_finger_down = landmarks[20].y > landmarks[17].y
            Controller.little_finger_up = landmarks[20].y < landmarks[17].y
            
            Controller.index_finger_down = landmarks[8].y > landmarks[5].y
            Controller.index_finger_up = landmarks[8].y < landmarks[5].y
            
            Controller.middle_finger_down = landmarks[12].y > landmarks[9].y
            Controller.middle_finger_up = landmarks[12].y < landmarks[9].y
            
            Controller.ring_finger_down = landmarks[16].y > landmarks[13].y
            Controller.ring_finger_up = landmarks[16].y < landmarks[13].y
            
            Controller.thumb_finger_down = landmarks[4].y > landmarks[3].y
            Controller.thumb_finger_up = landmarks[4].y < landmarks[3].y
            
            Controller.all_fingers_down = (Controller.index_finger_down and 
                                         Controller.middle_finger_down and 
                                         Controller.ring_finger_down and 
                                         Controller.little_finger_down)
            
            Controller.all_fingers_up = (Controller.index_finger_up and 
                                       Controller.middle_finger_up and 
                                       Controller.ring_finger_up and 
                                       Controller.little_finger_up)
            
            thumb_tip = landmarks[4]
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
        Dynamic position calculation with velocity and acceleration (rate of change in position)
        Uses a continuous equation for velocity multiplier based on both velocity and acceleration.
        """
        try:
            old_x, old_y = pyautogui.position()
            current_x = int(hand_x_position * Controller.screen_width)
            current_y = int(hand_y_position * Controller.screen_height)

            if Controller.prev_hand is None:
                Controller.prev_hand = (current_x, current_y)
                Controller._prev_time = time.time()
                Controller._prev_velocity = (0, 0)
                return (old_x, old_y)
            
            delta_x = current_x - Controller.prev_hand[0]
            delta_y = current_y - Controller.prev_hand[1]
            
            current_time = time.time()
            time_delta = current_time - getattr(Controller, '_prev_time', current_time)
            time_delta = max(time_delta, 0.001)
            
            velocity_x = delta_x / time_delta
            velocity_y = delta_y / time_delta
            velocity = math.sqrt(velocity_x**2 + velocity_y**2)

            # Calculate acceleration (rate of change of velocity)
            prev_velocity_x, prev_velocity_y = getattr(Controller, '_prev_velocity', (0, 0))
            acceleration_x = (velocity_x - prev_velocity_x) / time_delta
            acceleration_y = (velocity_y - prev_velocity_y) / time_delta
            acceleration = math.sqrt(acceleration_x**2 + acceleration_y**2)

            # Continuous multiplier equation using both velocity and acceleration
            base_ratio = Config.BASE_RATIO
            alpha = Config.ALPHA  # velocity influence
            beta = Config.BETA   # acceleration influence
            velocity_multiplier = 1 + alpha * velocity + beta * acceleration

            dynamic_ratio = base_ratio * velocity_multiplier
            new_x = old_x + delta_x * dynamic_ratio
            new_y = old_y + delta_y * dynamic_ratio

            Controller.prev_hand = (current_x, current_y)
            Controller._prev_time = current_time
            Controller._prev_velocity = (velocity_x, velocity_y)

            threshold = 5
            new_x = max(threshold, min(new_x, Controller.screen_width - threshold))
            new_y = max(threshold, min(new_y, Controller.screen_height - threshold))

            return (new_x, new_y)
            
        except Exception as e:
            print(f"Error in get_position: {e}")
            return pyautogui.position()

    @staticmethod
    def cursor_moving():
        """
        Improved cursor movement with velocity-based acceleration, tracking the wrist for stability
        """
        if not Controller.hand_Landmarks or not Controller.hand_Landmarks.landmark:
            return
            
        try:
            TRACKING_POINT = 0  # Wrist landmark
            
            landmark = Controller.hand_Landmarks.landmark[TRACKING_POINT]
            current_x, current_y = landmark.x, landmark.y
            target_x, target_y = Controller.get_position(current_x, current_y)
            
            cursor_frozen = Controller.all_fingers_up and Controller.thumb_finger_down
            
            if cursor_frozen:
                return
                
            current_cursor_pos = pyautogui.position()
            
            distance = math.sqrt((target_x - current_cursor_pos.x)**2 + 
                               (target_y - current_cursor_pos.y)**2)
            
            if distance < 10 and Controller._prev_smooth_x is not None and Controller._prev_smooth_y is not None:
                target_x = (Controller._prev_smooth_x + 
                          (target_x - Controller._prev_smooth_x) * 0.9)
                target_y = (Controller._prev_smooth_y + 
                          (target_y - Controller._prev_smooth_y) * 0.9)
            
            if distance > Controller._min_movement_threshold:
                duration = max(Config.MIN_DURATION, Config.BASE_DURATION / (5 + Config.K * distance))
                print(f"Distance: {distance:.2f}, Duration: {duration:.4f}")  # Debugging output
                pyautogui.moveTo(target_x, target_y, duration=duration)
            
            Controller._prev_smooth_x = target_x
            Controller._prev_smooth_y = target_y
                
        except pyautogui.FailSafeException:
            print("PyAutoGUI fail-safe triggered - move mouse to corner to stop")
        except Exception as e:
            print(f"Cursor movement error: {e}")

    @staticmethod
    def reset_smoothing():
        """
        Reset smoothing variables
        """
        Controller._prev_smooth_x = None
        Controller._prev_smooth_y = None
        Controller.prev_hand = None
        Controller._prev_time = None

    @staticmethod
    def set_movement_mode(mode="dynamic"):
        """
        Set movement mode: 'dynamic' or 'linear'
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
        Controller._max_velocity_multiplier = max(10.0, min(max_multiplier, 50.0))
        print(f"Sensitivity set to: base={Controller._base_sensitivity}, max_multiplier={Controller._max_velocity_multiplier}")

    @staticmethod
    def set_smoothing(factor=Config.SMOOTHING_FACTOR):
        """
        Adjust movement smoothing
        """
        Controller._smoothing_factor = max(0.0, min(factor, 0.9))
        print(f"Smoothing set to: {Controller._smoothing_factor}")

def initialize_controller():
    """
    Initialize the controller with optimal settings
    """
    pyautogui.PAUSE = Config.PAUSE
    Controller.set_smoothing(Config.SMOOTHING_FACTOR)
    print("Controller initialized successfully")