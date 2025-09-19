from config import Config
import pyautogui
import math
import time
import threading

class Controller:
    # State variables
    prev_hand = None
    right_clicked = False
    left_clicked = False
    double_clicked = False
    dragging = False
    hand_Landmarks = None

    # Finger status
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

    # Gesture detection
    index_finger_within_thumb_finger = None
    middle_finger_within_thumb_finger = None
    little_finger_within_thumb_finger = None
    ring_finger_within_thumb_finger = None

    # Screen dimensions
    screen_width, screen_height = pyautogui.size()

    # Movement control
    _prev_smooth_x = None
    _prev_smooth_y = None
    _prev_time = None
    _prev_velocity = (0, 0)
    _movement_mode = "dynamic"
    _smoothed_hand_x = None
    _smoothed_hand_y = None
    _prev_cursor_x = None
    _prev_cursor_y = None

    # Config sync lock
    _config_lock = threading.Lock()

    @staticmethod
    def reload_config():
        """
        Reloads all runtime config values from Config in a thread-safe way.
        """
        with Controller._config_lock:
            pyautogui.PAUSE = Config.PAUSE
            # Config values are accessed directly now

    @staticmethod
    def update_fingers_status():
        """
        Updates the status of all fingers based on hand landmarks
        """
        if not Controller.hand_Landmarks or not Controller.hand_Landmarks.landmark:
            return
        try:
            lm = Controller.hand_Landmarks.landmark
            Controller.little_finger_down = lm[20].y > lm[17].y
            Controller.little_finger_up = lm[20].y < lm[17].y
            Controller.index_finger_down = lm[8].y > lm[5].y
            Controller.index_finger_up = lm[8].y < lm[5].y
            Controller.middle_finger_down = lm[12].y > lm[9].y
            Controller.middle_finger_up = lm[12].y < lm[9].y
            Controller.ring_finger_down = lm[16].y > lm[13].y
            Controller.ring_finger_up = lm[16].y < lm[13].y
            Controller.thumb_finger_down = lm[4].y > lm[3].y
            Controller.thumb_finger_up = lm[4].y < lm[3].y
            Controller.all_fingers_down = (
                Controller.index_finger_down and Controller.middle_finger_down and
                Controller.ring_finger_down and Controller.little_finger_down)
            Controller.all_fingers_up = (
                Controller.index_finger_up and Controller.middle_finger_up and
                Controller.ring_finger_up and Controller.little_finger_up)
            thumb_tip = lm[4]
            Controller.index_finger_within_thumb_finger = Controller._is_finger_near_thumb(lm[8], thumb_tip)
            Controller.middle_finger_within_thumb_finger = Controller._is_finger_near_thumb(lm[12], thumb_tip)
            Controller.ring_finger_within_thumb_finger = Controller._is_finger_near_thumb(lm[16], thumb_tip)
            Controller.little_finger_within_thumb_finger = Controller._is_finger_near_thumb(lm[20], thumb_tip)
        except (IndexError, AttributeError) as e:
            print(f"Error updating finger status: {e}")

    @staticmethod
    def _is_finger_near_thumb(finger_tip, thumb_tip, threshold=0.05):
        """
        Check if finger tip is close to thumb tip using Euclidean distance
        """
        dx = finger_tip.x - thumb_tip.x
        dy = finger_tip.y - thumb_tip.y
        return math.hypot(dx, dy) < threshold

    @staticmethod
    def get_position(hand_x_position, hand_y_position):
        """Return relative cursor delta using smoothing and sensitivity-based scaling."""
        try:
            with Controller._config_lock:
                current_time = time.time()
                
                # Initialize on first run
                if Controller._prev_cursor_x is None or Controller._prev_time is None:
                    Controller._prev_cursor_x = hand_x_position * Controller.screen_width
                    Controller._prev_cursor_y = hand_y_position * Controller.screen_height
                    Controller._prev_time = current_time
                    return (0, 0)

                # Calculate delta time
                dt = current_time - Controller._prev_time
                if dt == 0:
                    return (0, 0)

                # 1. Exponential Smoothing on raw hand position
                alpha = Config.SMOOTHING_FACTOR
                x_smooth = alpha * (hand_x_position * Controller.screen_width) + (1 - alpha) * Controller._prev_cursor_x
                y_smooth = alpha * (hand_y_position * Controller.screen_height) + (1 - alpha) * Controller._prev_cursor_y

                # 2. Relative delta scaled by sensitivity (no extra acceleration)
                dx_raw = x_smooth - Controller._prev_cursor_x
                dy_raw = y_smooth - Controller._prev_cursor_y
                delta_x = dx_raw * Config.SENSITIVITY
                delta_y = dy_raw * Config.SENSITIVITY

                # Update previous state for next frame
                Controller._prev_cursor_x = x_smooth
                Controller._prev_cursor_y = y_smooth
                Controller._prev_time = current_time

                return (delta_x, delta_y)
        except Exception as e:
            print(f"Error in get_position: {e}")
            return (0, 0)

    @staticmethod
    def cursor_moving():
        """
        Moves the cursor based on hand gestures, using a velocity-based model.
        Movement is only active when thumb and index finger are pinched.
        """
        if not Controller.hand_Landmarks or not Controller.hand_Landmarks.landmark:
            return

        # Only move cursor if thumb and index finger are pinched
        if not Controller.index_finger_within_thumb_finger:
            # Reset state when not pinching to prevent jumps on re-pinch
            Controller._prev_time = None
            Controller._prev_cursor_x = None
            Controller._prev_cursor_y = None
            return
            
        try:
            # Use the midpoint of the ring and little finger tips for tracking
            ring_finger_tip = Controller.hand_Landmarks.landmark[16]
            little_finger_tip = Controller.hand_Landmarks.landmark[20]
            
            current_x = (ring_finger_tip.x + little_finger_tip.x) / 2
            current_y = (ring_finger_tip.y + little_finger_tip.y) / 2
            
            delta_x, delta_y = Controller.get_position(current_x, current_y)
            
            # The movement threshold is now applied to the calculated delta
            if math.hypot(delta_x, delta_y) > Config.MIN_MOVEMENT_THRESHOLD:
                pyautogui.move(delta_x, delta_y)

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
        Controller._prev_velocity = (0, 0)
        Controller._smoothed_hand_x = None
        Controller._smoothed_hand_y = None
        Controller._prev_cursor_x = None
        Controller._prev_cursor_y = None

    @staticmethod
    def set_movement_mode(mode="dynamic"):
        if mode in ["dynamic", "linear"]:
            Controller._movement_mode = mode
            print(f"Movement mode set to: {mode}")
        else:
            print("Invalid mode. Use 'dynamic' or 'linear'")

    @staticmethod
    def set_sensitivity(base_sensitivity=1.5, max_multiplier=5.0):
        Controller._base_sensitivity = max(0.5, min(base_sensitivity, 3.0))
        Controller._max_velocity_multiplier = max(10.0, min(max_multiplier, 50.0))
        print(f"Sensitivity set to: base={Controller._base_sensitivity}, max_multiplier={Controller._max_velocity_multiplier}")

    @staticmethod
    def set_smoothing(factor=Config.SMOOTHING_FACTOR):
        Controller._smoothing_factor = max(0.0, min(factor, 0.9))
        print(f"Smoothing set to: {Controller._smoothing_factor}")

def initialize_controller():
    """
    Initialize controller and set config-dependent runtime variables.
    """
    pyautogui.PAUSE = Config.PAUSE
    Controller.set_smoothing(Config.SMOOTHING_FACTOR)
    # No need to reload, Config is shared