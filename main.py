import gradio as gr
import threading
from config import Config, save_config
import app

controller_thread = None


def update_config(pause, smoothing_factor, min_movement_threshold, sensitivity, target_fps, failsafe, invert_hands):
    """Update configuration parameters from the Gradio UI and persist them."""
    cfg = {
        "PAUSE": pause,
        "SMOOTHING_FACTOR": max(0.0, min(float(smoothing_factor), 1.0)),
        "MIN_MOVEMENT_THRESHOLD": min_movement_threshold,
        "SENSITIVITY": sensitivity,
        "TARGET_FPS": int(target_fps),
    "FAILSAFE": bool(failsafe),
    "INVERT_HANDS": bool(invert_hands),
    }
    Config.update_from_dict(cfg)
    save_config(cfg)
    msg = (
        f"Config updated and saved!\n"
        f"PAUSE={pause}\nSMOOTHING_FACTOR={cfg['SMOOTHING_FACTOR']}\n"
        f"MIN_MOVEMENT_THRESHOLD={min_movement_threshold}\nSENSITIVITY={sensitivity}\nTARGET_FPS={int(target_fps)}\n"
    f"FAILSAFE={cfg['FAILSAFE']}\n"
    f"INVERT_HANDS={cfg['INVERT_HANDS']}\n"
    )
    print(msg)
    return msg

def start_controller():
    """
    Start the hand gesture controller in a background thread.
    """
    global controller_thread
    with Config.lock:
        if Config.running:
            return "Controller already running."
        Config.running = True
    def run_app():
        try:
            app.main()
        except Exception as e:
            print(f"Controller stopped: {e}")
            with Config.lock:
                Config.running = False
    controller_thread = threading.Thread(target=run_app, daemon=True)
    controller_thread.start()
    return "Controller started!"

def stop_controller():
    """
    Stop the controller by setting the running flag to False.
    """
    with Config.lock:
        Config.running = False
    return "Controller stopping... (may take a moment)"

def get_status():
    with Config.lock:
        if Config.running:
            return "Controller is RUNNING"
        else:
            return "Controller is STOPPED"

def reset_smoothing():
    from controller import Controller
    Controller.reset_smoothing()
    return "Smoothing reset!"

with gr.Blocks() as demo:
    gr.Markdown("# Cursor Controller Config & Launcher")
    status = gr.Textbox(label="Controller Status", value=get_status(), interactive=False)
    with gr.Row():
        pause = gr.Slider(0.0001, 0.01, value=Config.PAUSE, label="PAUSE (pyautogui pause)", step=0.00001)
    smoothing_factor = gr.Slider(0.0, 1.0, value=Config.SMOOTHING_FACTOR, label="SMOOTHING_FACTOR (alpha 0–1)", step=0.01)
    with gr.Row():
        min_movement_threshold = gr.Slider(0, 100, value=Config.MIN_MOVEMENT_THRESHOLD, label="MIN_MOVEMENT_THRESHOLD (dead zone)", step=1)
        sensitivity = gr.Slider(0.1, 50.0, value=Config.SENSITIVITY, label="SENSITIVITY (speed multiplier)", step=0.05)
        target_fps = gr.Slider(5, 60, value=Config.TARGET_FPS, label="TARGET_FPS (camera pacing)", step=1)
    failsafe = gr.Checkbox(value=getattr(Config, 'FAILSAFE', True), label="Enable PyAutoGUI FAILSAFE (corner abort)")
    invert_hands = gr.Checkbox(value=getattr(Config, 'INVERT_HANDS', False), label="Invert hands (left=move, right=clicks)")
    update_btn = gr.Button("Update Config")
    start_btn = gr.Button("Start Controller")
    stop_btn = gr.Button("Stop Controller")
    reset_btn = gr.Button("Reset Smoothing")
    output = gr.Textbox(label="Status")
    update_btn.click(fn=update_config, inputs=[pause, smoothing_factor, min_movement_threshold, sensitivity, target_fps, failsafe, invert_hands], outputs=output)
    start_btn.click(fn=start_controller, outputs=output)
    stop_btn.click(fn=stop_controller, outputs=output)
    reset_btn.click(fn=reset_smoothing, outputs=output)

demo.launch()
