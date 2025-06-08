import gradio as gr
import threading
from config import Config, save_config
import app

controller_thread = None
controller_running = False

def update_config(min_duration, base_duration, k, pause, smoothing_factor, min_movement_threshold, base_ratio, alpha, beta):
    cfg = {
        "MIN_DURATION": min_duration,
        "BASE_DURATION": base_duration,
        "K": k,
        "PAUSE": pause,
        "SMOOTHING_FACTOR": smoothing_factor,
        "MIN_MOVEMENT_THRESHOLD": min_movement_threshold,
        "BASE_RATIO": base_ratio,
        "ALPHA": alpha,
        "BETA": beta
    }
    Config.update_from_dict(cfg)
    save_config(cfg)
    return f"Config updated and saved!\nMIN_DURATION={min_duration}\nBASE_DURATION={base_duration}\nK={k}\nPAUSE={pause}\nSMOOTHING_FACTOR={smoothing_factor}\nMIN_MOVEMENT_THRESHOLD={min_movement_threshold}\nBASE_RATIO={base_ratio}\nALPHA={alpha}\nBETA={beta}"

def start_controller():
    global controller_thread, controller_running
    if controller_running:
        return "Controller already running."
    def run_app():
        try:
            app.main()
        except Exception as e:
            print(f"Controller stopped: {e}")
    controller_thread = threading.Thread(target=run_app, daemon=True)
    controller_thread.start()
    controller_running = True
    return "Controller started!"

def stop_controller():
    global controller_running
    controller_running = False
    return "To stop the controller, close the app window or press ESC in the app window."

with gr.Blocks() as demo:
    gr.Markdown("# Cursor Controller Config & Launcher")
    with gr.Row():
        min_duration = gr.Slider(0.00001, 0.01, value=Config.MIN_DURATION, label="MIN_DURATION (snappiness)")
        base_duration = gr.Slider(0.0001, 0.01, value=Config.BASE_DURATION, label="BASE_DURATION (base speed)")
        k = gr.Slider(0.1, 50, value=Config.K, label="K (duration decay)")
        pause = gr.Slider(0.0001, 0.01, value=Config.PAUSE, label="PAUSE (responsiveness)")
    with gr.Row():
        smoothing_factor = gr.Slider(0.0, 0.9, value=Config.SMOOTHING_FACTOR, label="SMOOTHING_FACTOR (smoothness)")
        min_movement_threshold = gr.Slider(1, 30, value=Config.MIN_MOVEMENT_THRESHOLD, label="MIN_MOVEMENT_THRESHOLD (jitter ignore)")
        base_ratio = gr.Slider(0.1, 5.0, value=Config.BASE_RATIO, label="BASE_RATIO (sensitivity)")
    with gr.Row():
        alpha = gr.Slider(0.0, 0.00001, value=Config.ALPHA, label="ALPHA (velocity effect)")
        beta = gr.Slider(0.0, 0.01, value=Config.BETA, label="BETA (acceleration effect)")
    update_btn = gr.Button("Update Config")
    start_btn = gr.Button("Start Controller")
    stop_btn = gr.Button("Stop Controller")
    output = gr.Textbox(label="Status")
    update_btn.click(fn=update_config, inputs=[min_duration, base_duration, k, pause, smoothing_factor, min_movement_threshold, base_ratio, alpha, beta], outputs=output)
    start_btn.click(fn=start_controller, outputs=output)
    stop_btn.click(fn=stop_controller, outputs=output)

demo.launch()
