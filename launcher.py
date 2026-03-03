# launcher.py
import os, sys, time, signal, logging, subprocess
from collections import deque

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [SUPERVISOR] %(message)s",
                    handlers=[logging.StreamHandler()])
log = logging.getLogger("Supervisor")

MAX_RESTARTS_PER_HOUR = 10
RESTART_DELAY         = 5
UI_PORT               = os.getenv("PORT", "8501")

bot_process = None
ui_process  = None
running     = True
bot_restarts: deque = deque(maxlen=MAX_RESTARTS_PER_HOUR)
ui_restarts:  deque = deque(maxlen=MAX_RESTARTS_PER_HOUR)


def start_bot():
    log.info("Starting bot.py...")
    return subprocess.Popen([sys.executable, "bot.py"],
                            stdout=sys.stdout, stderr=sys.stderr)


def start_ui():
    log.info("Starting Streamlit on port %s...", UI_PORT)
    return subprocess.Popen([
        sys.executable, "-m", "streamlit", "run", "app.py",
        "--server.port",              UI_PORT,
        "--server.address",           "0.0.0.0",
        "--server.headless",          "true",
        "--server.runOnSave",         "false",
        "--browser.gatherUsageStats", "false",
    ], stdout=sys.stdout, stderr=sys.stderr)


def restarts_this_hour(history):
    now = time.time()
    return sum(1 for t in history if now - t < 3600)


def shutdown(signum, frame):
    global running
    log.info("Shutdown received")
    running = False
    if bot_process: bot_process.terminate()
    if ui_process:  ui_process.terminate()
    sys.exit(0)


signal.signal(signal.SIGTERM, shutdown)
signal.signal(signal.SIGINT,  shutdown)

bot_process = start_bot()
ui_process  = start_ui()
log.info("Both processes started.")

while running:
    time.sleep(5)
    if bot_process.poll() is not None:
        log.error("bot.py died (exit=%d) — restarting", bot_process.returncode)
        if restarts_this_hour(bot_restarts) >= MAX_RESTARTS_PER_HOUR:
            log.error("Crash loop — waiting 5min"); time.sleep(300)
        time.sleep(RESTART_DELAY)
        bot_restarts.append(time.time())
        bot_process = start_bot()
    if ui_process.poll() is not None:
        log.warning("Streamlit died (exit=%d) — restarting", ui_process.returncode)
        if restarts_this_hour(ui_restarts) >= MAX_RESTARTS_PER_HOUR:
            log.warning("UI crash loop — waiting 5min"); time.sleep(300)
        time.sleep(RESTART_DELAY)
        ui_restarts.append(time.time())
        ui_process = start_ui()
