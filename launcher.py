import subprocess
import sys
import webbrowser
import time
import threading

_browser_opened = False

def open_browser():
    global _browser_opened
    if not _browser_opened:
        _browser_opened = True
        time.sleep(4)
        webbrowser.open("http://localhost:8501")

if __name__ == "__main__":
    threading.Thread(target=open_browser, daemon=True).start()
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        "finsight_main.py",
        "--server.headless=true",
        "--server.port=8501"
    ])