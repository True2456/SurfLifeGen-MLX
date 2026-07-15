# surflifegen/web_cli.py
"""
Command-Line Interface to boot the SurfLifeGen-MLX web dashboard server
and auto-launch the web browser.
"""
import time
import webbrowser
import threading
import argparse
from .app import start_server

def open_browser(url: str, delay: float = 1.5):
    time.sleep(delay)
    print(f"👉 Opening dashboard in browser: {url}")
    webbrowser.open(url)

def main():
    parser = argparse.ArgumentParser(
        description="SurfLifeGen-MLX Web Dashboard Launcher"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host address to bind the server (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to run the server on (default: 8000)"
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Skip auto-launching the web browser"
    )

    args = parser.parse_args()
    
    url = f"http://{args.host}:{args.port}"
    print("\n=======================================================")
    print("🚀 STARTING SURFLIFEGEN-MLX WEB DASHBOARD CONTROL PANEL")
    print(f"🔗 Local API server running at: {url}")
    print("=======================================================\n")

    if not args.no_browser:
        # Start browser launcher thread with a slight delay so uvicorn has time to bind
        threading.Thread(target=open_browser, args=(url, 1.2), daemon=True).start()

    start_server(host=args.host, port=args.port)

if __name__ == "__main__":
    main()
