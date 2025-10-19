#!/usr/bin/env python3
"""
Simple HTTP server to serve the trading dashboard.
"""

import http.server
import socketserver
import os
import webbrowser
from threading import Timer

PORT = 8081

class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=os.path.dirname(os.path.abspath(__file__)), **kwargs)
    
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

def open_browser():
    """Open the dashboard in the default browser."""
    webbrowser.open(f'http://localhost:{PORT}')

def main():
    """Start the dashboard server."""
    print(f"Starting trading dashboard server on port {PORT}...")
    print(f"Dashboard will be available at: http://localhost:{PORT}")
    print("Press Ctrl+C to stop the server")
    
    try:
        with socketserver.TCPServer(("", PORT), DashboardHandler) as httpd:
            # Open browser after a short delay
            Timer(1.0, open_browser).start()
            
            print(f"Server started successfully!")
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
    except Exception as e:
        print(f"Error starting server: {e}")

if __name__ == "__main__":
    main()
