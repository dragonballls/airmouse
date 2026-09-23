from __future__ import annotations

import os
import sys
import threading
import tkinter as tk
from tkinter import messagebox

if getattr(sys, "frozen", False):
    os.chdir(getattr(sys, "_MEIPASS"))
else:
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

import main


class AirMouseApp:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("AI Air Mouse")
        self.root.geometry("390x190")
        self.root.resizable(False, False)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self.stop_event = None
        self.worker = None
        self.error = None
        self.closing = False

        frame = tk.Frame(self.root, padx=22, pady=18)
        frame.pack(fill="both", expand=True)

        tk.Label(frame, text="AI Air Mouse", font=("Segoe UI", 18, "bold")).pack()

        self.status = tk.Label(frame, text="Starting...", font=("Segoe UI", 10))
        self.status.pack(pady=(8, 14))

        buttons = tk.Frame(frame)
        buttons.pack()

        self.start_button = tk.Button(buttons, text="Start", width=10, command=self.start)
        self.start_button.grid(row=0, column=0, padx=5)

        self.stop_button = tk.Button(
            buttons, text="Stop", width=10, command=self.stop, state=tk.DISABLED
        )
        self.stop_button.grid(row=0, column=1, padx=5)

        self.exit_button = tk.Button(buttons, text="Exit", width=10, command=self.close)
        self.exit_button.grid(row=0, column=2, padx=5)

        self.start()

    def start(self) -> None:
        if self.worker is not None and self.worker.is_alive():
            return

        self.stop_event = threading.Event()
        self.error = None
        self.closing = False

        self.status.config(text="Starting camera and hand tracking...")
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)

        self.worker = threading.Thread(
            target=self._worker_main,
            name="AirmouseWorker",
            daemon=False,
        )
        self.worker.start()
        self.root.after(100, self._poll)

    def _worker_main(self) -> None:
        try:
            assert self.stop_event is not None
            main.run(self.stop_event)
        except BaseException as exc:
            self.error = exc

    def stop(self) -> None:
        if self.worker is None or not self.worker.is_alive():
            self.status.config(text="Stopped")
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            return

        assert self.stop_event is not None
        self.stop_event.set()
        self.status.config(text="Stopping and restoring Windows settings...")
        self.stop_button.config(state=tk.DISABLED)

    def _poll(self) -> None:
        if self.worker is not None and self.worker.is_alive():
            if not self.closing:
                self.status.config(text="Running - hand control is active")
            self.root.after(150, self._poll)
            return

        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)

        if self.error is not None:
            self.status.config(text="Stopped because of an error")
            if not self.closing:
                messagebox.showerror(
                    "AI Air Mouse",
                    f"The application stopped because of an error:\n\n{self.error}",
                    parent=self.root,
                )
        else:
            self.status.config(text="Stopped")

        if self.closing:
            self.root.destroy()

    def close(self) -> None:
        self.closing = True

        if self.worker is not None and self.worker.is_alive():
            assert self.stop_event is not None
            self.stop_event.set()
            self.status.config(text="Stopping safely...")
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.DISABLED)
            self.exit_button.config(state=tk.DISABLED)
            self.root.after(100, self._poll)
        else:
            self.root.destroy()


if __name__ == "__main__":
    AirMouseApp().root.mainloop()
