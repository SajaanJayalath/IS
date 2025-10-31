import os
import threading
from typing import Optional, List, Tuple

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from tkinter import scrolledtext
from PIL import Image, ImageDraw, ImageTk

from preprocess import segment_symbols, segment_symbols_with_debug
from predict import predict_symbols_detailed
from evaluate import safe_eval


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Handwritten Math Expression Recognition and Solver")
        self.geometry("1200x700")

        # Initialize drawing config before building the UI so
        # attributes are available during widget creation.
        # (Previously these were defined after _build_ui, causing
        # AttributeError when the Canvas used self.canvas_width/height.)
        self.canvas_width = 680
        self.canvas_height = 360
        self.stroke_width = 12
        self.pen_color = "black"
        self.bg_color = "white"
        # Drawing mode
        self.mode = 'draw'  # 'draw' or 'erase'
        self.eraser_width = 16
        # Offscreen image for clean capture
        self.pil_image = Image.new("RGB", (self.canvas_width, self.canvas_height), self.bg_color)
        self.draw = ImageDraw.Draw(self.pil_image)

        # Build UI and bind events
        self._build_ui()
        self._bind_events()

        self.last_x: Optional[int] = None
        self.last_y: Optional[int] = None

    def _build_ui(self):
        container = ttk.Frame(self)
        container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Layout: Canvas on top (full row), other panels below
        top = ttk.LabelFrame(container, text="Input Methods")
        top.pack(fill=tk.X, padx=0, pady=(0, 8))

        bottom = ttk.Frame(container)
        bottom.pack(fill=tk.BOTH, expand=True)

        center = ttk.LabelFrame(bottom, text="Settings & Controls")
        right = ttk.LabelFrame(bottom, text="Recognition Results")
        center.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(8, 0))

        # Left: drawing area
        ttk.Label(top, text="Draw expression:").pack(anchor=tk.W, padx=6, pady=(6, 2))
        self.canvas = tk.Canvas(top, bg="white", width=self.canvas_width, height=self.canvas_height,
                                highlightthickness=1, highlightbackground="#ccc")
        # Stretch horizontally with the window
        self.canvas.pack(fill=tk.X, expand=True, padx=6, pady=(0, 8))
        # Keep offscreen buffer in sync when resized
        self.canvas.bind("<Configure>", self._on_canvas_resize)

        btn_row = ttk.Frame(top)
        btn_row.pack(fill=tk.X, padx=6, pady=(0, 8))
        self.btn_clear = ttk.Button(btn_row, text="Clear Canvas", command=self.clear_canvas)
        self.btn_clear.pack(side=tk.LEFT)
        # Eraser toggle button
        self.btn_eraser = ttk.Button(btn_row, text="Eraser: Off", command=self.toggle_eraser)
        self.btn_eraser.pack(side=tk.LEFT, padx=(8, 0))
        self.btn_recognize = ttk.Button(btn_row, text="Recognize Drawing", command=self.on_recognize)
        self.btn_recognize.pack(side=tk.LEFT, padx=(8, 0))

        ttk.Separator(top, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=6, pady=6)
        ttk.Label(top, text="Or upload an image:").pack(anchor=tk.W, padx=6)
        ttk.Button(top, text="Upload Image File", command=self.on_upload_image).pack(anchor=tk.W, padx=6, pady=(4, 6))

        # Center: controls
        header = ttk.Label(center, text="Handwritten Math Expression Solver", font=("Segoe UI", 14, "bold"))
        header.pack(pady=(6, 10))

        recog_frame = ttk.LabelFrame(center, text="Recognition Target")
        recog_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        self.recog_var = tk.StringVar(value="expr")
        self.rb_digits = ttk.Radiobutton(recog_frame, text="Digits (0-9)", value="digits", variable=self.recog_var, state=tk.DISABLED)
        self.rb_letters = ttk.Radiobutton(recog_frame, text="Letters (A-Z, a-z)", value="letters", variable=self.recog_var, state=tk.DISABLED)
        self.rb_expr = ttk.Radiobutton(recog_frame, text="Arithmetic Expressions", value="expr", variable=self.recog_var)
        self.rb_digits.grid(row=0, column=0, padx=8, pady=6, sticky=tk.W)
        self.rb_letters.grid(row=0, column=1, padx=8, pady=6, sticky=tk.W)
        self.rb_expr.grid(row=0, column=2, padx=8, pady=6, sticky=tk.W)

        model_frame = ttk.Frame(center)
        model_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        ttk.Label(model_frame, text="Select Model:").grid(row=0, column=0, sticky=tk.W)
        self.model_cmb = ttk.Combobox(model_frame, values=["CNN"], state="readonly")
        self.model_cmb.current(0)
        self.model_cmb.grid(row=0, column=1, sticky=tk.W, padx=(6, 0))
        ttk.Label(model_frame, text="Segmentation:").grid(row=1, column=0, sticky=tk.W, pady=(8, 0))
        # Display-only: we now use a connected-components based method
        self.seg_cmb = ttk.Combobox(model_frame, values=["Smart (Components)"], state="disabled")
        self.seg_cmb.current(0)
        self.seg_cmb.grid(row=1, column=1, sticky=tk.W, padx=(6, 0), pady=(8, 0))

        options = ttk.Frame(center)
        options.pack(fill=tk.X, padx=10, pady=(0, 10))
        self.var_show_pre = tk.BooleanVar(value=True)
        ttk.Checkbutton(options, text="Show preprocessing steps", variable=self.var_show_pre).grid(row=0, column=0, sticky=tk.W)

        # Right: results panel
        self.result_text = scrolledtext.ScrolledText(right, width=42, height=28)
        self.result_text.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        # Bottom: image preview
        preview = ttk.LabelFrame(self, text="Image Preview")
        preview.pack(fill=tk.X, padx=10, pady=(0, 10))
        self.preview_img_label = ttk.Label(preview, text="No image loaded")
        self.preview_img_label.pack(side=tk.LEFT, padx=10, pady=8)

        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status = ttk.Label(self, textvariable=self.status_var, anchor=tk.W, relief=tk.SUNKEN)
        status.pack(fill=tk.X, side=tk.BOTTOM)

    def _bind_events(self):
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)

    def _on_canvas_resize(self, event):
        # Adjust the backing PIL image to match the canvas size so
        # recognition sees the full drawing. Preserve existing pixels.
        new_w = max(1, int(event.width))
        new_h = max(1, int(event.height))
        if new_w == self.canvas_width and new_h == self.canvas_height:
            return
        old_img = self.pil_image
        self.canvas_width, self.canvas_height = new_w, new_h
        new_img = Image.new("RGB", (new_w, new_h), self.bg_color)
        # paste overlapping region to preserve previous drawing
        ow, oh = old_img.size
        box_w, box_h = min(ow, new_w), min(oh, new_h)
        if box_w > 0 and box_h > 0:
            new_img.paste(old_img.crop((0, 0, box_w, box_h)), (0, 0))
        self.pil_image = new_img
        self.draw = ImageDraw.Draw(self.pil_image)

    def _on_press(self, event):
        self.last_x, self.last_y = event.x, event.y
        # Draw a small dot
        width = self.eraser_width if self.mode == 'erase' else self.stroke_width
        color = self.bg_color if self.mode == 'erase' else self.pen_color
        r = width // 2
        self.canvas.create_oval(event.x - r, event.y - r, event.x + r, event.y + r, fill=color, outline=color)
        self.draw.ellipse((event.x - r, event.y - r, event.x + r, event.y + r), fill=color)

    def _on_drag(self, event):
        if self.last_x is None or self.last_y is None:
            self.last_x, self.last_y = event.x, event.y
        width = self.eraser_width if self.mode == 'erase' else self.stroke_width
        color = self.bg_color if self.mode == 'erase' else self.pen_color
        self.canvas.create_line(self.last_x, self.last_y, event.x, event.y, fill=color, width=width, capstyle=tk.ROUND, smooth=True)
        self.draw.line((self.last_x, self.last_y, event.x, event.y), fill=color, width=width)
        self.last_x, self.last_y = event.x, event.y

    def _on_release(self, _event):
        self.last_x, self.last_y = None, None

    def clear_canvas(self):
        self.canvas.delete("all")
        self.pil_image.paste(self.bg_color, (0, 0, self.canvas_width, self.canvas_height))
        self.result_text.delete("1.0", tk.END)
        # Reset eraser state
        self.mode = 'draw'
        try:
            self.btn_eraser.configure(text="Eraser: Off")
            self.canvas.config(cursor='pencil')
        except Exception:
            pass
        self.status_var.set("Ready")

    def toggle_eraser(self):
        # Toggle draw/erase mode and update UI
        self.mode = 'erase' if self.mode != 'erase' else 'draw'
        is_erase = self.mode == 'erase'
        try:
            self.btn_eraser.configure(text=f"Eraser: {'On' if is_erase else 'Off'}")
            self.canvas.config(cursor='circle' if is_erase else 'pencil')
        except Exception:
            pass
        self._set_status("Eraser enabled" if is_erase else "Eraser disabled")

    def on_recognize(self):
        # Run in a worker thread to keep UI responsive
        threading.Thread(target=self._recognize_worker, daemon=True).start()

    def _recognize_worker(self):
        try:
            self._set_status("Processing...")
            show_pre = self.var_show_pre.get()
            # Always get boxes for downstream heuristics; only show when requested
            imgs, boxes, th, gray = segment_symbols_with_debug(self.pil_image)
            segments = imgs
            if show_pre:
                self._update_previews(gray, th, boxes)

            if not segments:
                self._set_status("No symbols detected")
                return

            expr, labels, confs, _ = predict_symbols_detailed(segments, boxes=boxes)
            pretty, result = safe_eval(expr)
            self._write_results(pretty, labels, confs, result)
            self._set_status(f"Recognition complete: {pretty}")
        except FileNotFoundError as e:
            self._set_status("Model not found")
            messagebox.showerror("Model Missing", f"{e}\n\nPlease train the model first (see model.py).")
        except Exception as e:
            self._set_status("Error")
            messagebox.showerror("Error", str(e))

    def _set_status(self, text: str):
        self.status_var.set(text)

    def _write_results(self, expr: str, labels: List[str], confs: List[float], result: str):
        self.result_text.delete("1.0", tk.END)
        # If expression is empty or operator-only, make that explicit
        pretty_expr = expr if expr.strip() else "<no valid expression>"
        self.result_text.insert(tk.END, f"RECOGNIZED EXPRESSION: {pretty_expr}\n")
        self.result_text.insert(tk.END, "="*60 + "\n\n")
        if labels:
            self.result_text.insert(tk.END, "Individual Symbol Predictions:\n")
            # Display-friendly mapping (UI only)
            def _disp(l: str) -> str:
                if l == "times":
                    return "x"
                if l == "forward_slash" or l == "div":
                    return "/"
                return l
            for i, lab in enumerate(labels, 1):
                conf = f" ({confs[i-1]:.3f})" if i-1 < len(confs) else ""
                self.result_text.insert(tk.END, f"Symbol {i}: {_disp(lab)}{conf}\n")
            self.result_text.insert(tk.END, "\n")
        self.result_text.insert(tk.END, f"Result: {result}\n")

    def _update_previews(self, gray, th, boxes: List[Tuple[int,int,int,int]]):
        # Build preview images with segmentation boxes
        import numpy as np
        import cv2

        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        overlay = rgb.copy()
        for (x, y, w, h) in boxes:
            cv2.rectangle(overlay, (x, y), (x+w, y+h), (0, 200, 0), 2)

        vis1 = Image.fromarray(overlay)
        # Binary view
        th_rgb = Image.fromarray(cv2.cvtColor(th, cv2.COLOR_GRAY2RGB))

        def _thumb(im: Image.Image, max_w=280, max_h=180):
            im = im.copy()
            im.thumbnail((max_w, max_h))
            return ImageTk.PhotoImage(im)

        # Keep references to avoid GC
        self._img_preview1 = _thumb(vis1)
        self.preview_img_label.configure(image=self._img_preview1, text="")

    def on_upload_image(self):
        path = filedialog.askopenfilename(title="Select image", filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.bmp"), ("All files", "*.*")])
        if not path:
            return
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            messagebox.showerror("Open Image", f"Failed to open image: {e}")
            return

        # Fit into canvas and offscreen buffer
        self.canvas.delete("all")
        img_fit = img.copy()
        img_fit.thumbnail((self.canvas_width, self.canvas_height))
        self.pil_image.paste(self.bg_color, (0, 0, self.canvas_width, self.canvas_height))
        self.pil_image.paste(img_fit, (0, 0))
        self.tk_canvas_image = ImageTk.PhotoImage(img_fit)
        self.canvas.create_image(0, 0, image=self.tk_canvas_image, anchor=tk.NW)


def run():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    # Allow running this file directly (python gui.py)
    run()
