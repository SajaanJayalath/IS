"""
GUI Application for Handwritten Character Recognition System
Provides user interface for drawing, uploading, and recognizing handwritten characters
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
import json
from PIL import Image, ImageTk, ImageDraw
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import sys
from typing import Any, Dict

# Add src directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import CNNModel, SVMModel, RandomForestModel
from image_preprocessing import ImagePreprocessor, preprocess_for_mnist_model, prepare_for_character_model
from image_segmentation import ImageSegmenter, MultiDigitProcessor
from letters_new import LettersPipeline

class DrawingCanvas:
    """Canvas for drawing handwritten characters"""
    
    def __init__(self, parent, width=280, height=280):
        self.parent = parent
        self.width = width
        self.height = height
        
        # Create canvas
        self.canvas = tk.Canvas(parent, width=width, height=height, bg='white', cursor='pencil')
        self.canvas.pack(pady=10)
        
        # Create PIL image for drawing
        self.image = Image.new('RGB', (width, height), 'white')
        self.draw = ImageDraw.Draw(self.image)
        
        # Bind mouse events
        self.canvas.bind('<Button-1>', self.start_draw)
        self.canvas.bind('<B1-Motion>', self.draw_line)
        self.canvas.bind('<ButtonRelease-1>', self.end_draw)
        
        self.last_x = None
        self.last_y = None
        
    def start_draw(self, event):
        """Start drawing"""
        self.last_x = event.x
        self.last_y = event.y
        
    def draw_line(self, event):
        """Draw line on canvas"""
        if self.last_x and self.last_y:
            # Draw on tkinter canvas
            self.canvas.create_line(self.last_x, self.last_y, event.x, event.y, 
                                  width=8, fill='black', capstyle=tk.ROUND, smooth=tk.TRUE)
            
            # Draw on PIL image
            self.draw.line([self.last_x, self.last_y, event.x, event.y], fill='black', width=8)
            
        self.last_x = event.x
        self.last_y = event.y
        
    def end_draw(self, event):
        """End drawing"""
        self.last_x = None
        self.last_y = None
        
    def clear_canvas(self):
        """Clear the drawing canvas"""
        self.canvas.delete('all')
        self.image = Image.new('RGB', (self.width, self.height), 'white')
        self.draw = ImageDraw.Draw(self.image)
        
    def get_image_array(self):
        """Get the drawn image as numpy array"""
        # Convert PIL image to numpy array
        img_array = np.array(self.image)
        # Convert to grayscale
        gray_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        return gray_array

class HNRSApplication:
    """Main GUI Application for Handwritten Character Recognition System"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Handwritten Character Recognition System (HNRS)")
        self.root.geometry("1200x800")
        
        # Initialize components
        self.preprocessor = ImagePreprocessor()
        self.segmenter = ImageSegmenter()
        self.multi_digit_processor = MultiDigitProcessor()
        self.letter_pipeline = LettersPipeline()
        self.models = {}
        self.last_auto_seg_method: str | None = None

        self.recognition_modes: Dict[str, Dict[str, Any]] = {
            "digits": {
                "label": "Digits (0-9)",
                "dataset": "mnist_csv",
                "allowed_type": "digits",
            },
            "letters": {
                "label": "Letters (A-Z, a-z)",
                "dataset": "letters_new",
                "allowed_type": "letters",
            },
        }
        self.model_cache: Dict[str, Dict[str, Any]] = {}
        self.active_mode: str = "digits"
        self.active_dataset: str = self.recognition_modes[self.active_mode]["dataset"]
        self.active_label_mapping: Dict[int, str] = {}

        # Current image
        self.current_image = None
        self.processed_image = None

        # Load models
        self.load_models(self.active_mode)
        
        # Create GUI
        self.create_widgets()
        
    def load_models(self, mode: str | None = None, force_reload: bool = False):
        """Load trained models for the requested recognition mode."""
        mode_key = mode or self.active_mode
        if mode_key not in self.recognition_modes:
            raise ValueError(f"Unknown recognition mode: {mode_key}")
        config = self.recognition_modes[mode_key]
        dataset = config["dataset"]

        try:
            src_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(src_dir)
            primary_models_dir = os.path.join(project_root, "models")
            fallback_models_dir = os.path.join(src_dir, "models")

            def find_model_path(filename: str) -> str | None:
                primary_path = os.path.join(primary_models_dir, filename)
                fallback_path = os.path.join(fallback_models_dir, filename)
                if os.path.exists(primary_path):
                    return primary_path
                if os.path.exists(fallback_path):
                    return fallback_path
                return None

            if not force_reload and mode_key in self.model_cache:
                cached = self.model_cache[mode_key]
                self.models = dict(cached["models"])
                label_mapping = dict(cached["label_mapping"])
                self.active_mode = mode_key
                self.active_dataset = dataset
                self.active_label_mapping = label_mapping
                allowed_chars = self._allowed_characters_for_mode(mode_key, label_mapping)
                self._configure_multi_digit_models(label_mapping, allowed_chars, config.get("allowed_type"))
                self._configure_letter_pipeline(mode_key, allowed_chars)
                self._refresh_model_selector()
                return

            def candidate_names(basename: str) -> list[str]:
                candidates: list[str] = []
                if dataset:
                    candidates.append(basename.replace('.', f'_{dataset}.'))
                if dataset == "nist_by_class":
                    candidates.append(basename.replace('.', "_combined."))
                candidates.append(basename)
                seen: set[str] = set()
                ordered: list[str] = []
                for name in candidates:
                    if name not in seen:
                        ordered.append(name)
                        seen.add(name)
                return ordered

            loaded_models: Dict[str, Any] = {}
            model_specs = [] if dataset == "letters_new" else [
                ("CNN", CNNModel, "cnn_model.h5"),
                ("SVM", SVMModel, "svm_model.pkl"),
                ("Random Forest", RandomForestModel, "rf_model.pkl"),
            ]

            for display_name, cls, basename in model_specs:
                model_path = None
                for candidate in candidate_names(basename):
                    model_path = find_model_path(candidate)
                    if model_path:
                        break
                if not model_path:
                    print(f"Warning: {display_name} weights not found for dataset {dataset!r}")
                    continue
                try:
                    model = cls()
                    model.load_model(model_path)
                except Exception as exc:
                    print(f"Warning: failed to load {display_name} model from {model_path}: {exc}")
                    continue
                loaded_models[display_name] = model
                print(f"Loaded {display_name} model from {model_path}")

            self.models = loaded_models
            label_mapping = {}
            self.active_mode = mode_key
            self.active_dataset = dataset
            self.active_label_mapping = label_mapping
            self.model_cache[mode_key] = {
                "models": dict(loaded_models),
                "label_mapping": dict(label_mapping),
            }

            allowed_chars = self._allowed_characters_for_mode(mode_key, label_mapping)
            self._configure_multi_digit_models(label_mapping, allowed_chars, config.get("allowed_type"))
            self._configure_letter_pipeline(mode_key, allowed_chars)

            self._refresh_model_selector()
            if hasattr(self, "status_var"):
                mode_label = config.get("label", dataset)
                self.update_status(f"Loaded {mode_label} models")

        except Exception as e:
            messagebox.showerror("Model Loading Error", f"Error loading models: {e}")
            print(f"Error loading models: {e}")
    def _load_label_mapping_for_dataset(self, dataset: str, search_dirs: list[str]) -> Dict[int, str]:
        """Load a label mapping for the requested dataset with graceful fallbacks."""
        candidates: list[str] = []
        for base_dir in search_dirs:
            if not base_dir:
                continue
            dataset_file = os.path.join(base_dir, f"label_mapping_{dataset}.json")
            if dataset_file not in candidates:
                candidates.append(dataset_file)
        for base_dir in search_dirs:
            if not base_dir:
                continue
            generic_file = os.path.join(base_dir, "label_mapping.json")
            if generic_file not in candidates:
                candidates.append(generic_file)
        for path_candidate in candidates:
            if path_candidate and os.path.exists(path_candidate):
                try:
                    with open(path_candidate, "r", encoding="utf-8") as handle:
                        data = json.load(handle)
                    mapping = {int(k): str(v) for k, v in data.items()} if isinstance(data, dict) else {}
                    if mapping:
                        print(f"Loaded label mapping from {path_candidate}")
                        return mapping
                except Exception as exc:
                    print(f"Warning: failed to load label mapping {path_candidate}: {exc}")
        if dataset == "mnist_csv":
            return {i: str(i) for i in range(10)}
        fallback_chars = self._alphanumeric_fallback_chars()
        return {idx: char for idx, char in enumerate(fallback_chars)}

    def _alphanumeric_fallback_chars(self) -> list[str]:
        digits = [str(i) for i in range(10)]
        uppercase = [chr(c) for c in range(ord("A"), ord("Z") + 1)]
        lowercase = [chr(c) for c in range(ord("a"), ord("z") + 1)]
        return digits + uppercase + lowercase

    def _allowed_characters_for_mode(self, mode_key: str, label_mapping: Dict[int, str]) -> set[str] | None:
        config = self.recognition_modes.get(mode_key, {})
        allowed_type = config.get("allowed_type")
        base_chars: list[str]
        if allowed_type == "digits":
            base_chars = [str(i) for i in range(10)]
        elif allowed_type == "letters":
            uppercase = [chr(c) for c in range(ord("A"), ord("Z") + 1)]
            lowercase = [chr(c) for c in range(ord("a"), ord("z") + 1)]
            base_chars = uppercase + lowercase
        else:
            return None
        mapping_values = set(label_mapping.values())
        filtered = {char for char in base_chars if char in mapping_values}
        return filtered or None

    def _configure_multi_digit_models(self, label_mapping: Dict[int, str], allowed_chars: set[str] | None = None, profile: str | None = None) -> None:
        if label_mapping:
            self.multi_digit_processor.index_to_char = dict(label_mapping)
        else:
            self.multi_digit_processor.index_to_char = {}
        if not self.multi_digit_processor.index_to_char:
            if self.active_dataset == "mnist_csv":
                self.multi_digit_processor.index_to_char = {i: str(i) for i in range(10)}
            else:
                fallback_chars = self._alphanumeric_fallback_chars()
                self.multi_digit_processor.index_to_char = {idx: char for idx, char in enumerate(fallback_chars)}
        profile_key = profile or self.recognition_modes.get(self.active_mode, {}).get("allowed_type", "digits")
        self.multi_digit_processor.set_processing_profile(profile_key)
        self.multi_digit_processor.set_allowed_characters(allowed_chars)
        available_models = {
            "cnn": self.models.get("CNN"),
            "svm": self.models.get("SVM"),
            "rf": self.models.get("Random Forest"),
        }
        base_models = {name: model for name, model in available_models.items() if model is not None}
        self.multi_digit_processor.models = dict(base_models)
        if len(base_models) >= 2:
            self.multi_digit_processor.models["ensemble"] = dict(base_models)
        else:
            self.multi_digit_processor.models.pop("ensemble", None)

    def _configure_letter_pipeline(self, mode_key: str, allowed_chars: set[str] | None) -> None:
        if mode_key != "letters":
            return
        if not self.models or not self.active_label_mapping:
            return
        # new pipeline loads and manages its own model; nothing to configure
        return

    def _run_letter_recognition(
        self,
        model_display: str,
        segmentation_method: str,
        selection_raw: str,
    ) -> tuple[str, list[tuple[str, float]], list[np.ndarray], str]:
        try:
            text, predictions, processed_imgs, method_used = self.letter_pipeline.process_image(
                self.current_image, model_display, segmentation_method
            )
        except Exception as exc:
            messagebox.showerror("Letter Recognition Error", f"Error processing letter input: {exc}")
            raise

        method_display = method_used or selection_raw

        if method_used.lower().startswith("auto"):
            auto_method = method_used.split("->", 1)[-1].strip()
            self.last_auto_seg_method = auto_method if auto_method else None
        else:
            self.last_auto_seg_method = segmentation_method

        digit_images: list[np.ndarray] = []
        for img in processed_imgs:
            arr = np.asarray(img)
            if arr.ndim == 2:
                base = arr
            else:
                base = arr.reshape(arr.shape[0], arr.shape[1])
            if base.max() <= 1.0:
                base = (base * 255).astype(np.uint8)
            else:
                base = base.astype(np.uint8)
            digit_images.append(base)

        if not digit_images and text:
            prepared = prepare_for_character_model(self.current_image)
            base = prepared.squeeze()
            if base.max() <= 1.0:
                base = (base * 255).astype(np.uint8)
            digit_images = [base]

        return text, predictions, digit_images, method_display
    def _refresh_model_selector(self) -> None:
        if not hasattr(self, "model_var") or not hasattr(self, "model_combo"):
            return
        model_names = list(self.models.keys())
        self.model_combo["values"] = model_names
        if model_names:
            if self.model_var.get() not in model_names:
                self.model_var.set(model_names[0])
        else:
            self.model_var.set("")

    def on_recognition_mode_change(self) -> None:
        selected_mode = self.recognition_var.get()
        if selected_mode == self.active_mode:
            return
        previous_mode = self.active_mode
        self.load_models(selected_mode)
        if self.active_mode != selected_mode:
            self.recognition_var.set(previous_mode)


    def create_widgets(self):
        """Create and arrange GUI widgets"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="Handwritten Character Recognition System", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Left panel - Input methods
        input_frame = ttk.LabelFrame(main_frame, text="Input Methods", padding="10")
        input_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # Drawing canvas
        canvas_label = ttk.Label(input_frame, text="Draw characters:")
        canvas_label.pack(pady=(0, 5))
        
        self.drawing_canvas = DrawingCanvas(input_frame)
        
        # Canvas controls
        canvas_controls = ttk.Frame(input_frame)
        canvas_controls.pack(pady=10)
        
        ttk.Button(canvas_controls, text="Clear Canvas", 
                  command=self.clear_canvas).pack(side=tk.LEFT, padx=5)
        ttk.Button(canvas_controls, text="Recognize Drawing", 
                  command=self.recognize_drawing).pack(side=tk.LEFT, padx=5)
        
        # File upload
        ttk.Separator(input_frame, orient='horizontal').pack(fill=tk.X, pady=20)
        
        file_label = ttk.Label(input_frame, text="Or upload an image:")
        file_label.pack(pady=(0, 5))
        
        ttk.Button(input_frame, text="Upload Image File", 
                  command=self.upload_image).pack(pady=5)
        
        # Middle panel - Settings and controls
        control_frame = ttk.LabelFrame(main_frame, text="Settings & Controls", padding="10")
        control_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        
        # Recognition target selection
        target_label = ttk.Label(control_frame, text="Recognition Target:")
        target_label.pack(pady=(0, 5))

        self.recognition_var = tk.StringVar(value=self.active_mode)
        for mode_key in ("digits", "letters"):
            if mode_key not in self.recognition_modes:
                continue
            mode_info = self.recognition_modes[mode_key]
            ttk.Radiobutton(
                control_frame,
                text=mode_info["label"],
                value=mode_key,
                variable=self.recognition_var,
                command=self.on_recognition_mode_change,
            ).pack(anchor=tk.W, pady=2)

        ttk.Separator(control_frame, orient='horizontal').pack(fill=tk.X, pady=10)

        # Model selection
        model_label = ttk.Label(control_frame, text="Select Model:")
        model_label.pack(pady=(0, 5))

        loaded_model_names = list(self.models.keys())
        default_model = loaded_model_names[0] if loaded_model_names else ""
        self.model_var = tk.StringVar(value=default_model)
        self.model_combo = ttk.Combobox(
            control_frame,
            textvariable=self.model_var,
            values=loaded_model_names,
            state="readonly"
        )
        self.model_combo.pack(pady=(0, 10))

        # Segmentation method
        seg_label = ttk.Label(control_frame, text="Segmentation Method:")
        seg_label.pack(pady=(0, 5))
        
        self.segmentation_var = tk.StringVar(value="Auto selection")
        seg_combo = ttk.Combobox(control_frame, textvariable=self.segmentation_var,
                                values=["Auto selection", "contours", "connected_components", "projection"], state="readonly")
        seg_combo.pack(pady=(0, 10))
        
        # Processing options
        ttk.Separator(control_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        
        self.show_preprocessing = tk.BooleanVar(value=True)
        ttk.Checkbutton(control_frame, text="Show preprocessing steps", 
                       variable=self.show_preprocessing).pack(pady=2)
        
        self.show_segmentation = tk.BooleanVar(value=True)
        ttk.Checkbutton(control_frame, text="Show segmentation process", 
                       variable=self.show_segmentation).pack(pady=2)
        
        # Action buttons
        ttk.Separator(control_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        
        ttk.Button(control_frame, text="Process Current Image", 
                  command=self.process_current_image).pack(pady=5, fill=tk.X)
        
        ttk.Button(control_frame, text="Compare All Models", 
                  command=self.compare_all_models).pack(pady=5, fill=tk.X)

        ttk.Button(control_frame, text="Show Model Performance", 
                  command=self.show_model_performance).pack(pady=5, fill=tk.X)

        ttk.Button(control_frame, text="Compare Segmentation Methods",
                  command=self.compare_segmentation_methods).pack(pady=5, fill=tk.X)
        
        # Right panel - Results
        results_frame = ttk.LabelFrame(main_frame, text="Recognition Results", padding="10")
        results_frame.grid(row=1, column=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(10, 0))
        
        # Results display
        self.results_text = tk.Text(results_frame, width=40, height=15, wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=scrollbar.set)
        
        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Image preview frame
        preview_frame = ttk.LabelFrame(main_frame, text="Image Preview", padding="10")
        preview_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))
        
        self.image_preview = ttk.Label(preview_frame, text="No image loaded")
        self.image_preview.pack()
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))
        
    def clear_canvas(self):
        """Clear the drawing canvas"""
        self.drawing_canvas.clear_canvas()
        self.current_image = None
        self.update_status("Canvas cleared")
        
    def upload_image(self):
        """Upload image file"""
        file_path = filedialog.askopenfilename(
            title="Select Image File",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff *.tif"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                # Load image
                self.current_image = cv2.imread(file_path)
                if self.current_image is None:
                    raise ValueError("Could not load image")
                
                # Update preview
                self.update_image_preview(self.current_image)
                self.update_status(f"Image loaded: {os.path.basename(file_path)}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Could not load image: {e}")
                
    def recognize_drawing(self):
        """Recognize text drawn on canvas"""
        try:
            # Get image from canvas
            canvas_image = self.drawing_canvas.get_image_array()
            self.current_image = canvas_image
            
            # Update preview
            self.update_image_preview(canvas_image)
            
            # Process the image
            self.process_current_image()
            
        except Exception as e:
            messagebox.showerror("Error", f"Error recognizing drawing: {e}")
            
    def process_current_image(self):
        """Process the current image with selected settings"""
        if self.current_image is None:
            messagebox.showwarning("Warning", "No image to process")
            return
            
        try:
            self.update_status("Processing image...")
            
            model_display = self.model_var.get()
            if not model_display:
                messagebox.showerror("Model Not Selected", "Please select a model before processing.")
                self.update_status("Model not selected")
                return

            selection_raw = self.segmentation_var.get()
            segmentation_method = selection_raw.lower().replace(' ', '_')

            if self.active_mode == "letters":
                text, preds, imgs = self.letter_pipeline.recognize(self.current_image,
                                                                   'cc' if segmentation_method == 'auto_selection' else segmentation_method)
                number_string, predictions, digit_images = text, preds, imgs
                used_method_display = selection_raw
            else:
                model_name = model_display.lower().replace(' ', '_')
                if model_name == 'random_forest':
                    model_name = 'rf'

                # Ensure the selected model is actually loaded
                if model_name not in self.multi_digit_processor.models or \
                   self.multi_digit_processor.models.get(model_name) is None:
                    messagebox.showerror(
                        "Model Not Loaded",
                        "The selected model is not loaded. Please train or load models first."
                    )
                    self.update_status("Model not loaded")
                    return

                # Process multi-digit number (with optional auto selection)
                used_method_display = selection_raw
                if segmentation_method == 'auto_selection':
                    number_string, predictions, digit_images, used_method, _ = self.multi_digit_processor.auto_select_segmentation(
                        self.current_image, model_name
                    )
                    used_method_display = f"Auto -> {used_method}" if used_method else "Auto -> none"
                    self.last_auto_seg_method = used_method
                else:
                    number_string, predictions, digit_images = self.multi_digit_processor.process_multi_digit_number(
                        self.current_image, model_name, segmentation_method
                    )
                    used_method = segmentation_method
                    self.last_auto_seg_method = segmentation_method
            
            # Display results
            self.display_results(number_string, predictions, digit_images, used_method_display)
            
            # Show visualizations if requested
            if self.show_preprocessing.get():
                self.show_preprocessing_steps()
                
            if self.show_segmentation.get():
                self.show_segmentation_visualization()
            
            self.update_status(f"Recognition complete: {number_string}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error processing image: {e}")
            self.update_status("Error during processing")
            
    def display_results(self, number_string, predictions, digit_images, segmentation_used: str | None = None):
        """Display recognition results in the text widget"""
        self.results_text.delete(1.0, tk.END)
        
        # Main result
        self.results_text.insert(tk.END, f"RECOGNIZED TEXT: {number_string}\n")
        self.results_text.insert(tk.END, "="*40 + "\n\n")
        
        # Individual digit results
        self.results_text.insert(tk.END, "Individual Character Predictions:\n")
        self.results_text.insert(tk.END, "-"*30 + "\n")
        
        for i, (digit, confidence) in enumerate(predictions):
            self.results_text.insert(tk.END, f"Character {i+1}: {digit} (Confidence: {confidence:.3f})\n")
        
        self.results_text.insert(tk.END, "\n")
        
        # Model information
        model_name = self.model_var.get()
        self.results_text.insert(tk.END, f"Model Used: {model_name}\n")
        mode_label = self.recognition_modes.get(self.active_mode, {}).get("label", self.active_mode)
        self.results_text.insert(tk.END, f"Recognition Target: {mode_label}\n")
        seg_disp = segmentation_used if segmentation_used else self.segmentation_var.get()
        self.results_text.insert(tk.END, f"Segmentation: {seg_disp}\n")
        self.results_text.insert(tk.END, f"Character Count: {len(predictions)}\n")
        
        # Average confidence
        if predictions:
            avg_confidence = np.mean([conf for _, conf in predictions])
            self.results_text.insert(tk.END, f"Average Confidence: {avg_confidence:.3f}\n")
            
    def compare_all_models(self):
        """Compare recognition results across all models"""
        if self.current_image is None:
            messagebox.showwarning("Warning", "No image to process")
            return
            
        try:
            self.update_status("Comparing all models...")
            
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, "MODEL COMPARISON RESULTS\n")
            self.results_text.insert(tk.END, "="*50 + "\n\n")
            mode_label = self.recognition_modes.get(self.active_mode, {}).get("label", self.active_mode)
            self.results_text.insert(tk.END, f"Recognition Target: {mode_label}\n\n")
            
            selection_raw = self.segmentation_var.get()
            segmentation_method = selection_raw.lower().replace(' ', '_')

            if self.active_mode == "letters":
                for model_display_name, model_obj in self.models.items():
                    if model_obj is None:
                        self.results_text.insert(tk.END, f"{model_display_name} Model: Not loaded\n\n")
                        continue
                    try:
                        number_string, predictions, _, seg_used = self.letter_pipeline.process_image(
                            self.current_image, model_display_name, segmentation_method
                        )
                        avg_confidence = np.mean([conf for _, conf in predictions]) if predictions else 0
                        self.results_text.insert(tk.END, f"{model_display_name} Model:\n")
                        self.results_text.insert(tk.END, f"  Result: {number_string}\n")
                        self.results_text.insert(tk.END, f"  Avg Confidence: {avg_confidence:.3f}\n")
                        self.results_text.insert(tk.END, f"  Segmentation: {seg_used}\n")
                        self.results_text.insert(tk.END, f"  Individual: {[pred[0] for pred in predictions]}\n\n")
                    except Exception as e:
                        self.results_text.insert(tk.END, f"{model_display_name} Model: Error - {e}\n\n")
                return

            # Digit comparison (original pipeline)
            for model_display_name, model_key in [('CNN', 'cnn'), ('SVM', 'svm'), ('Random Forest', 'rf')]:
                if model_key in self.multi_digit_processor.models and self.multi_digit_processor.models[model_key]:
                    try:
                        if segmentation_method == 'auto_selection':
                            number_string, predictions, _, used_method, avg = self.multi_digit_processor.auto_select_segmentation(
                                self.current_image, model_key
                            )
                            seg_used = f"Auto -> {used_method} ({avg:.3f})"
                        else:
                            number_string, predictions, _ = self.multi_digit_processor.process_multi_digit_number(
                                self.current_image, model_key, segmentation_method
                            )
                            seg_used = selection_raw
                        
                        avg_confidence = np.mean([conf for _, conf in predictions]) if predictions else 0
                        
                        self.results_text.insert(tk.END, f"{model_display_name} Model:\n")
                        self.results_text.insert(tk.END, f"  Result: {number_string}\n")
                        self.results_text.insert(tk.END, f"  Avg Confidence: {avg_confidence:.3f}\n")
                        self.results_text.insert(tk.END, f"  Segmentation: {seg_used}\n")
                        self.results_text.insert(tk.END, f"  Individual: {[pred[0] for pred in predictions]}\n\n")
                        
                    except Exception as e:
                        self.results_text.insert(tk.END, f"{model_display_name} Model: Error - {e}\n\n")
                else:
                    self.results_text.insert(tk.END, f"{model_display_name} Model: Not loaded\n\n")

            self.update_status("Model comparison complete")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error comparing models: {e}")

    def compare_segmentation_methods(self):
        """Run recognition using all segmentation methods and report outputs."""
        if self.current_image is None:
            messagebox.showwarning("Warning", "No image to process")
            return

        try:
            # Determine selected model key
            model_name = self.model_var.get().lower().replace(' ', '_')
            if model_name == 'random_forest':
                model_name = 'rf'

            if self.active_mode == "letters":
                methods = ["contours", "connected_components", "projection"]
                self.results_text.delete(1.0, tk.END)
                self.results_text.insert(tk.END, "SEGMENTATION METHOD COMPARISON\n")
                self.results_text.insert(tk.END, "="*50 + "\n\n")

                for m in methods:
                    try:
                        number_string, predictions, _, _ = self.letter_pipeline.process_image(
                            self.current_image, self.model_var.get(), m
                        )
                        avg_conf = np.mean([c for _, c in predictions]) if predictions else 0
                        chars = [p[0] for p in predictions]
                        self.results_text.insert(tk.END, f"{m}: {number_string}  | characters: {chars}  | avg: {avg_conf:.3f}\n")
                    except Exception as e:
                        self.results_text.insert(tk.END, f"{m}: Error - {e}\n")

                self.update_status("Segmentation methods compared")
                return

            if model_name not in self.multi_digit_processor.models or \
               self.multi_digit_processor.models.get(model_name) is None:
                messagebox.showerror("Model Not Loaded", "The selected model is not loaded.")
                return

            methods = ["contours", "connected_components", "projection"]
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, "SEGMENTATION METHOD COMPARISON\n")
            self.results_text.insert(tk.END, "="*50 + "\n\n")

            for m in methods:
                try:
                    number_string, predictions, _ = self.multi_digit_processor.process_multi_digit_number(
                        self.current_image, model_name, m
                    )
                    avg_conf = np.mean([c for _, c in predictions]) if predictions else 0
                    self.results_text.insert(tk.END, f"{m}: {number_string}  | characters: {[p[0] for p in predictions]}  | avg: {avg_conf:.3f}\n")
                except Exception as e:
                    self.results_text.insert(tk.END, f"{m}: Error - {e}\n")

            self.update_status("Segmentation methods compared")
        except Exception as e:
            messagebox.showerror("Error", f"Error comparing segmentation methods: {e}")
            
    def show_model_performance(self):
        """Show model performance metrics"""
        performance_window = tk.Toplevel(self.root)
        performance_window.title("Model Performance Metrics")
        performance_window.geometry("600x400")
        
        # Create text widget for performance data
        perf_text = tk.Text(performance_window, wrap=tk.WORD, padx=10, pady=10)
        perf_scrollbar = ttk.Scrollbar(performance_window, orient=tk.VERTICAL, command=perf_text.yview)
        perf_text.configure(yscrollcommand=perf_scrollbar.set)
        
        perf_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        perf_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Display performance metrics
        perf_text.insert(tk.END, "MODEL PERFORMANCE METRICS\n")
        perf_text.insert(tk.END, "="*50 + "\n\n")
        
        perf_text.insert(tk.END, "Training Results on MNIST Dataset:\n")
        perf_text.insert(tk.END, "-"*40 + "\n")
        perf_text.insert(tk.END, "CNN Model: 99.23% accuracy\n")
        perf_text.insert(tk.END, "  - Training time: 218.51 seconds\n")
        perf_text.insert(tk.END, "  - Architecture: 3 Conv blocks + Dense layers\n")
        perf_text.insert(tk.END, "  - Best for: Complex patterns, high accuracy\n\n")
        
        perf_text.insert(tk.END, "SVM Model: 96.34% accuracy\n")
        perf_text.insert(tk.END, "  - Training time: 43.49 seconds\n")
        perf_text.insert(tk.END, "  - Kernel: RBF\n")
        perf_text.insert(tk.END, "  - Best for: Fast inference, good generalization\n\n")
        
        perf_text.insert(tk.END, "Random Forest: 95.11% accuracy\n")
        perf_text.insert(tk.END, "  - Training time: 0.78 seconds\n")
        perf_text.insert(tk.END, "  - Estimators: 100\n")
        perf_text.insert(tk.END, "  - Best for: Very fast training and inference\n\n")
        
        perf_text.insert(tk.END, "Recommendations:\n")
        perf_text.insert(tk.END, "-"*20 + "\n")
        perf_text.insert(tk.END, "• Use CNN for highest accuracy\n")
        perf_text.insert(tk.END, "• Use SVM for balanced speed/accuracy\n")
        perf_text.insert(tk.END, "• Use Random Forest for fastest processing\n")
        
    def show_preprocessing_steps(self):
        """Show preprocessing visualization in new window"""
        if self.current_image is None:
            return
            
        try:
            # Create new window for preprocessing visualization
            prep_window = tk.Toplevel(self.root)
            prep_window.title("Preprocessing Steps")
            prep_window.geometry("800x600")
            
            # Create matplotlib figure
            fig, axes = plt.subplots(2, 3, figsize=(12, 8))
            axes = axes.ravel()
            
            # Show preprocessing steps
            steps = ['grayscale', 'clahe', 'threshold', 'morphology', 'deskew']
            processed = self.current_image.copy()
            
            # Original
            axes[0].imshow(self.current_image, cmap='gray')
            axes[0].set_title('Original')
            axes[0].axis('off')
            
            # Apply each step
            for i, step in enumerate(steps, 1):
                if step == 'grayscale':
                    processed = self.preprocessor.convert_to_grayscale(processed)
                elif step == 'clahe':
                    processed = self.preprocessor.adaptive_histogram_equalization(processed)
                elif step == 'threshold':
                    processed = self.preprocessor.adaptive_threshold(processed)
                elif step == 'morphology':
                    processed = self.preprocessor.morphological_operations(processed, 'opening')
                elif step == 'deskew':
                    processed = self.preprocessor.deskew_image(processed)
                
                axes[i].imshow(processed, cmap='gray')
                axes[i].set_title(f'After {step}')
                axes[i].axis('off')
            
            plt.tight_layout()
            
            # Embed in tkinter window
            canvas = FigureCanvasTkAgg(fig, prep_window)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
        except Exception as e:
            print(f"Error showing preprocessing steps: {e}")
            
    def show_segmentation_visualization(self):
        """Show segmentation visualization in new window"""
        if self.current_image is None:
            return
            
        try:
            # Create new window
            seg_window = tk.Toplevel(self.root)
            seg_window.title("Segmentation Visualization")
            seg_window.geometry("1000x600")
            
            # Get a Matplotlib figure and embed it in the Tk window
            sel = self.segmentation_var.get()
            method = sel.lower().replace(' ', '_')
            if method == 'auto_selection':
                method = self.last_auto_seg_method or 'contours'
            fig = self.segmenter.visualize_segmentation(
                self.current_image, method, return_fig=True
            )
            if fig is None:
                ttk.Label(seg_window, text="No characters found to visualize").pack(padx=20, pady=20)
                return

            canvas = FigureCanvasTkAgg(fig, master=seg_window)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
        except Exception as e:
            print(f"Error showing segmentation visualization: {e}")
            
    def update_image_preview(self, image):
        """Update the image preview"""
        try:
            # Convert to PIL Image for display
            if len(image.shape) == 3:
                # Color image
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(image_rgb)
            else:
                # Grayscale image
                pil_image = Image.fromarray(image)
            
            # Resize for preview (maintain aspect ratio)
            pil_image.thumbnail((300, 200), Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(pil_image)
            
            # Update preview label
            self.image_preview.configure(image=photo, text="")
            self.image_preview.image = photo  # Keep a reference
            
        except Exception as e:
            print(f"Error updating image preview: {e}")
            
    def update_status(self, message):
        """Update status bar"""
        self.status_var.set(message)
        self.root.update_idletasks()

def main():
    """Main function to run the GUI application"""
    root = tk.Tk()
    app = HNRSApplication(root)
    
    # Center window on screen
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (1200 // 2)
    y = (root.winfo_screenheight() // 2) - (800 // 2)
    root.geometry(f"1200x800+{x}+{y}")
    
    root.mainloop()

if __name__ == "__main__":
    main()
