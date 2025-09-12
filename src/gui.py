"""
GUI Application for Handwritten Number Recognition System
Provides user interface for drawing, uploading, and recognizing handwritten numbers
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk, ImageDraw
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import sys

# Add src directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import CNNModel, SVMModel, RandomForestModel
from image_preprocessing import ImagePreprocessor, preprocess_for_mnist_model
from image_segmentation import ImageSegmenter, MultiDigitProcessor

class DrawingCanvas:
    """Canvas for drawing handwritten digits"""
    
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
    """Main GUI Application for Handwritten Number Recognition System"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Handwritten Number Recognition System (HNRS)")
        self.root.geometry("1200x800")
        
        # Initialize components
        self.preprocessor = ImagePreprocessor()
        self.segmenter = ImageSegmenter()
        self.multi_digit_processor = MultiDigitProcessor()
        self.models = {}
        self.last_auto_seg_method: str | None = None
        
        # Current image
        self.current_image = None
        self.processed_image = None
        
        # Load models
        self.load_models()
        
        # Create GUI
        self.create_widgets()
        
    def load_models(self):
        """Load all trained models"""
        try:
            # Determine possible model directories
            src_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(src_dir)
            primary_models_dir = os.path.join(project_root, 'models')
            fallback_models_dir = os.path.join(src_dir, 'models')

            def find_model_path(filename):
                primary_path = os.path.join(primary_models_dir, filename)
                fallback_path = os.path.join(fallback_models_dir, filename)
                if os.path.exists(primary_path):
                    return primary_path
                if os.path.exists(fallback_path):
                    return fallback_path
                return None

            # Prefer models trained on the combined dataset if present
            def find_prefer_combined(basename: str) -> str | None:
                pref = [
                    basename.replace('.', '_combined.'),  # e.g., cnn_model_combined.h5
                    basename,                              # generic fallback
                ]
                for name in pref:
                    p = find_model_path(name)
                    if p:
                        return p
                return None

            # Load CNN model
            cnn_model = CNNModel()
            cnn_path = find_prefer_combined('cnn_model.h5')
            if cnn_path:
                cnn_model.load_model(cnn_path)
                self.models['CNN'] = cnn_model
                print("CNN model loaded successfully")
            
            # Load SVM model
            svm_model = SVMModel()
            svm_path = find_prefer_combined('svm_model.pkl')
            if svm_path:
                svm_model.load_model(svm_path)
                self.models['SVM'] = svm_model
                print("SVM model loaded successfully")
            
            # Load Random Forest model
            rf_model = RandomForestModel()
            rf_path = find_prefer_combined('rf_model.pkl')
            if rf_path:
                rf_model.load_model(rf_path)
                self.models['Random Forest'] = rf_model
                print("Random Forest model loaded successfully")
                
            # Set up multi-digit processor
            # Only include models that actually loaded (non-None)
            available_models = {
                'cnn': self.models.get('CNN'),
                'svm': self.models.get('SVM'),
                'rf': self.models.get('Random Forest')
            }
            # Filter out None entries to prevent NoneType.predict errors
            self.multi_digit_processor.models = {k: v for k, v in available_models.items() if v is not None}

        except Exception as e:
            messagebox.showerror("Model Loading Error", f"Error loading models: {e}")
            print(f"Error loading models: {e}")
    
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
        title_label = ttk.Label(main_frame, text="Handwritten Number Recognition System", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Left panel - Input methods
        input_frame = ttk.LabelFrame(main_frame, text="Input Methods", padding="10")
        input_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # Drawing canvas
        canvas_label = ttk.Label(input_frame, text="Draw a number:")
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
        
        # Model selection
        model_label = ttk.Label(control_frame, text="Select Model:")
        model_label.pack(pady=(0, 5))
        
        # Populate model dropdown with actually loaded models
        loaded_model_names = list(self.models.keys())
        # Default to first loaded model if available; otherwise keep placeholder
        default_model = loaded_model_names[0] if loaded_model_names else "CNN"
        self.model_var = tk.StringVar(value=default_model)
        model_combo = ttk.Combobox(control_frame, textvariable=self.model_var,
                                  values=loaded_model_names, state="readonly")
        model_combo.pack(pady=(0, 10))
        
        # Segmentation method
        seg_label = ttk.Label(control_frame, text="Segmentation Method:")
        seg_label.pack(pady=(0, 5))
        
        self.segmentation_var = tk.StringVar(value="contours")
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
        """Recognize number drawn on canvas"""
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
            
            # Get selected model and segmentation method
            model_name = self.model_var.get().lower().replace(' ', '_')
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

            selection_raw = self.segmentation_var.get()
            segmentation_method = selection_raw.lower().replace(' ', '_')

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
        self.results_text.insert(tk.END, f"RECOGNIZED NUMBER: {number_string}\n")
        self.results_text.insert(tk.END, "="*40 + "\n\n")
        
        # Individual digit results
        self.results_text.insert(tk.END, "Individual Digit Predictions:\n")
        self.results_text.insert(tk.END, "-"*30 + "\n")
        
        for i, (digit, confidence) in enumerate(predictions):
            self.results_text.insert(tk.END, f"Digit {i+1}: {digit} (Confidence: {confidence:.3f})\n")
        
        self.results_text.insert(tk.END, "\n")
        
        # Model information
        model_name = self.model_var.get()
        self.results_text.insert(tk.END, f"Model Used: {model_name}\n")
        seg_disp = segmentation_used if segmentation_used else self.segmentation_var.get()
        self.results_text.insert(tk.END, f"Segmentation: {seg_disp}\n")
        self.results_text.insert(tk.END, f"Number of Digits: {len(predictions)}\n")
        
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
            
            selection_raw = self.segmentation_var.get()
            segmentation_method = selection_raw.lower().replace(' ', '_')
            
            # Test each model
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
                    self.results_text.insert(tk.END, f"{m}: {number_string}  | digits: {[p[0] for p in predictions]}  | avg: {avg_conf:.3f}\n")
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
                ttk.Label(seg_window, text="No digits found to visualize").pack(padx=20, pady=20)
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
