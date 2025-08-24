#!/usr/bin/env python

import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np

import multiprocessing
import queue
import threading
from concurrent.futures import ProcessPoolExecutor
import sys
import os

# Keep math/TF threads polite in the GUI proc (optional but nice)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from run_flownet_fenicsx import (
    load_flownet_model,
    process_image_channels,           # (unused now; kept if you want it)
    create_2ch_test_data_from_img,
    convert_binary_to_color,
    run_flownet_cpu_preload_model,
    load_normalize_factor,
)

# --------------------------
# FEM worker function (short-lived per run)
# --------------------------
def _fem_only_job(img_gray_np, flowrate_ratio, u_max):
    # Kill MPICH async progress *before* any dolfinx/petsc imports
    os.environ["MPICH_ASYNC_PROGRESS"] = "0"
    os.environ["PETSC_MPICH_ASYNC_PROGRESS"] = "0"
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    import run_flownet_fenicsx as api  # heavy stuff imported only in worker
    inner_model, outer_model, inner_shape = api.image2gmshfromimg(img_gray_np)
    flow_profile, inner_shape = api.solve_inlet_profiles(
        inner_model, outer_model, inner_shape, flowrate_ratio, u_max
    )
    return flow_profile, inner_shape


# Global variables for image sizes
nx, ny = 300, 300


class ImageProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Processor")

        self.queue = queue.Queue()

        # Configure grid layout for four quadrants
        self.root.rowconfigure([0, 1], weight=1, minsize=ny)
        self.root.columnconfigure([0, 1], weight=1, minsize=nx)

        # Upper Left Quadrant: Controls with Title
        self.control_frame = tk.Frame(root, relief='sunken', bd=2)
        self.control_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        self.control_label = tk.Label(self.control_frame, text="Controls",
                                      font=("Arial", 12, 'bold'))
        self.control_label.grid(row=0, column=0, columnspan=4, pady=5, sticky="nsew")

        self.load_button = tk.Button(
            self.control_frame, text="Load Geometry Image", command=self.load_image
        )
        self.load_button.grid(row=1, column=0, columnspan=4, pady=10)

        # Ratio Input for flowrates
        self.flowrate_label = tk.Label(self.control_frame, text="Flowrate Ratio:")
        self.flowrate_label.grid(row=3, column=0, sticky="e", padx=5, pady=5)

        self.flowrate_entry_left = tk.Entry(self.control_frame, width=5)
        self.flowrate_entry_left.insert(0, "1")
        self.flowrate_entry_left.grid(row=3, column=1, sticky="e", pady=5)

        self.colon_label = tk.Label(self.control_frame, text=":")
        self.colon_label.grid(row=3, column=2, pady=5)

        self.flowrate_entry_right = tk.Entry(self.control_frame, width=5)
        self.flowrate_entry_right.insert(0, "1")
        self.flowrate_entry_right.grid(row=3, column=3, sticky="w", pady=5)

        # Message area
        self.message_frame = tk.Frame(self.control_frame)
        self.message_frame.grid(row=5, column=0, columnspan=4,
                                pady=10, padx=5, sticky="nsew")

        self.message_area = tk.Text(self.message_frame,
                                    height=15, width=40, state="disabled", wrap="word")
        self.message_area.pack(side="left", fill="both", expand=True)

        self.message_scrollbar = tk.Scrollbar(self.message_frame,
                                              command=self.message_area.yview)
        self.message_scrollbar.pack(side="right", fill="y")
        self.message_area.config(yscrollcommand=self.message_scrollbar.set)

        # Redirect stdout/stderr to the message box (note: can add buffering if you want)
        sys.stdout = self
        sys.stderr = self

        # Other Quadrants
        self._create_image_display_areas()

        # Attributes to hold images
        self.input_image = None
        self.output_image_1 = None
        self.output_image_2 = None

        self.root.after(100, self.process_queue)

        # Load FlowNet once in GUI process
        self.flownet_model = load_flownet_model()
        self.u_max = load_normalize_factor()

    def _create_image_display_areas(self):
        # Upper Right Quadrant: Loaded Image
        self.input_frame = tk.Frame(self.root, relief='sunken', bd=2)
        self.input_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)

        self.input_label = tk.Label(self.input_frame, text="Loaded Nozzle Geometry",
                                    font=("Arial", 12, 'bold'))
        self.input_label.pack(pady=5)

        self.input_canvas = tk.Canvas(self.input_frame, bg="gray",
                                      width=nx, height=ny, highlightthickness=0)
        self.input_canvas.pack()

        # Lower Left Quadrant: U-Net Image
        self.output_frame_1 = tk.Frame(self.root, relief='sunken', bd=2)
        self.output_frame_1.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)

        self.output_label_1 = tk.Label(self.output_frame_1, text="U-Net Prediction",
                                       font=("Arial", 12, 'bold'))
        self.output_label_1.pack(pady=5)

        self.process_button_1 = tk.Button(
            self.output_frame_1, text='Run U-Net',
            command=lambda: self.run_in_thread(1)
        )
        self.process_button_1.pack(pady=5)

        self.output_canvas_1 = tk.Canvas(self.output_frame_1, bg="gray",
                                         width=nx, height=ny, highlightthickness=0)
        self.output_canvas_1.pack()

        self.save_button_1 = tk.Button(
            self.output_frame_1, text="Save U-Net image", command=self.save_image_1
        )
        self.save_button_1.pack(pady=5)

        # Lower Right Quadrant: 3D Model Image (disabled placeholder)
        self.output_frame_2 = tk.Frame(self.root, relief='sunken', bd=2)
        self.output_frame_2.grid(row=1, column=1, sticky="nsew", padx=10, pady=10)

        self.output_label_2 = tk.Label(self.output_frame_2, text="CFD Prediction",
                                       font=("Arial", 12, 'bold'))
        self.output_label_2.pack(pady=5)

        self.process_button_2 = tk.Button(
            self.output_frame_2, text='Run CFD Model',
            command=lambda: self.run_in_thread(2),
            state='disabled'
        )
        self.process_button_2.pack(pady=5)

        self.output_canvas_2 = tk.Canvas(self.output_frame_2, bg="gray",
                                         width=nx, height=ny, highlightthickness=0)
        self.output_canvas_2.pack()

        self.save_button_2 = tk.Button(
            self.output_frame_2, text="Save CFD Model image", command=self.save_image_2
        )
        self.save_button_2.pack(pady=5)

    # --- logging redirection ---
    def write(self, message):
        self.message_area.config(state="normal")
        self.message_area.insert("end", message)
        self.message_area.config(state="disabled")
        self.message_area.see("end")

    def flush(self):
        pass

    # --- image helpers ---
    def make_square_by_mirroring(self, im, tol_ratio=0.02):
        w, h = im.size
        if abs(w - h) <= tol_ratio * max(w, h):
            return im
        if h > w:
            mirrored = im.transpose(Image.FLIP_LEFT_RIGHT)
            new_im = Image.new(im.mode, (w * 2, h))
            new_im.paste(im, (0, 0))
            new_im.paste(mirrored, (w, 0))
            return new_im
        else:
            mirrored = im.transpose(Image.FLIP_TOP_BOTTOM)
            new_im = Image.new(im.mode, (w, h * 2))
            new_im.paste(im, (0, 0))
            new_im.paste(mirrored, (0, h))
            return new_im

    def load_image(self):
        file_path = filedialog.askopenfilename(title="Select an Image File")
        if file_path:
            img = Image.open(file_path)
            img = self.make_square_by_mirroring(img, tol_ratio=0.02)
            self.input_image = img
            self.display_image(self.input_image, self.input_canvas)

    def display_image(self, image, canvas):
        resized_image = image.resize((nx, ny))
        canvas.delete("all")
        canvas.image = ImageTk.PhotoImage(resized_image)
        canvas.create_image(nx//2, ny//2, anchor=tk.CENTER, image=canvas.image)

    # --- threading entrypoints ---
    def run_in_thread(self, variant):
        thread = threading.Thread(target=self.process_image, args=(variant,), daemon=True)
        thread.start()

    def process_image(self, variant):
        if variant == 1:
            self.root.after(0, self.process_flownet)  # U-Net pipeline (FEM in worker)
        else:
            pass  # CFD button currently disabled

    # --- main pipeline: FEM in short-lived worker, TF in GUI ---
    def process_flownet(self, _variant_ignored=None):
        if not self.input_image:
            messagebox.showerror("Error", "No input image loaded!")
            return

        try:
            inner_flow = float(self.flowrate_entry_left.get())
            outer_flow = float(self.flowrate_entry_right.get())
            flowrate_ratio = inner_flow / (inner_flow + outer_flow)

            # Make a clean grayscale [0..1] array for the FEM worker
            img_gray_np = np.asarray(self.input_image.convert("L"), dtype=np.float32) / 255.0

            # ---- Run FEM in a short-lived worker (no idle CPU after) ----
            with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context('spawn')) as pool:
                fut = pool.submit(_fem_only_job, img_gray_np, flowrate_ratio, self.u_max)
                flow_profile, inner_shape = fut.result()

            # ---- TF in GUI (model already loaded) ----
            X = create_2ch_test_data_from_img(flow_profile, inner_shape, 256, 256)
            pred_mask = run_flownet_cpu_preload_model(self.flownet_model, X)
            rgb = convert_binary_to_color(pred_mask)
            out_img = Image.fromarray(rgb, mode="RGB")

            self.output_image_1 = out_img
            self.display_image(self.output_image_1, self.output_canvas_1)

        except Exception as e:
            self.queue.put(f'Error occured:\n{e}\n')

    # --- queue pump ---
    def process_queue(self):
        while not self.queue.empty():
            message = self.queue.get_nowait()
            self.write(message)
        self.root.after(100, self.process_queue)

    # --- save helpers ---
    def save_image_1(self):
        self.save_image(self.output_image_1)

    def save_image_2(self):
        self.save_image(self.output_image_2)

    def save_image(self, image):
        if not image:
            messagebox.showerror("Error", "No processed image to save!")
            return
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")],
        )
        try:
            if file_path:
                image.save(file_path)
        except ValueError as e:
            messagebox.showerror("Input Error", f"{e}")


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    root = tk.Tk()
    app = ImageProcessorApp(root)
    root.mainloop()

