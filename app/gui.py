#!/usr/bin/env python

import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageOps, ImageEnhance
import numpy as np

import multiprocessing
import queue
import threading

import sys
import os
# Suppress Tensorflow warnings
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from run_flownet_fenicsx import (run_job_preload_model_preload_img, 
                                 load_flownet_model,
                                 process_image_channels)

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

        '''
        # Number Inputs for Brightness
        self.brightness_label = tk.Label(self.control_frame, text="Brightness (1-3):")
        self.brightness_label.grid(row=2, column=0, sticky="e", padx=5, pady=5)
        self.brightness_entry = tk.Entry(self.control_frame, width=5)
        self.brightness_entry.insert(0, "1.0")  # Default value
        self.brightness_entry.grid(row=2, column=1, pady=5)
        '''

        # Ratio Input for flowrates
        self.flowrate_label = tk.Label(self.control_frame, text="Flowrate Ratio:")
        self.flowrate_label.grid(row=3, column=0, sticky="e", padx=5, pady=5)

        # Left text entry for ratio
        self.flowrate_entry_left = tk.Entry(self.control_frame, width=5)
        self.flowrate_entry_left.insert(0, "1")  # Default left value
        self.flowrate_entry_left.grid(row=3, column=1, sticky="e", pady=5)

        # Colon label
        self.colon_label = tk.Label(self.control_frame, text=":")
        #self.colon_label.grid(row=3, column=2, sticky="w", pady=5)
        self.colon_label.grid(row=3, column=2, pady=5)

        # Right text entry for ratio
        self.flowrate_entry_right = tk.Entry(self.control_frame, width=5)
        self.flowrate_entry_right.insert(0, "1")  # Default right value
        self.flowrate_entry_right.grid(row=3, column=3, sticky="w", pady=5)

        # Frame for Message Area with Scrollbar
        self.message_frame = tk.Frame(self.control_frame)
        self.message_frame.grid(row=5, column=0, columnspan=4, 
                                pady=10, padx=5, sticky="nsew")

        # Text Widget for Messages
        self.message_area = tk.Text(self.message_frame, 
                                    height=15, width=40, state="disabled", wrap="word")
        self.message_area.pack(side="left", fill="both", expand=True)

        # Scrollbar for Message Area
        self.message_scrollbar = tk.Scrollbar(self.message_frame, 
                                              command=self.message_area.yview)
        self.message_scrollbar.pack(side="right", fill="y")

        # Link scrollbar to text widget
        self.message_area.config(yscrollcommand=self.message_scrollbar.set)

        # Redirect standard output to message area
        sys.stdout = self
        sys.stderr = self

        # Other Quadrants: Image Display and Save Buttons (unchanged)
        self._create_image_display_areas()

        # Attributes to hold images
        self.input_image = None
        self.output_image_1 = None
        self.output_image_2 = None

        self.root.after(100, self.process_queue)

        self.flownet_model = load_flownet_model()

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
                command=lambda:self.run_in_thread(1)
                )
        self.process_button_1.pack(pady=5)


        self.output_canvas_1 = tk.Canvas(self.output_frame_1, bg="gray", 
                                         width=nx, height=ny, highlightthickness=0)
        self.output_canvas_1.pack()

        self.save_button_1 = tk.Button(
            self.output_frame_1, text="Save U-Net image", command=self.save_image_1
        )
        self.save_button_1.pack(pady=5)

        # Lower Right Quadrant: 3D Model Image
        self.output_frame_2 = tk.Frame(self.root, relief='sunken', bd=2)
        self.output_frame_2.grid(row=1, column=1, sticky="nsew", padx=10, pady=10)

        self.output_label_2 = tk.Label(self.output_frame_2, text="CFD Prediction", 
                                       font=("Arial", 12, 'bold'))
        self.output_label_2.pack(pady=5)

        self.process_button_2 = tk.Button(
                self.output_frame_2, text='Run CFD Model', command=lambda:
                self.run_in_thread(2),
                state='normal'
                )
        self.process_button_2.pack(pady=5)

        self.output_canvas_2 = tk.Canvas(self.output_frame_2, bg="gray", width=nx, height=ny, highlightthickness=0)
        self.output_canvas_2.pack()

        self.save_button_2 = tk.Button(
            self.output_frame_2, text="Save CFD Model image", command=self.save_image_2
        )
        self.save_button_2.pack(pady=5)

        # Mesh size input
        self.mesh_label = tk.Label(self.output_frame_2, text="Mesh Size:")
        self.mesh_label.pack()
        self.mesh_entry = tk.Entry(self.output_frame_2, width=10)
        self.mesh_entry.insert(0, "50")  # Default mesh size
        self.mesh_entry.pack()

        # Reynolds number input
        self.re_label = tk.Label(self.output_frame_2, text="Reynolds Number (1-10):")
        self.re_label.pack()
        self.re_entry = tk.Entry(self.output_frame_2, width=10)
        self.re_entry.insert(0, "1")  # Default Reynolds number
        self.re_entry.pack()

        # Streamtrace seeds input
        self.seeds_label = tk.Label(self.output_frame_2, text="Streamtrace Seeds (10-400):")
        self.seeds_label.pack()
        self.seeds_entry = tk.Entry(self.output_frame_2, width=10)
        self.seeds_entry.insert(0, "25")  # Default seeds
        self.seeds_entry.pack()


    def write(self, message):
        # Write to the message area
        self.message_area.config(state="normal")
        self.message_area.insert("end", message)
        self.message_area.config(state="disabled")
        self.message_area.see("end")

    def flush(self):
        pass

    def load_image(self):
        file_path = filedialog.askopenfilename(title="Select an Image File")
        if file_path:
            self.input_image = Image.open(file_path)
            self.display_image(self.input_image, self.input_canvas)

    def display_image(self, image, canvas):
        resized_image = image.resize((nx, ny))
        canvas.image = ImageTk.PhotoImage(resized_image)
        canvas.create_image(nx//2, ny//2, anchor=tk.CENTER, image=canvas.image)

    def process_image_1(self):
        self.process_image(1)

    def process_image_2(self):
        self.process_image(2)

    def run_in_thread(self, variant):
        thread = threading.Thread(target=self.process_image, args=(variant,))
        thread.start()

    def process_image(self, variant):
        if variant == 1:
            # Variant 1 is the DL model
            self.root.after(0, self.process_flownet, variant)
        else:
            # Variant 2 is the CFD model
            self.root.after(0, self.process_cfd_model, variant)
            pass

    def process_flownet(self, variant):
        if not self.input_image:
            messagebox.showerror("Error", "No input image loaded!")
            return

        try:
            # Get flowrate ratio
            inner_flow = float(self.flowrate_entry_left.get())
            outer_flow = float(self.flowrate_entry_right.get())
            flowrate_ratio = inner_flow / (inner_flow + outer_flow)  # Compute the ratio

            # Convert PIL image to grayscale numpy
            input_image_np = np.asarray(self.input_image)
            input_image_ch = process_image_channels(input_image_np)

            # Run flownet
            flownet_pred = run_job_preload_model_preload_img(self.flownet_model, 
                                                             input_image_ch,
                                                             flowrate_ratio,
                                                             'test_img.png')
            # Get brightness value
            '''
            brightness = float(self.brightness_entry.get())
            if brightness < 0.1 or brightness > 3.0:
                raise ValueError("Brightness out of range (1-3).")
            '''

            self.output_image_1 = flownet_pred
            # DL model is image 1 for canvas 1
            self.display_image(self.output_image_1, self.output_canvas_1)

        except ValueError as e:
            self.queue.put(f'Error occured:\n{e}\n')
            #messagebox.showerror("Input Error", f"{e}")

    def process_cfd_model(self, variant):
        if not self.input_image:
            messagebox.showerror("Error", "No input image loaded!")
            return

        try:
            mesh_size = int(self.mesh_entry.get())
            reynolds = float(self.re_entry.get())
            seeds = int(self.seeds_entry.get())

            if not (1 <= reynolds <= 10):
                raise ValueError("Reynolds number must be between 1 and 10.")
            if not (10 <= seeds <= 400):
                raise ValueError("Streamtrace seeds must be between 10 and 400.")

            # Send values to the display window
            self.queue.put(f"Running CFD Model...\n")
            self.queue.put(f"  Mesh Size: {mesh_size}\n")
            self.queue.put(f"  Reynolds number set to: {reynolds}\n")
            self.queue.put(f"  Streamtrace Seeds: {seeds}\n")

            # If you have a CFD function to call, you'd do it here:
            # result = run_cfd_model(self.input_image, mesh_size, reynolds, seeds)
            # self.output_image_2 = result
            # self.display_image(self.output_image_2, self.output_canvas_2)

        except ValueError as e:
            self.queue.put(f"Error in CFD input:\n{e}\n")

    def process_queue(self):
        while not self.queue.empty():
            message = self.queue.get_nowait()
            self.write(message)
        self.root.after(100, self.process_queue)

    def save_image_1(self):
        # Save DL model image
        self.save_image(self.output_image_1)

    def save_image_2(self):
        # Save CFD model image
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

