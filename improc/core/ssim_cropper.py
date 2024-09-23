"""Class which enables two sections of two different images to be manually cropped and compared."""
import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import Scrollbar


class ImageCropper:
    def __init__(self, root, image_path):
        """
        Initialize the ImageCropper class.

        Parameters:
        root (tk.Tk): The root Tkinter window.
        image_path (str): The path to the image to be loaded.
        """
        self.tk_image = None
        self.image = None
        self.root = root
        self.image_path = image_path
        self.rect_coords = []
        self.rects = []

        # Load the first image
        self.load_image()

        # Initialize rectangle variables
        self.start_x = None
        self.start_y = None
        self.rect = None

        # Bind mouse events to the canvas
        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)

    def load_image(self):
        """
        Load the image using OpenCV and display it on a Tkinter canvas.
        """
        # Load image using OpenCV
        self.image = cv2.imread(self.image_path, cv2.IMREAD_UNCHANGED)
        local_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        local_image = cv2.normalize(local_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)
        local_image = Image.fromarray(local_image)
        self.tk_image = ImageTk.PhotoImage(local_image)

        # Create canvas to display the image with scroll bars
        self.canvas = tk.Canvas(self.root, width=800, height=600,
                                scrollregion=(0, 0, self.tk_image.width(), self.tk_image.height()))
        self.canvas.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)

        # Add scroll bars
        hbar = Scrollbar(self.root, orient=tk.HORIZONTAL, command=self.canvas.xview)
        hbar.pack(side=tk.BOTTOM, fill=tk.X)
        vbar = Scrollbar(self.root, orient=tk.VERTICAL, command=self.canvas.yview)
        vbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.config(xscrollcommand=hbar.set, yscrollcommand=vbar.set)

        # Add the image to the canvas
        self.canvas.create_image(0, 0, anchor='nw', image=self.tk_image)

    def on_button_press(self, event):
        """
        Handle the button press event to start drawing a rectangle.

        Parameters:
        event (tk.Event): The Tkinter event object.
        """
        # Save starting coordinates
        self.start_x = self.canvas.canvasx(event.x)
        self.start_y = self.canvas.canvasy(event.y)

        # If a rectangle exists, delete it
        if self.rect:
            self.canvas.delete(self.rect)
        self.rect = None

    def on_mouse_drag(self, event):
        """
        Handle the mouse drag event to update the rectangle as the mouse is dragged.

        Parameters:
        event (tk.Event): The Tkinter event object.
        """
        # Update rectangle as mouse is dragged
        cur_x, cur_y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)

        if self.rect:
            self.canvas.delete(self.rect)

        self.rect = self.canvas.create_rectangle(self.start_x, self.start_y, cur_x, cur_y, outline="red")

    def on_button_release(self, event):
        """
        Handle the button release event to finalize the rectangle and output its coordinates.

        Parameters:
        event (tk.Event): The Tkinter event object.
        """
        # Output coordinates of the rectangle when mouse is released
        end_x, end_y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        print(f'Rect coords: ({self.start_x}, {self.start_y}), ({end_x}, {end_y})')

        # Normalize coordinates if dragged from bottom right to top left
        x1, x2 = sorted([self.start_x, end_x])
        y1, y2 = sorted([self.start_y, end_y])
        print(f'Normalized coords: x1 = {x1}, y1 = {y1}, x2 = {x2}, y2 = {y2}')

        self.rects.append([x1, y1, x2, y2])
        self.canvas.delete(self.rect)
        self.root.destroy()
