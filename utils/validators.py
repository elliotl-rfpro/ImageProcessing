"""Functions used to validate other functions within this package"""
import numpy as np
import os
from numpy.typing import NDArray
from typing import Tuple, List, Callable


def validate_file_path(func: Callable[[str], np.ndarray]) -> Callable[[str], np.ndarray]:
    def wrapper(file_path: str, *args, **kwargs) -> np.ndarray:
        if not isinstance(file_path, str):
            raise TypeError("file_path must be a string")
        return func(file_path, *args, **kwargs)
    return wrapper


def validate_file_paths(func):
    def wrapper(fpath_1: str, fpath_2: str) -> tuple[np.ndarray, np.ndarray]:
        # Check if fpath_1 and fpath_2 are strings
        if not isinstance(fpath_1, str) or not isinstance(fpath_2, str):
            raise TypeError("Both file paths must be strings")
        # Check if the files exist
        if not os.path.isfile(fpath_1):
            raise FileNotFoundError(f"File not found: {fpath_1}")
        if not os.path.isfile(fpath_2):
            raise FileNotFoundError(f"File not found: {fpath_2}")
        return func(fpath_1, fpath_2)

    return wrapper


def validate_blend(func):
    def wrapper(img1: NDArray[Tuple[int, int, int]], img2: NDArray[Tuple[int, int, int]], blend=None):
        if blend is not None and not (0 <= blend <= 1):
            raise ValueError("Blend must be None, or between 0 and 1")
        return func(img1, img2, blend)
    return wrapper


def validate_image(func):
    def wrapper(img1: NDArray[Tuple[int, int, int]]):
        # Check if source and template are numpy arrays
        if not isinstance(img1, np.ndarray):
            raise TypeError("img1 and img2 must be a numpy array")

        # Check if source and template are 3-channel images
        if img1.ndim != 3 or img1.shape[2] != 3:
            raise ValueError("img1 must be a 3-channel image")

        return func(img1)
    return wrapper


def validate_images(func):
    def wrapper(img1: NDArray[Tuple[int, int, int]], img2: NDArray[Tuple[int, int, int]]):
        # Check if source and template are numpy arrays
        if not isinstance(img1, np.ndarray) or not isinstance(img2, np.ndarray):
            raise TypeError("Both img1 and img2 must be numpy arrays")

        # Check if source and template have the same shape
        if img1.shape != img2.shape:
            raise ValueError("img1 and img2 must have the same shape")

        # Check if source and template are 3-channel images
        if img1.ndim != 3 or img2.ndim != 3 or img1.shape[2] != 3 or img2.shape[2] != 3:
            raise ValueError("img1 and img2 must be 3-channel images")

        return func(img1, img2)
    return wrapper


def validate_area(func):
    def wrapper(img1: NDArray[Tuple[int, int, int]], img2: NDArray[Tuple[int, int, int]], area: List):
        if not isinstance(area, list) and not isinstance(area, np.ndarray):
            raise TypeError("Area must either be a list or numpy array.")
        if len(area) != 2:
            raise ValueError("Area must contain two elements (roi1, roi2)")
        for sublist in area:
            for item in sublist:
                if not isinstance(item, int):
                    raise ValueError("All elements in the list must be integers")

        return func(img1, img2, area)
    return wrapper
