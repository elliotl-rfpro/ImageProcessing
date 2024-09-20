"""Image processing functions and tools."""
import OpenEXR
import Imath
import cv2
import matplotlib.pyplot as plt

from utils.validators import *


@validate_file_path
def read_exr(file_path: str) -> np.ndarray:
    """
    Function for reading .exr files.

    Parameters:
    file_path (str): Path to the .exr file.

    Returns:
    numpy.ndarray: Image read from the .exr file.
    """
    exr_file = OpenEXR.InputFile(file_path)
    header = exr_file.header()
    dw = header['dataWindow']
    size = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)

    # Read the RGB channels
    float_ = Imath.PixelType(Imath.PixelType.FLOAT)
    red = np.frombuffer(exr_file.channel('R', float_), dtype=np.float32).reshape(size)
    green = np.frombuffer(exr_file.channel('G', float_), dtype=np.float32).reshape(size)
    blue = np.frombuffer(exr_file.channel('B', float_), dtype=np.float32).reshape(size)

    # Stack channels to create an image
    image = np.stack([red, green, blue], axis=-1)

    return image


@validate_file_paths
def read_images(fpath_1: str, fpath_2: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Helper function for reading images correctly independent of image format.

    Parameters:
    fpath_1 (str): Path to the first image file.
    fpath_2 (str): Path to the second image file.

    Returns:
    tuple: Two images read from the provided file paths.
    """
    if '.exr' in fpath_1:
        image_1 = read_exr(fpath_1)
    else:
        image_1 = cv2.imread(fpath_1, cv2.IMREAD_UNCHANGED)
    if '.exr' in fpath_2:
        image_2 = read_exr(fpath_2)
    else:
        image_2 = cv2.imread(fpath_2, cv2.IMREAD_UNCHANGED)

    return image_1, image_2


def simulate_mosaicing_rggb(image: np.ndarray, blend: float = None) -> np.ndarray:
    """
    Process a simulated image through a RGGB Bayer pattern in order to simulate mosaicing.

    Parameters:
    image (numpy.ndarray): Input image.
    blend (float, optional): Blending factor for the final image.

    Returns:
    numpy.ndarray: Image after simulating mosaicing.
    """
    # Convert the image to 8-bit if it's not already
    if image.dtype != np.uint8:
        image = (255 * (image - np.min(image)) / (np.max(image) - np.min(image))).astype(np.uint8)

    # Downsample and upsample to simulate pixelation
    scale_factor = 0.5
    small_image = cv2.resize(image, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
    image = cv2.resize(small_image, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Split image into colour channels
    r, g, b = cv2.split(image)

    # Create a Bayer pattern image
    bayer_image = np.zeros_like(r)
    bayer_image[0::2, 0::2] = b[0::2, 0::2]  # Blue
    bayer_image[0::2, 1::2] = g[0::2, 1::2]  # Green
    bayer_image[1::2, 0::2] = g[1::2, 0::2]  # Green
    bayer_image[1::2, 1::2] = r[1::2, 1::2]  # Red

    # Demosaic
    demosaiced_image = cv2.cvtColor(bayer_image, cv2.COLOR_BAYER_BGGR2RGB)

    # Apply optional blending of original image to output
    if blend is not None:
        demosaiced_image = cv2.addWeighted(image, blend, demosaiced_image, 1 - blend, 0)

    return demosaiced_image


@validate_area
def process_images(
        img1: NDArray[Tuple[int, int, int]],
        img2: NDArray[Tuple[int, int, int]],
        area: List[List[int]]
) -> Tuple[NDArray[Tuple[int, int, int]], NDArray[Tuple[int, int, int]]]:
    """
    Normalize, remosaic, greyscale, colour shift etc. input images.

    Parameters:
    img1 (NDArray[Tuple[int, int, int]]): The first input image.
    img2 (NDArray[Tuple[int, int, int]]): The second input image.
    area (List[List[int]]): List of coordinates for cropping the images.
                            Format: [[x1, y1, x2, y2], [x1, y1, x2, y2]]
                            where the first list is for img1 and the second is for img2.

    Returns:
    Tuple[NDArray[Tuple[int, int, int]], NDArray[Tuple[int, int, int]]]:
    The processed images after normalization and cropping.
    """
    # Crop images to regions of interest
    img1 = img1[area[0][1]:area[0][3], area[0][0]:area[0][2]]
    img2 = img2[area[1][1]:area[1][3], area[1][0]:area[1][2]]

    # Normalize img1 with respect to its own maximum value
    max1 = np.max(img1)
    img1 = cv2.normalize(img1, None, alpha=0, beta=max1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    # Normalize img2 with respect to the maximum value of img1
    img2 = cv2.normalize(img2, None, alpha=0, beta=max1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    # Further normalize both images to the range [0, 1] for consistent color correction
    img1 = cv2.normalize(img1, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    img2 = cv2.normalize(img2, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    return img1, img2


@validate_image
def get_colour_histograms(img_data: NDArray[Tuple[int, int, int]]) -> Tuple[np.histogram, np.histogram, np.histogram]:
    """
    Process image data and separate into each colour channel, enabling it to be plotted separately.

    Parameters:
    img_data (NDArray[Tuple[int, int, int]]): The input image data.

    Returns:
    Tuple[np.histogram, np.histogram, np.histogram]:
    Histograms for the red, green, and blue channels.
    """
    # Start by ensuring that img data goes from 0 to 255
    img_data = cv2.normalize(img_data, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    # Convert image data to RGB format
    img_data = np.array(img_data)[:, :, ::-1].copy()

    # Compute histograms for each colour channel
    red_px = np.histogram(img_data[:, :, 0], bins=256, range=[0, 255])
    green_px = np.histogram(img_data[:, :, 1], bins=256, range=[0, 255])
    blue_px = np.histogram(img_data[:, :, 2], bins=256, range=[0, 255])

    return red_px, green_px, blue_px


def manual_colour_adjust(img_in: NDArray[Tuple[int, int, int]], rgb_arr: List) -> NDArray[Tuple[int, int, int]]:
    """
    Manually adjust the colours of a cv2 uint8 image array.

    Parameters:
    img_in (numpy.ndarray): Input image array.
    rgb_arr (list or numpy.ndarray): List or array of RGB adjustments.

    Returns:
    numpy.ndarray: Colour-adjusted image array.
    """
    rgb_arr = np.array(rgb_arr)

    for colour in range(0, 2):
        if rgb_arr[colour] == 0:
            continue
        elif rgb_arr[colour] < 0:
            rgb = np.zeros_like(np.array(img_in), dtype=np.uint8)
            rgb[:, :, colour] = abs(rgb_arr[colour])
            img_in = cv2.subtract(img_in, rgb)
        elif rgb_arr[colour] > 0:
            rgb = np.zeros_like(np.array(img_in), dtype=np.uint8)
            rgb[:, :, colour] = abs(rgb_arr[colour])
            img_in = cv2.add(img_in, rgb)

    return img_in


@validate_images
def white_balance_adjust(
        img1: NDArray[Tuple[int, int, int]],
        img2: NDArray[Tuple[int, int, int]]
) -> NDArray[Tuple[int, int, int]]:
    """
    Adjust the white balance of the first image to match the second image.

    Parameters:
    img1 (numpy.ndarray): First input image.
    img2 (numpy.ndarray): Second input image to match white balance.

    Returns:
    numpy.ndarray: White balance adjusted image.
    """
    # Convert images to LAB color space
    img1_lab = cv2.cvtColor(img1, cv2.COLOR_BGR2LAB)
    img2_lab = cv2.cvtColor(img2, cv2.COLOR_BGR2LAB)

    # Find mean and stdev of L channel
    l_mean1, l_std1 = cv2.meanStdDev(img1_lab[:, :, 0])
    l_mean2, l_std2 = cv2.meanStdDev(img2_lab[:, :, 0])

    # Adjust L channel to match the second image
    l_channel1 = img1_lab[:, :, 0].astype(np.float32)
    l_channel1 = (l_channel1 - l_mean1) * (l_std2 / l_std1) + l_mean2
    l_channel1 = np.clip(l_channel1, 0, 255).astype(np.uint8)
    img1_lab[:, :, 0] = l_channel1

    return cv2.cvtColor(img1_lab, cv2.COLOR_LAB2BGR)


@validate_blend
def auto_colour_adjust(
        img1: NDArray[Tuple[int, int, int]],
        img2: NDArray[Tuple[int, int, int]],
        blend=None
) -> NDArray[Tuple[int, int, int]]:
    """
    Adjust the colours of img1 to suit img2 by finding the median peaks of each channel between images.

    Parameters:
    img1 (numpy.ndarray): First input image.
    img2 (numpy.ndarray): Second input image to match colours.
    blend (float, optional): Blending factor for the final image.

    Returns:
    numpy.ndarray: Colour-adjusted image.
    """
    # Convert images to LAB colour space and split into LAB channels
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2Lab)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2Lab)
    l1, a1, b1 = cv2.split(img1)
    l2, a2, b2 = cv2.split(img2)

    # Apply histogram matching channels
    matched_l = match_histograms(l1, l2)
    matched_a = match_histograms(a1, a2)
    matched_b = match_histograms(b1, b2)

    # Merge back together and convert back to BGR
    matched_image = cv2.merge([matched_l, matched_a, matched_b])
    matched_image = cv2.cvtColor(matched_image, cv2.COLOR_LAB2BGR)

    # Apply bilateral filter to smooth the image while preserving edges
    smoothed_image = cv2.bilateralFilter(matched_image, d=5, sigmaColor=200, sigmaSpace=200)

    # Apply optional blending of original image to output
    if blend is not None:
        final_image = cv2.addWeighted(matched_image, blend, smoothed_image, 1 - blend, 0)
        return final_image

    return smoothed_image


def match_histograms(
        img1: NDArray[Tuple[int, int, int]],
        img2: NDArray[Tuple[int, int, int]]
) -> NDArray[Tuple[int, int, int]]:
    """
    Match the histograms from a source to a template.

    Parameters:
    source (numpy.ndarray): Source image.
    template (numpy.ndarray): Template image.

    Returns:
    numpy.ndarray: Image with matched histograms.
    """
    old_shape = img1.shape
    source = img1.ravel()
    template = img2.ravel()

    # Get the set of unique pixel values and their corresponding indices and counts
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True, return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    # Calculate normalised cumulative distribution functions (CDFs)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # Use interpolation to find the pixel values in the template image which corresponds to each pixel in source
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values).astype(source.dtype)

    return interp_t_values[bin_idx].reshape(old_shape)


def simulate_basic_isp(img1, img2):
    """
    Simulate a basic Image Signal Processor (ISP) pipeline.

    Parameters:
    img1 (numpy.ndarray): The first image.
    img2 (numpy.ndarray): The second image.

    Returns:
    numpy.ndarray: The processed image after ISP simulation.
    """
    # If images are not the same size, then reshape
    if img1.shape[0] != img2.shape[0] or img1.shape[1] != img2.shape[1]:
        img2 = reshape_img(img1, img2)

    # Perform white balance and colour correction
    img1_corr = white_balance_adjust(img1, img2)
    img1_corr = auto_colour_adjust(img1_corr, img2, blend=0.5)

    # Perform simulated mosaicing
    img1_mos = simulate_mosaicing_rggb(img1_corr, blend=0.5)

    return img1_mos


def reshape_img(img1, img2):
    """Reshape img2 to match the dimensions of img1"""
    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    return img2
