"""File to store object locations within simulated/measured images, and related functions for processing
macbeth charts."""
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from typing import List
import cv2

"""Measured macbeth chart: test_macbeth.exr"""
regions_measure = [
    [950, 1300, 1050, 1400], [950, 1500, 1050, 1600], [950, 1700, 1050, 1800], [950, 1900, 1050, 2000], [950, 2100, 1050, 2200], [1000, 2300, 1100, 2400],
    [1100, 1300, 1200, 1400], [1150, 1500, 1250, 1600], [1150, 1700, 1250, 1800], [1150, 1900, 1250, 2000], [1150, 2100, 1250, 2200], [1200, 2300, 1300, 2400],
    [1300, 1275, 1400, 1375], [1350, 1475, 1450, 1575], [1350, 1650, 1450, 1750], [1350, 1875, 1450, 1975], [1350, 2100, 1450, 2200], [1400, 2300, 1500, 2375],
    [1500, 1250, 1600, 1350], [1550, 1450, 1650, 1550], [1550, 1650, 1650, 1750], [1550, 1850, 1650, 1950], [1550, 2050, 1650, 2150], [1575, 2250, 1675, 2350],
]

"""Rendered macbeth chart: sim_macbeth_20241031_cd07.exr"""
regions_render = [
    [380, 580, 480, 680], [380, 880, 480, 980], [380, 1280, 480, 1380], [380, 1580, 480, 1680], [380, 1880, 480, 1980], [380, 2180, 480, 2280],
    [780, 580, 880, 680], [780, 880, 880, 980], [780, 1280, 880, 1380], [780, 1580, 880, 1680], [780, 1880, 880, 1980], [780, 2180, 880, 2280],
    [1080, 580, 1180, 680], [1080, 880, 1180, 980], [1080, 1280, 1180, 1380], [1080, 1580, 1180, 1680], [1080, 1880, 1180, 1980], [1080, 2180, 1180, 2280],
    [1380, 580, 1480, 680], [1380, 880, 1480, 980], [1380, 1280, 1480, 1380], [1380, 1580, 1480, 1680], [1380, 1880, 1480, 1980], [1380, 2180, 1480, 2280]
]

# List of color names corresponding to the Macbeth chart squares
colour_names_landscape = [
    'dark_skin', 'light_skin', 'blue_sky', 'foliage', 'blue_flower', 'bluish_green',
    'orange', 'purplish_blue', 'moderate_red', 'purple', 'yellow_green', 'orange_yellow',
    'blue', 'green', 'red', 'yellow', 'magenta', 'orange_yellow',
    'white', 'neutral_8', 'neutral_6d5', 'neutral_5', 'neutral_3d5', 'black'
]

colour_names_portrait = [
    'white', 'blue', 'orange', 'dark_skin',
    'neutral_8', 'green', 'purplish_blue', 'light_skin',
    'neutral_6d5', 'red', 'moderate_red', 'blue_sky',
    'neutral_5', 'yellow', 'purple', 'foliage',
    'neutral_3d5', 'magenta', 'yellow_green', 'blue_flower',
    'black', 'cyan', 'orange_yellow', 'bluish_green',
]


def display_colours(image, title: str, orientation: str = 'landscape'):
    """
    Visualize different Macbeth charts.

    Parameters:
        image (array-like): The input image, which can be a NumPy array or an image file.
        title (str): The title of the plot.
        orientation (str): The orientation of the plot, either 'landscape' or 'portrait'.
    Raises:
        ValueError: If the orientation is not 'landscape' or 'portrait'.
    """

    # Ensure the image is a NumPy array
    if not isinstance(image, np.ndarray):
        image = np.array(image)

    # Check if the image is likely in .raw format and normalize if so
    if np.max(image) > 255:
        image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        image = (image * 255).astype(np.uint8)

    # Set the layout based on the orientation
    if orientation == 'landscape':
        rows = 4
        columns = 6
        figsize = (10, 6)
    elif orientation == 'portrait':
        rows = 6
        columns = 4
        figsize = (6, 10)
    else:
        raise ValueError("Arg: 'orientation' requires value 'landscape' or 'portrait'.")

    # Create the plot
    fig, axes = plt.subplots(rows, columns, figsize=figsize)
    fig.suptitle(title, color='white')

    # Iterate over each subplot and display the corresponding color
    for i, ax in enumerate(axes.flat):
        colour = image[i]  # Normalize to [0, 1] for display
        ax.imshow(colour)
        ax.axis('off')  # Hide the axis

    # Set the background color of the figure
    fig.patch.set_facecolor('black')


def get_bayer_subpattern(x0: int, y0: int) -> str:
    """
    Determine the Bayer pattern based on the starting pixel coordinates.

    Parameters:
        x0 (int): The initial x coordinate
        y0 (int): The initial y coordinate

    Returns:
        str corresponding to bayer pattern
    """
    if (x0 % 2 == 0) and (y0 % 2 == 0):
        return 'RGGB'
    elif (x0 % 2 == 1) and (y0 % 2 == 0):
        return 'GRBG'
    elif (x0 % 2 == 0) and (y0 % 2 == 1):
        return 'GBRG'
    else:
        return 'BGGR'


def get_subpixels(image: NDArray, regions: List, avg: bool = True) -> NDArray:
    """
    Each colour within the measured image has its own Bayer pattern format as a result of the top-left pixel
    corresponding to an unknown colour. This script analyses the displacement from the (0, 0) pixel using the
    function get_bayer_subpattern and returns a complete list of all subpixels for each channel (RGB).

    Parameters:
        image (numpy.ndarray): The input image array as a NumPy array.
        regions (list): The list of regions to consider within the image, given in image coordinates [x1, y1, x2, y2].
        avg (bool): Bool flag to whether the return is averaged or same as input array.

    Returns:
        NDArray of the subpixels, provided in indexed order within each channel.
    """

    results = []
    for region in regions:
        x1, y1, x2, y2 = region
        subregion = image[x1:x2, y1:y2]
        bayer_pattern = get_bayer_subpattern(x1, y1)

        r_pixels = []
        g_pixels = []
        b_pixels = []
        for i in range(subregion.shape[0]):
            for j in range(subregion.shape[1]):
                if bayer_pattern == 'RGGB':
                    if i % 2 == 0 and j % 2 == 0:
                        r_pixels.append(subregion[i, j])
                    elif i % 2 == 0 and j % 2 == 1:
                        g_pixels.append(subregion[i, j])
                    elif i % 2 == 1 and j % 2 == 0:
                        g_pixels.append(subregion[i, j])
                    else:
                        b_pixels.append(subregion[i, j])
                elif bayer_pattern == 'GRBG':
                    if i % 2 == 0 and j % 2 == 0:
                        g_pixels.append(subregion[i, j])
                    elif i % 2 == 0 and j % 2 == 1:
                        r_pixels.append(subregion[i, j])
                    elif i % 2 == 1 and j % 2 == 0:
                        b_pixels.append(subregion[i, j])
                    else:
                        g_pixels.append(subregion[i, j])
                elif bayer_pattern == 'GBRG':
                    if i % 2 == 0 and j % 2 == 0:
                        g_pixels.append(subregion[i, j])
                    elif i % 2 == 0 and j % 2 == 1:
                        b_pixels.append(subregion[i, j])
                    elif i % 2 == 1 and j % 2 == 0:
                        r_pixels.append(subregion[i, j])
                    else:
                        g_pixels.append(subregion[i, j])
                else:  # BGGR
                    if i % 2 == 0 and j % 2 == 0:
                        b_pixels.append(subregion[i, j])
                    elif i % 2 == 0 and j % 2 == 1:
                        g_pixels.append(subregion[i, j])
                    elif i % 2 == 1 and j % 2 == 0:
                        g_pixels.append(subregion[i, j])
                    else:
                        r_pixels.append(subregion[i, j])

        if avg:
            # Average all channels and append to results
            r_avg = np.mean(r_pixels)
            g_avg = np.mean(g_pixels)
            b_avg = np.mean(b_pixels)
            results.append(np.array([r_avg, g_avg, b_avg]))
        else:
            results.append([np.array(r_pixels), np.array(g_pixels), np.array(b_pixels)])

    return np.array(results)


def scale_subpixels(image, scaling_factors):
    """
    Apply a series of multiplications to each pixel in an image.

    Parameters:
        image (numpy.ndarray): The input image with Bayer pattern.
        scaling_factors (list): A list of scaling factors for R, G, and B channels.

    Returns:
        numpy.ndarray: The scaled image.
    """

    # Ensure the image is a NumPy array
    if not isinstance(image, np.ndarray):
        image = np.array(image)

    # Ensure the scaling factors are a NumPy array
    scaling_factors = np.array(scaling_factors)

    # Create an empty array to store the scaled image
    scaled_image = np.zeros_like(image, dtype=np.float32)

    # Determine the Bayer pattern for each pixel and apply the corresponding scaling factor
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if (i % 2 == 0) and (j % 2 == 0):  # R pixel
                scaled_image[i, j] = image[i, j] * scaling_factors[0]
            elif (i % 2 == 0) and (j % 2 == 1):  # G pixel
                scaled_image[i, j] = image[i, j] * scaling_factors[1]
            elif (i % 2 == 1) and (j % 2 == 0):  # G pixel
                scaled_image[i, j] = image[i, j] * scaling_factors[1]
            else:  # B pixel
                scaled_image[i, j] = image[i, j] * scaling_factors[2]

    return scaled_image


def calculate_mses(image1: NDArray, image2: NDArray) -> List:
    """
    Evaluate mean squared error for each value between two images and sum for global figure of merit.

    Parameters:
        image1 (numpy.ndarray): The first image
        image2 (numpy.ndarray): The second image

    Returns:
        mses (List): List of mses.
    """
    image1 = np.array(image1)
    image2 = np.array(image2)
    mses = []
    for img1, img2 in zip(image1, image2):
        channel_mses = []
        for channel in range(3):  # Iterate over R, G, B channels
            mse = np.mean((img1[channel] - img2[channel]) ** 2)
            channel_mses.append(mse)
        mses.append(channel_mses)
    return mses


def apply_colour_filters(subregions: List, rgb_weights: List) -> NDArray:
    """
    Apply separate colour filters to an array of image subregions depending upon the supplied weightings.

    Parameters:
        subregions (list): List of subregions of an image, in RGB.
        rgb_weights (list): List of RGB weightings.

    Returns:
        image (numpy.ndarray): the filtered image.
    """
    filtered_img = []
    for subregion in subregions:
        r_new = (subregion[0] * rgb_weights[0] +
                 subregion[1] * rgb_weights[1] +
                 subregion[2] * rgb_weights[2]) + \
                 rgb_weights[9]
        g_new = (subregion[0] * rgb_weights[3] +
                 subregion[1] * rgb_weights[4] +
                 subregion[2] * rgb_weights[5]) + \
                 rgb_weights[10]
        b_new = (subregion[0] * rgb_weights[6] +
                 subregion[1] * rgb_weights[7] +
                 subregion[2] * rgb_weights[8]) + \
                 rgb_weights[11]
        filtered_img.append([r_new, g_new, b_new])

    return np.array(filtered_img)


def apply_global_filter(image: NDArray, rgb_weights: List) -> NDArray:
    """
    Apply colour filters to each channel of an image depending upon the supplied weightings.

    Parameters:
        image (numpy.ndarray): The input image
        rgb_weights (list): List of RGB weightings including global scaling factors.

    Returns:
        image (numpy.ndarray): the filtered image.
    """
    # Create a copy of the image to avoid modifying the original
    filtered_image = image.copy()

    # Apply the colour filters with global scaling factors
    filtered_image[:, :, 0] = (image[:, :, 0] * rgb_weights[0] +
                               image[:, :, 1] * rgb_weights[1] +
                               image[:, :, 2] * rgb_weights[2] +
                               rgb_weights[9])

    filtered_image[:, :, 1] = (image[:, :, 0] * rgb_weights[3] +
                               image[:, :, 1] * rgb_weights[4] +
                               image[:, :, 2] * rgb_weights[5] +
                               rgb_weights[10])

    filtered_image[:, :, 2] = (image[:, :, 0] * rgb_weights[6] +
                               image[:, :, 1] * rgb_weights[7] +
                               image[:, :, 2] * rgb_weights[8] +
                               rgb_weights[11])

    # Ensure values are within the valid range for uint8 format
    filtered_image = np.clip(filtered_image, 0, 255).astype(np.uint8)

    return filtered_image


def apply_channel_scaling(image: NDArray, r_scale: float, g_scale: float, b_scale: float) -> NDArray:
    # Split the image into its R, G, and B channels
    b, g, r = cv2.split(image)

    # Apply the scaling factors
    r = np.clip(r * r_scale, 0, 255).astype(np.uint8)
    g = np.clip(g * g_scale, 0, 255).astype(np.uint8)
    b = np.clip(b * b_scale, 0, 255).astype(np.uint8)

    # Merge the channels back together
    scaled_image = cv2.merge([b, g, r])

    return scaled_image


def eval_filters(rgb_weights: List, rgb1: List, rgb2: List) -> np.signedinteger:
    """
    Evaluate the applied filters. Used as a metric to determine the optimal colour filter settings.

    Parameters:
        rgb_weights (list): List of weights including global shifts.
        rgb1 (list): List of RGB values.
        rgb2 (list): List of RGB values.

    Returns:
        numpy.signedinteger: Integer value which corresponds to the performance of the current filter settings.
    """
    r_filter = rgb_weights[:3]
    g_filter = rgb_weights[3:6]
    b_filter = rgb_weights[6:9]
    r_shift = rgb_weights[9]
    g_shift = rgb_weights[10]
    b_shift = rgb_weights[11]

    estimated_rgb2 = np.zeros_like(rgb1)
    estimated_rgb2[:, 0] = np.dot(rgb1, r_filter) + r_shift
    estimated_rgb2[:, 1] = np.dot(rgb1, g_filter) + g_shift
    estimated_rgb2[:, 2] = np.dot(rgb1, b_filter) + b_shift

    return np.sum((estimated_rgb2 - rgb2) ** 2)


def plot_differences(image1: NDArray, image2: NDArray, title: str = '') -> NDArray:
    """
    Plot the differences between two lists of RGB colour arrays.

    Parameters:
        image1 (numpy.ndarray): The first image
        image2 (numpy.ndarray): The second image
        title (str): Title of the plot (optional)

    Returns:
        diff (numpy.ndarray): The array of differences between each region of the two images (RGB)
    """
    # Work out whether the images are in HDR or u8bit format and normalize accordingly to have agnostic input arrays.
    if isinstance(image1[0][0], np.uint8) or isinstance(image1[0][0], int):
        pass
    else:
        image1 = cv2.normalize(image1, None, 0, 255, cv2.NORM_MINMAX).astype(np.float32)
        image2 = cv2.normalize(image2, None, 0, 255, cv2.NORM_MINMAX).astype(np.float32)
        image1 = np.clip(image1, 0, 255)
        image2 = np.clip(image2, 0, 255)

    # Calculate difference between arrays and percentage differences
    diff = np.array(image1) - np.array(image2)
    diff_pcent = []
    for element in zip(image1, image2):
        r = abs(element[0][0] - element[1][0]) / ((element[0][0] + element[1][0]) / 2) * 100
        g = abs(element[0][1] - element[1][1]) / ((element[0][1] + element[1][1]) / 2) * 100
        b = abs(element[0][2] - element[1][2]) / ((element[0][2] + element[1][2]) / 2) * 100
        diff_pcent.append([r, g, b])
    diff_pcent = np.array(diff_pcent)

    # Create a figure
    fig, axes = plt.subplots(4, 6, figsize=(10, 7))
    cmap = plt.get_cmap('inferno')

    # Find the maximum percentage difference for scaling
    max_diff = np.max(diff_pcent)

    # Iterate over each region and plot the RGB differences
    for i in range(4):
        for j in range(6):
            ax = axes[i, j]
            region_idx = i * 6 + j
            rgb_diff = diff_pcent[region_idx]

            # Create a heatmap for the current region
            cax = ax.matshow(rgb_diff.reshape(1, -1), cmap=cmap, aspect='auto', vmin=0, vmax=max_diff)

            # Add text annotations with the most visible color
            for k in range(3):
                text_color = 'white' if rgb_diff[k] < max_diff / 2 else 'black'
                ax.text(k, 0, f"{rgb_diff[k]:.0f}", ha="center", va="center", color=text_color)

            # Set axis labels and ticks
            ax.set_yticks([])
            ax.set_xticks([])

    # Set the main title
    plt.suptitle(f'Percentage Differences Between Regions [R, G, B]. Total diff: {np.sum(np.abs(diff)):.2f}', fontsize=16)
    plt.grid(False)

    # Create RGB arrays for plotting
    rs = [r for r in np.array(diff)[:, 0]]
    gs = [g for g in np.array(diff)[:, 1]]
    bs = [b for b in np.array(diff)[:, 2]]

    plt.figure(figsize=(8, 6))
    plt.title(title)
    plt.plot(range(24), rs, 'ro')
    plt.plot(range(24), gs, 'go')
    plt.plot(range(24), bs, 'bo')
    plt.ylabel('Pixel value')

    return diff


def process_macbeth_colours(chart: NDArray, rows: int, columns: int, regions: List, ao: str, avg: bool = False) -> list:
    """
    Load and analyze a reference Macbeth chart.

    Parameters:
        chart (NDArray): The loaded chart, in pixel coordinates.
        rows (int): Number of rows within the Macbeth chart image.
        columns (int): Number of columns within the Macbeth chart image.
        regions (List): The regions of the colours within the Macbeth chart image. Manually defined as [x1, y1, x2, y2].
        ao (str): Flag for 'array orientation' that enables a switch case between [x1, y1] and [y1, x1] formats.
        avg (bool): Bool whether to return averaged (True) or un-averaged (False) colour data.

    Returns:
        list: A list containing either the average luminance value (if .RAW. input array), or the average of each
        RGB channel (if 3-channel input array).
    """

    # Initialize a list to store the average color values
    colour_values = []

    # Loop through each color square to calculate average values
    i = 0
    for row in range(rows):
        for column in range(columns):
            # Crop the current square from the image using the calculated coordinates
            region = regions[i]
            if ao == 'xy':
                current_colour = chart[region[0]:region[2], region[1]:region[3]]
            elif ao == 'yx':
                current_colour = chart[region[1]:region[3], region[0]:region[2]]
            else:
                raise ValueError("ao param requires values of 'xy' or 'yx'.")

            if avg:
                if len(chart.shape) == 3 and chart.shape[2] == 3:
                    # If RGB, calculate average colour for each channel
                    avg_colour = np.mean(current_colour, axis=(0, 1))
                elif len(chart.shape) == 2:
                    # If single channel .raw, throw an exception:
                    raise NotImplementedError('You have passed flag avg=True for a 2d .RAW image. Averaging over the '
                                              'pixel values without demosaicing or subpixel processing would create '
                                              'an unphysical array of values. Script exiting...')
                else:
                    raise ValueError(f'arg[0] has incorrect dimensions: must be np.NDArray with shape [x, y, 3] or '
                                     f'[x, y]. Script exiting...')
                colour_values.append(avg_colour)
            else:
                colour_values.append(current_colour)
            i += 1

    return colour_values
