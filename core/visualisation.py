import cv2
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
from skimage.metrics import structural_similarity as ssim

from utils.validators import *
from core.improc import get_colour_histograms, reshape_img

sns.set(palette="dark", font_scale=1.1, color_codes=True)
sns.set_style('darkgrid', {'axes.linewidth': 1, 'axes.edgecolor': 'black'})

matplotlib.use('TkAgg')


def calculate_ssim(img1: NDArray[Tuple[int, int, int]], img2: NDArray[Tuple[int, int, int]]) -> float:
    # Greyscale images
    img1_g = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    img2_g = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

    # Resize regions so that they match
    img2_g = reshape_img(img1_g, img2_g)
    img2_g = cv2.resize(img2_g, (img1.shape[1], img1.shape[0]))

    ssim_, _ = ssim(img1_g, img2_g, full=True, data_range=np.max(img1), channel_axis=-1)

    return ssim_


def plot_colour_histogram(img):
    """
    Plot the colour histogram of an image.

    Parameters:
    img (numpy.ndarray): The input image.

    Returns:
    None
    """
    # Get histograms for each colour channel
    b1, g1, r1 = get_colour_histograms(img)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(r1[1][:256], r1[0], color='r')
    plt.plot(g1[1][:256], g1[0], color='g')
    plt.plot(b1[1][:256], b1[0], color='b')
    plt.title('Colour histograms')
    plt.tight_layout()
    plt.show()


def view_image(img):
    """
    Display an image using matplotlib.

    Parameters:
    img (numpy.ndarray): The input image.

    Returns:
    None
    """
    plt.figure(figsize=(10, 6))
    plt.imshow(img)
    plt.grid(False)
    plt.show()


def plot_lineout(img, line):
    """
    Plot the lineout of a specific row in the image.

    Parameters:
    img (numpy.ndarray): The input image.
    line (int): The row index to plot the lineout for.

    Returns:
    None
    """
    plt.figure(figsize=(10, 6))
    plt.plot(img[line, :], '.-')
    plt.title(f'Lineout for row {line}')
    plt.ylabel('Pixel Intensity')
    plt.xlabel('Pixel index')
    plt.legend()
    plt.show()
