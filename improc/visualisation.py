import cv2
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
from skimage.metrics import structural_similarity as ssim

from improc.utils.validators import *
from improc.improc import get_colour_histograms, reshape_img

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


def view_raw(image: NDArray, title: str = '') -> None:
    """
    Normalise a .RAW image loaded as a np array between 0 and 255 in uint8 type, then simply plot using pyplot.

    Parameters:
        image (numpy.ndarray): The input image, which is a NumPy array.
        title (str): The title of the plot.
    """
    # Normalize the image to the range [0, 1]
    normalized_image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    # Convert the normalized image to 8-bit
    rgb_image = (normalized_image * 255).astype(np.uint8)

    # View image
    plt.imshow(rgb_image)
    plt.title(title)
    plt.grid(False)
    plt.show()


def stb2rgb(image: NDArray, view: bool = False):
    """
    Outputs and optionally displays a .stb file in RGB format.

    Parameters:
        image (numpy.ndarray): The image to be viewed.
        view (bool): Bool flag to visualise the image, if True.

    Returns:
        balanced_image (numpy.ndarray): The .stb image converted into 8bit RGB.
    """

    # Convert the RGGB Bayer pattern to an RGB image using OpenCV
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BAYER_GRBG2RGB)

    # Adjust white balance. This calculates the average color of the image. It's crude so won't be fully correct
    # or replicate the algorithm which is performed within the Sony signal processing chain, but as this is just
    # for observation purposes, this was acceptable for me.
    avg_color_per_row = np.average(rgb_image, axis=0)
    avg_color = np.average(avg_color_per_row, axis=0)
    balanced_image = rgb_image / avg_color * 128

    balanced_image = cv2.normalize(balanced_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # cv2 defaults to BGR for some inexplicable reason
    balanced_image = cv2.cvtColor(balanced_image, cv2.COLOR_BGR2RGB)

    # Display the image using imshow, if view is passed as true
    if view:
        plt.imshow(balanced_image)
        plt.grid(False)
        plt.show()

    # Optionally return the 8bit RGB image for further processing
    return balanced_image


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
