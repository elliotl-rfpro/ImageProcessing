"""Example for the image processing workflow of this module."""
import cv2
import matplotlib.pyplot as plt

from core.improc import read_images, process_images, simulate_basic_isp
from core.visualisation import calculate_ssim, view_image, reshape_img


if __name__ == '__main__':
    # Load the images
    fpath1 = 'C:/Users/ElliotLondon/Documents/PythonLocal/ImageProcessing/examples/roadsign1.png'
    fpath2 = 'C:/Users/ElliotLondon/Documents/PythonLocal/ImageProcessing/examples/roadsign2.png'
    img1, img2 = read_images(fpath1, fpath2)

    # Change from BGR to RGB (cv2 uses BGR by default...)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    # Define regions of interest
    area1 = [19, 84, 147, 227]
    area2 = [15, 55, 176, 198]
    area = [area1, area2]

    # Process images
    img1_p, img2_p = process_images(img1, img2, area=area)

    # Show regions of interest for defaults
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(img1_p)
    ax[1].imshow(img2_p)
    ax[0].grid(False)
    ax[1].grid(False)
    plt.show()

    # Calculate SSIM for entire images, for entire image and for regions of interest
    ssim_total = calculate_ssim(img1, img2)
    ssim_roi = calculate_ssim(img1_p, img2_p)
    print(f'SSIM total = {ssim_total}')
    print(f'SSIM ROI = {ssim_roi}')

    # Repeat, but now simulating basic ISP
    img1_isp = simulate_basic_isp(img1_p, img2_p)
    ssim_isp = calculate_ssim(img1_isp, img2_p)
    print(f'SSIM ISP = {ssim_roi}')

    # Show regions of interest for ISP
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(img1_isp)
    ax[1].imshow(img2_p)
    ax[0].grid(False)
    ax[1].grid(False)
    plt.show()

    print('Example: improc_example.py completed successfully.')
