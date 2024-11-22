"""Library for functions which enable access to various camera sensors, and processing of their data."""
import os

import matplotlib.pyplot as plt
import numpy as np
import array
import sys
import pathlib
import time
from datetime import datetime, timedelta
import cv2

capture_folder = "C:/Users/ElliotLondon/Documents/PLITS/Capture"


# def auto_white_balance(image: NDArray):
#     """Performs auto white balancing upon the input image."""
#     # Convert the image to float32 for precision
#     image = image.astype(np.float32)
#
#     # Compute the average of each channel
#     avg_b = np.mean(image[:, :, 0])
#     avg_g = np.mean(image[:, :, 1])
#     avg_r = np.mean(image[:, :, 2])
#
#     # Compute the overall average
#     avg_gray = (avg_b + avg_g + avg_r) / 3
#
#     # Compute the scaling factors for each channel
#     scale_b = avg_gray / avg_b
#     scale_g = avg_gray / avg_g
#     scale_r = avg_gray / avg_r
#
#     # Apply the scaling factors to each channel
#     image[:, :, 0] *= scale_b
#     image[:, :, 1] *= scale_g
#     image[:, :, 2] *= scale_r
#
#     # # Clip the values to the valid range [0, 255] and convert to uint8
#     # image = np.clip(image, 0, 255).astype(np.uint8)
#
#     return image


def auto_white_balance(image, reference_white):
    # Convert the image to LAB color space
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Split the LAB image into its channels
    l, a, b = cv2.split(lab_image)

    # Calculate the average a and b values of the reference white area
    avg_a = np.mean(a[reference_white])
    avg_b = np.mean(b[reference_white])

    # Calculate the difference from the ideal white point (128, 128)
    delta_a = 128 - avg_a
    delta_b = 128 - avg_b

    # Apply the correction to the entire image
    a = a + delta_a
    b = b + delta_b

    # Ensure the channels have the same size and depth
    a = np.clip(a, 0, 255).astype(np.uint8)
    b = np.clip(b, 0, 255).astype(np.uint8)

    # Merge the channels back
    corrected_lab_image = cv2.merge([l, a, b])

    # Convert back to BGR color space
    corrected_image = cv2.cvtColor(corrected_lab_image, cv2.COLOR_LAB2BGR)

    return corrected_image


def gamma_correction(image, gamma):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


def read_stb_file(file_path: str) -> np.ndarray:
    """Reads a .stb file
    Args:
        inputFilePath (str): Path to .stb file
    Returns:
        np.ndarray: Image
    """

    assert os.path.exists(file_path), f"File {file_path} does not exist."

    with open(file_path, "rb") as fid:
        data_type = array.array("L")
        image_shape = array.array("L")
        data_type.fromfile(fid, 2)
        if data_type[1] == 0:
            image_data = array.array("H")  # H is the typecode for uint16
        else:
            image_data = array.array("L")  # L is the typecode for uint32
        image_shape.fromfile(fid, 2)
        image_data.fromfile(fid, image_shape[0] * image_shape[1])

    image_data = np.reshape(image_data, (image_shape[1], image_shape[0]))

    return image_data


def plits_request(block, method, param='', print_=True):
    """Function to interact with a PLITS instance."""
    req_str = block + '.' + method

    if isinstance(param, str):
        if len(param) >= 1:
            req_str += ',' + param
    else:
        req_str += ',' + str(param)

    ret, message = plits.Request(req_str)
    log = '[PLITSRequest] ' + req_str + ' -> ret=' + str(ret)
    if len(message) >= 1:
        log += ' message=' + message
    if print_:
        print(log)
    return ret, message


def capture_images_for_duration(duration_minutes, wait_time):
    end_time = datetime.now() + timedelta(minutes=duration_minutes)

    print(f'Recording an image every {wait_time} seconds, for {duration_minutes} minutes...')
    while datetime.now() < end_time:
        # Wait until the next whole minute
        now = datetime.now()
        next_minute = (now + timedelta(minutes=1)).replace(second=0, microsecond=0)
        time_to_wait = (next_minute - now).total_seconds()
        time.sleep(time_to_wait)

        # Capture an image every minute for 5 minutes
        for _ in range(5):
            if datetime.now() >= end_time:
                break
            ret, message = plits_request('SmartGUI', 'ControlUI', 'Capture', print_=False)
            print("Image captured at", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

            time.sleep(60)  # Wait for 60 seconds (1 minute) before capturing the next image


def load_most_recent_image(directory):
    files = os.listdir(directory)

    # Only consider .stb files
    image_files = [f for f in files if f.endswith('.stb')]
    image_files.sort(key=lambda x: os.path.getmtime(os.path.join(directory, x)), reverse=True)

    # Load the most recent image, return it and its name
    most_recent_image_path = os.path.join(directory, image_files[0])
    most_recent_image = plt.imread(most_recent_image_path)
    return most_recent_image, image_files[0]


if __name__ == '__main__':
    """Example script: connect to PLITS, then run capture_images_for_duration and close PLITS."""
    # Add path with PythonToPLITS module
    plits_dir = pathlib.Path('C:/Users/ElliotLondon/Documents/PLITS/dll')
    sys.path.append(str(plits_dir))
    import PythonToPLITS as plits

    # Connect to PLITS
    hostname = "localhost"
    port = "50001"
    ret = plits.Open(hostname, port)
    print('Open(hostname=' + hostname + ' , port=' + port + ') , ret=' + str(ret))

    # Ensure that PLITS is connected properly
    ret, message = plits_request('Core', 'setConnectOn', '10000')
    ret, message = plits_request('Core', 'MonitorOn')
    ret, message = plits_request('Core', 'getSequencer')

    # Capture an image, on the minute, for X minutes
    capture_images_for_duration(duration_minutes=200, wait_time=300)

    # PLITS disconnecting
    ret = plits.Close()
    print('Close , ret=' + str(ret))
