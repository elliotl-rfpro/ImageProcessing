import os
import shutil
from tqdm import tqdm
import subprocess

# This sets OpenCV to import using the .EXR opener. This only works for .EXR files. If you wish to use a different
# file type, such as .png or .jpg, please comment out the following line.
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import cv2


def copy_bmps(src: str, dst: str):
    # Copies all .bmp files from the source to the destination. Interval of 5s
    size = 0
    print(f'Moving from {src}...')
    for file in tqdm(os.listdir(src)):
        if file.endswith('_1.bmp') or file.endswith('5-00_Camera14_#0907.png') or file.endswith(
                '0-00_Camera14_#0907.png'):
            pathname = os.path.join(src, file)
            if os.path.isfile(pathname):
                if not os.path.isfile(dst + '/' + pathname):
                    shutil.copy2(pathname, dst)

                    size += os.stat(pathname).st_size / 8e+6
    print(f'Total size: {size} MB')
    print(f'Moved to {dst}...')


def copy_txts(src: str, dst: str):
    # Copies all .txt files from the source to the destination. Interval of 5s
    size = 0
    print(f'Moving from {src}...')
    for file in tqdm(os.listdir(src)):
        if file.endswith('5_IMX728_RCCB_1_RGB_Mean') or file.endswith('0_IMX728_RCCB_1_RGB_Mean'):
            pathname = os.path.join(src, file)
            if os.path.isfile(pathname):
                if not os.path.isfile(dst + '/' + pathname):
                    shutil.copy2(pathname, dst)

                    size += os.stat(pathname).st_size / 8e+6
    print(f'Total size: {size} MB')
    print(f'Moved to {dst}...')


def copy_exrs(src: str, dst: str):
    """Copies all .exrs from src to dst"""
    size = 0
    print(f'Moving from {src}...')
    for file in tqdm(os.listdir(src)):
        if file.endswith('image1.exr'):
            pathname = os.path.join(src, file)
            if os.path.isfile(pathname):
                if not os.path.isfile(dst + '/' + pathname):
                    shutil.copy2(pathname, dst)

                    size += os.stat(pathname).st_size / 8e+6
    print(f'Total size: {size} MB')
    print(f'Moved to {dst}...')


def raw2exr(folder_path, exe_path):
    """Script which converts all .raw camera files in a folder to .exr files using a precompiled .exe."""
    print(f'Converting files in {folder_path} to .exr...')
    skipped = 0
    done = 0
    for fname in tqdm(os.listdir(folder_path)):
        exr_name = os.path.join(folder_path, fname).replace('.raw', '.exr')
        if os.path.exists(exr_name):
            skipped += 1
            continue
        if fname.endswith("image1.raw"):
            raw_file_path = os.path.join(folder_path, fname)
            subprocess.run([exe_path, raw_file_path])
            done += 1

    print(f"Process complete. Converted {done} files, skipped {skipped} files.")


def remove_exrs(folder_path):
    """Remove all exrs from a folder (in case of erroneous processing)."""
    removed = 0
    for fname in tqdm(os.listdir(folder_path)):
        if fname.endswith(".exr"):
            file_path = os.path.join(folder_path, fname)
            os.remove(file_path)
            removed += 1

    print(f"Process complete. Removed {removed} .exr files.")


def exr2png(folder_path):
    """Convert all .exr files in a folder to .png files"""
    for filename in os.listdir(folder_path):
        if filename.endswith('.exr'):
            exr_path = os.path.join(folder_path, filename)
            png_path = os.path.join(folder_path, filename.replace('.exr', '.png'))

            # Load the .exr image
            exr_image = cv2.imread(exr_path, cv2.IMREAD_UNCHANGED)

            # Convert the image to 8-bit (if necessary)
            png_image = cv2.normalize(exr_image, None, 0, 255, cv2.NORM_MINMAX)
            png_image = cv2.convertScaleAbs(png_image)

            # Save the image as .png
            cv2.imwrite(png_path, png_image)
            print(f"Converted {exr_path} to {png_path}")
