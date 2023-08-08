from typing import List, Optional, Union, Dict, Tuple
import json
import argparse
from pathlib import Path

import cv2
import csv
from numpy import uint8
from scipy import ndimage, signal

import numpy as np
from scipy import signal as sg
from scipy.ndimage import maximum_filter
from PIL import Image, ImageFilter, ImageEnhance
import matplotlib.pyplot as plt

# if you wanna iterate over multiple files and json, the default source folder name is this.
DEFAULT_BASE_DIR: str = 'INSERT_YOUR_DIR_WITH_PNG_AND_JSON_HERE'

# The label we wanna look for in the polygons json file
TFL_LABEL = ['traffic light']
fieldnames = ['path', 'x', 'y', 'zoom', 'col']
POLYGON_OBJECT = Dict[str, Union[str, List[int]]]
X_COORDINATES = List[int]
Y_COORDINATES = List[int]
RED_X_COORDINATES = List[int]
RED_Y_COORDINATES = List[int]
GREEN_X_COORDINATES = List[int]
GREEN_Y_COORDINATES = List[int]

high_pass_kernel = np.array([[-1, -1, -1],
                             [-1, 8, -1],
                             [-1, -1, -1]])

low_pass_kernel = np.array([[1 / 9, 1 / 9, 1 / 9],
                            [1 / 9, 1 / 9, 1 / 9],
                            [1 / 9, 1 / 9, 1 / 9]])

high_pass_kernel2 = [[-1, -1, -1, -1, -1],
                     [-1, 1, 2, 1, -1],
                     [-1, 2, 4, 2, -1],
                     [-1, 1, 2, 1, -1],
                     [-1, -1, -1, -1, -1]]

# high_pass_kernel3 =[[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
#                     [-1,  1,  1,  1,  1,  1,  1,  1,  1, -1],
#                     [-1,  1,  1,  1,  1,  1,  1,  1,  1, -1],
#                     [-1,  1,  1,  1,  1,  1,  1,  1,  1,  -1],
#                     [-1,  1,  2,  1, -1],
#                     [-1,  1,  2,  1, -1],
#                     [-1,  1,  2,  1, -1],
#                     [-1, -1, -1, -1, -1]]

huge_hp_kernel = np.array([[-3, -3, -3, -3, -3, -3, -3, -3, -3, -3],
                           [-3, 1, 1, 1, 1, 1, 1, 1, 1, -3],
                           [-3, 1, 1, 2, 2, 2, 2, 1, 1, -3],
                           [-3, 1, 2, 2, 3, 3, 2, 2, 1, -3],
                           [-3, 1, 2, 3, 4, 4, 3, 2, 1, -3],
                           [-3, 1, 2, 3, 4, 4, 3, 2, 1, -3],
                           [-3, 1, 2, 2, 3, 3, 2, 2, 1, -3],
                           [-3, 1, 1, 2, 2, 2, 2, 1, 1, -3],
                           [-3, 1, 1, 1, 1, 1, 1, 1, 1, -3],
                           [-3, -3, -3, -3, -3, -3, -3, -3, -3, -3]])

huge_hp_kernel2 = np.array([[0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                            [0, 0, 1, 2, 2, 2, 2, 1, 0, 0],
                            [1, 1, 2, 2, 3, 3, 2, 2, 1, 1],
                            [1, 2, 2, -5, -5, -5, -5, 2, 2, 1],
                            [1, 2, 3, -5, -9, -9, -5, 3, 2, 1],
                            [1, 2, 3, -5, -9, -9, -5, 3, 2, 1],
                            [1, 2, 2, -5, -5, -5, -5, 2, 2, 1],
                            [1, 1, 2, 2, 3, 3, 2, 2, 1, 1],
                            [0, 0, 1, 2, 2, 2, 2, 1, 0, 0],
                            [0, 0, 1, 1, 1, 1, 1, 1, 0, 0]])


def plot(data, title):
    plot.i += 1
    plt.subplot(2, 2, plot.i)
    plt.imshow(data)
    plt.gray()
    plt.title(title)


def filter_image(im: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Apply a high pass filter on the image.

    :param kernel:
    :param im: The image itself as np.uint8, shape of (H, W, 3).
    :return: The filtered image.
    """

    high_pass_image = np.zeros_like(im)
    for channel in range(im.shape[2]):
        channel_filtered_image = signal.convolve2d(im[:, :, channel], kernel, mode='same', boundary='symm')

        # Ensure the image is in an 8-bit range
        channel_filtered_image = np.clip(channel_filtered_image, 0, 255).astype('uint8')
        high_pass_image[:, :, channel] = channel_filtered_image

    return high_pass_image


def inhance_image_without_affecting_brightess_and_constract(image: np.ndarray) -> np.ndarray:
    enhancer = ImageEnhance.Color(Image.fromarray(image))
    enhanced_image = enhancer.enhance(1.5)
    enhanced_image_array = np.array(enhanced_image)
    return enhanced_image_array


def cluster_points(x: List, y: List) -> Tuple[X_COORDINATES, RED_Y_COORDINATES]:
    return x, y


def calc_max_suppression(image: np.array, threshold: int = 200) -> object:
    max_filtered = ndimage.maximum_filter(image, size=30)

    mask = np.equal(max_filtered, image)

    img2 = np.uint8(mask) * image
    y, x = np.where(mask & (img2 >= threshold))

    # cluster_points(points, 20)
    # x, y = cluster_points(x.tolist(), y.tolist())

    return x.tolist(), y.tolist()




def find_tfl_lights(c_image: np.ndarray,
                    **kwargs) -> Tuple[List[int], List[int], List[float], List[int], List[int], List[float]]:
    # HSV color range for green
    lower_green = np.array([40, 80, 80])
    upper_green = np.array([75, 255, 255])

    # Accumulators for x and y coordinates of detected green lights
    green_x, green_y, green_zoom = [], [], []

    # Create an image pyramid with scales 1.0 (original), 0.75, 0.5, 0.25
    scales = [1.0, 0.75, 0.5, 0.25]

    for scale in scales:
        # Resize the image according to the scale
        resized = cv2.resize(c_image, (0, 0), fx=scale, fy=scale)

        # Convert to HSV and detect green colors
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_green, upper_green)
        res = cv2.bitwise_and(resized, resized, mask=mask)

        # Convert back to BGR and then to grayscale
        ccimg = cv2.cvtColor(res, cv2.COLOR_HSV2BGR)
        cimg = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

        # Detect green lights and accumulate their coordinates
        x, y = calc_max_suppression(cimg)
        green_x.extend(np.array(x) / scale)  # Adjust coordinates according to the scale
        green_y.extend(np.array(y) / scale)  # Adjust for the cropping
        green_zoom.extend([scale] * len(x))  # Add the scale for each detected light

    return [500, 700, 900], [500, 550, 600], [1, 1, 1], green_x, green_y, green_zoom


### GIVEN CODE TO TEST YOUR IMPLENTATION AND PLOT THE PICTURES
def show_image_and_gt(c_image: np.ndarray, objects: Optional[List[POLYGON_OBJECT]], fig_num: int = None):
    # ensure a fresh canvas for plotting the image and objects.
    plt.figure(fig_num).clf()
    # displays the input image.
    plt.imshow(c_image)
    labels = set()
    if objects:
        for image_object in objects:
            # Extract the 'polygon' array from the image object
            poly: np.array = np.array(image_object['polygon'])
            # Use advanced indexing to create a closed polygon array
            # The modulo operation ensures that the array is indexed circularly, closing the polygon
            polygon_array = poly[np.arange(len(poly)) % len(poly)]
            # gets the x coordinates (first column -> 0) anf y coordinates (second column -> 1)
            x_coordinates, y_coordinates = polygon_array[:, 0], polygon_array[:, 1]
            color = 'r'
            plt.plot(x_coordinates, y_coordinates, color, label=image_object['label'])
            labels.add(image_object['label'])
        if 1 < len(labels):
            # The legend provides a visual representation of the labels associated with the plotted objects.
            # It helps in distinguishing different objects in the plot based on their labels.
            plt.legend()

    plt.show()


def save_to_csv(data: List[Dict[str, Union[str, int, float]]], filename: str):
    # Get a list of all keys in the dictionaries
    fieldnames = set().union(*[row.keys() for row in data])

    with open(filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for row in data:
            writer.writerow(row)


def test_find_tfl_lights(image_path: str, image_json_path: Optional[str] = None, csvfile=None, fig_num=None):
    """
    Run the attention code.
    """
    # using pillow to load the image
    image: Image = Image.open(image_path)
    # converting the image to a numpy ndarray array
    c_image: np.ndarray = np.array(image)

    objects = None
    if image_json_path:
        image_json = json.load(Path(image_json_path).open())
        objects: List[POLYGON_OBJECT] = [image_object for image_object in image_json['objects']
                                         if image_object['label'] in TFL_LABEL]

    show_image_and_gt(c_image, objects, fig_num)

    red_x, red_y, red_zoom, green_x, green_y, green_zoom = find_tfl_lights(c_image)

    if csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        for x, y, zoom in zip(red_x, red_y, red_zoom):
            writer.writerow({'path': image_path, 'x': x, 'y': y, 'zoom': zoom, 'col': 'red'})

        for x, y, zoom in zip(green_x, green_y, green_zoom):
            writer.writerow({'path': image_path, 'x': x, 'y': y, 'zoom': zoom, 'col': 'green'})

    # 'ro': This specifies the format string. 'r' represents the color red, and 'o' represents circles as markers.
    plt.imshow(c_image)
    plt.plot(red_x, red_y, 'ro', markersize=4)
    plt.plot(green_x, green_y, 'go', markersize=4)


def main(argv=None):
    """
    It's nice to have a standalone tester for the algorithm.
    Consider looping over some images from here, so you can manually examine the results.
    Keep this functionality even after you have all system running, because you sometime want to debug/improve a module.

    :param argv: In case you want to programmatically run this.
    """

    parser = argparse.ArgumentParser("Test TFL attention mechanism")
    parser.add_argument('-i', '--image', type=str, help='Path to an image')
    parser.add_argument("-j", "--json", type=str, help="Path to image json file -> GT for comparison")
    parser.add_argument('-d', '--dir', type=str, help='Directory to scan images in')
    args = parser.parse_args(argv)

    # If you entered a custom dir to run from or the default dir exist in your project then:
    directory_path: Path = Path(args.dir or DEFAULT_BASE_DIR)
    if directory_path.exists():
        # gets a list of all the files in the directory that ends with "_leftImg8bit.png".
        file_list: List[Path] = list(directory_path.glob('*_leftImg8bit.png'))

        with open("attention_results.csv", 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for image in file_list:
                # Convert the Path object to a string using as_posix() method
                image_path: str = image.as_posix()
                path: Optional[str] = image_path.replace('_leftImg8bit.png', '_gtFine_polygons.json')
                image_json_path: Optional[str] = path if Path(path).exists() else None
                test_find_tfl_lights(image_path, image_json_path, csvfile)

    if args.image and args.json:
        test_find_tfl_lights(args.image, args.json)
    elif args.image:
        test_find_tfl_lights(args.image)
    plt.show(block=True)


if __name__ == '__main__':
    main()
