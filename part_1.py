from typing import List, Optional, Union, Dict, Tuple
import json
import argparse
from pathlib import Path

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

X_COORDINATES = List[int]
Y_COORDINATES = List[int]
def cluster_points(x: List, y: List) -> Tuple[X_COORDINATES, RED_Y_COORDINATES]:





    return x,y







def calc_max_suppression(image: np.array, threshold: int = 120) -> object:
    max_filtered = ndimage.maximum_filter(image, size=30)
    # Create a binary mask by comparing the filtered image to the original grayscale image
    mask = np.equal(max_filtered, image)
    # Apply the mask to the original image to get the final result
    img2 = np.uint8(mask) * image
    y, x = np.where(mask & (img2 >= threshold))

    # cluster_points(points, 20)
    # x, y = cluster_points(x.tolist(), y.tolist())

    return x.tolist(), y.tolist()


def find_tfl_lights(c_image: np.ndarray,
                    **kwargs) -> Tuple[RED_X_COORDINATES, RED_Y_COORDINATES, GREEN_X_COORDINATES, GREEN_Y_COORDINATES]:
    """
    Detect candidates for TFL lights. Use c_image, kwargs and you imagination to implement.

    :param c_image: The image itself as np.uint8, shape of (H, W, 3).
    :param kwargs: Whatever config you want to pass in here.
    :return: 4-tuple of x_red, y_red, x_green, y_green.
    """
    ### WRITE YOUR CODE HERE ###
    ### USE HELPER FUNCTIONS ###

    c_image = inhance_image_without_affecting_brightess_and_constract(c_image)
    c_image = filter_image(c_image, low_pass_kernel)
    c_image = filter_image(c_image, high_pass_kernel)
    red_x, red_y = calc_max_suppression(c_image[:, :, 0])
    green_x, green_y = calc_max_suppression(c_image[:, :, 2])

    return red_x, red_y, green_x, green_y
    # return [500, 700, 900], [500, 550, 600], [600, 800], [400, 300]


#

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


def test_find_tfl_lights(image_path: str, image_json_path: Optional[str] = None, fig_num=None):
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

    red_x, red_y, green_x, green_y = find_tfl_lights(c_image)
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

        for image in file_list:
            # Convert the Path object to a string using as_posix() method
            image_path: str = image.as_posix()
            path: Optional[str] = image_path.replace('_leftImg8bit.png', '_gtFine_polygons.json')
            image_json_path: Optional[str] = path if Path(path).exists() else None
            test_find_tfl_lights(image_path, image_json_path)

    if args.image and args.json:
        test_find_tfl_lights(args.image, args.json)
    elif args.image:
        test_find_tfl_lights(args.image)
    plt.show(block=True)


if __name__ == '__main__':
    main()
