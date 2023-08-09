import csv
import os
from collections import deque
from typing import List, Optional, Union, Dict, Tuple, Any
import json
import argparse
from pathlib import Path

import cv2
from scipy import ndimage
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
import crops
import pandas as pd

DEFAULT_BASE_DIR: str = 'leftImg8bit_trainvaltest/leftImg8bit/train'
CVS_PATH_FILE: str = "data/attention_results/attention_results.csv"

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
# HSV color range for green
lower_green = np.array([40, 80, 80])
upper_green = np.array([75, 255, 255])

# HSV color range for red
lower_red = np.array([0, 50, 50])
upper_red = np.array([10, 255, 255])

# High-red: Hue values from around 170 to 180
lower_red2 = np.array([170, lower_red[1], lower_red[2]])
upper_red2 = np.array([180, upper_red[1], upper_red[2]])

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

GREEN_IMAGE_KERNEL = 'kernels/green_kernel.jpg'
RED_IMAGE_KERNEL = 'kernels/red_kernel.jpg'

THRESHOLD = 126

RED = 0
GREEN = 1
SCALES = [1.0, 0.75, 0.5, 0.25, 0.125, 0.0625]


def extract_color_from_image(image: np.array, color: int) -> np.array:
    return image[:, :, color].copy()


def extract_kernel(kernel_image_path: str, color: int) -> np.array:
    kernel_arr = np.array(Image.open(kernel_image_path))
    kernel_arr = extract_color_from_image(kernel_arr, color)

    # convert to float and normalize
    kernel_arr = kernel_arr.astype(float)
    kernel_arr -= np.mean(kernel_arr)  # normalize
    # print(kernel_arr)
    return kernel_arr


def find_tfl_lights(c_image: np.ndarray,
                    **kwargs) -> tuple[list[Any], list[Any], list[Any], list[Any], list[Any], list[Any]]:
    # Accumulators for x and y coordinates of detected green lights
    green_x, green_y = [], []
    red_x, red_y = [], []
    red_zoom, green_zoom = [], []
    c_image = cv2.cvtColor(c_image, cv2.COLOR_BGR2RGB)
    for scale in SCALES:
        # Resize the image according to the scale
        resized = cv2.resize(c_image, (0, 0), fx=scale, fy=scale)
        # Convert to HSV
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)

        # detecting green lights
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        res = cv2.bitwise_and(resized, resized, mask=mask_green)
        # detecting red lights
        mask_red = cv2.inRange(hsv, lower_red, upper_red)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask_red = mask_red + mask_red2
        res2 = cv2.bitwise_and(resized, resized, mask=mask_red)

        # Convert back to BGR and then to grayscale
        cimg = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        cimg2 = cv2.cvtColor(res2, cv2.COLOR_BGR2GRAY)

        # Detect green lights and accumulate their coordinates
        x, y = calc_max_suppression(cimg)
        green_x.extend(np.array(x) / scale)  # Adjust coordinates according to the scale
        green_y.extend(np.array(y) / scale)
        green_zoom.extend(np.array(scale * np.ones(len(x))))

        # Detect red lights and accumulate their coordinates
        x, y = calc_max_suppression(cimg2)
        red_x.extend(np.array(x) / scale)  # Adjust coordinates according to the scale
        red_y.extend(np.array(y) / scale)
        red_zoom.extend(np.array(scale * np.ones(len(x))))

    return red_x, red_y, red_zoom, green_x, green_y, green_zoom


def calc_max_suppression(image: np.array, threshold: int = 200) -> object:
    max_filtered = ndimage.maximum_filter(image, size=30)

    mask = np.equal(max_filtered, image)

    img2 = np.uint8(mask) * image
    y, x = np.where(mask & (img2 >= threshold))

    # cluster_points(points, 20)
    # x, y = cluster_points(x.tolist(), y.tolist())

    return x.tolist(), y.tolist()


def filter_color(color: int, image: np.ndarray, local_max_list: list) -> list:
    filter_color_list = []

    for local_max in local_max_list:
        r, g, b = image[local_max[0], local_max[1]]  # get the RGB values channels
        if (color == RED and r > 127 and g < r and b < r) or \
                (color == GREEN and g > 166 and r < g and b < g):
            filter_color_list.append([local_max[0], local_max[1]])

    return filter_color_list


def get_local_list(image: np.ndarray, locale_max_list) -> List[Tuple[int, int]]:
    """ filter the local max that are above the threshold
    :param image: np.ndarray of the image
    :param locale_max_list: list of the local max
    :return: list of the local max that are above the threshold
    """
    return list(filter(lambda x: (image[x[0], x[1]] > THRESHOLD), locale_max_list))


def get_local_max_list(image: np.ndarray) -> List[Tuple[int, int]]:
    """ get the local maximum of the image
    :param image: np.ndarray of the image
    :return: list of the local max
    """
    local_max_list = peak_local_max(image, min_distance=80)
    local_max_list = get_local_list(image, local_max_list)
    return local_max_list


def filter_by_color(c_image_arr: np.array, c_image: np.ndarray, kernel_image_path: str, color: int):
    """ filter the image by color
    :param c_image_arr: np.array of the image (data)
    :param c_image: np.ndarray of the image
    :param kernel_image_path: path to the kernel image
    :param color: the color to filter by
    :return: the normalized image and the filtered color
    """
    image_after_extracting_color = extract_color_from_image(c_image_arr, color)
    kernel_image = extract_kernel(kernel_image_path, color)
    print(image_after_extracting_color)
    print("kernel_image, ", kernel_image)
    image = ndimage.convolve(image_after_extracting_color.astype(float), kernel_image[::-1, ::-1])
    normalized_image = image / np.linalg.norm(image)
    local_max_list = get_local_max_list(image)
    filtered_color = filter_color(color, c_image, local_max_list)
    return normalized_image, filtered_color


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

    # show_image_and_gt(c_image, objects, fig_num)

    red_x, red_y, red_zoom, green_x, green_y, green_zoom = find_tfl_lights(c_image)

    red_coordinates = list(zip(red_x, red_y))
    green_coordinates = list(zip(green_x, green_y))

    group_red = group_coordinates_by_order(red_coordinates)
    group_green = group_coordinates_by_order(green_coordinates)

    first_point_form_every_red = [group[0] for group in group_red]
    first_point_form_every_green = [group[0] for group in group_green]

    red_x = [point[0] for point in first_point_form_every_red]
    red_y = [point[1] for point in first_point_form_every_red]
    green_x = [point[0] for point in first_point_form_every_green]
    green_y = [point[1] for point in first_point_form_every_green]

    red_zoom = np.array(np.ones(len(first_point_form_every_red)))
    green_zoom = np.array(np.ones(len(first_point_form_every_green)))

    if csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        for x, y, zoom in zip(red_x, red_y, red_zoom):
            writer.writerow({'path': image_path, 'x': x, 'y': y, 'zoom': zoom, 'col': 'red'})

        for x, y, zoom in zip(green_x, green_y, green_zoom):
            writer.writerow({'path': image_path, 'x': x, 'y': y, 'zoom': zoom, 'col': 'green'})

    # plt.imshow(c_image)
    # plt.plot(red_x, red_y, 'ro', markersize=4)
    # plt.plot(green_x, green_y, 'go', markersize=4)


# GIVEN CODE TO TEST YOUR IMPLEMENTATION AND PLOT THE PICTURES
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
            # plt.plot(x_coordinates, y_coordinates, color, label=image_object['label'])
            labels.add(image_object['label'])
        if 1 < len(labels):
            # The legend provides a visual representation of the labels associated with the plotted objects.
            # It helps in distinguishing different objects in the plot based on their labels.
            plt.legend()

    # plt.show()


# def find_tfl_lights(c_image: np.ndarray,
#                     **kwargs) -> Tuple[List[int], List[int], List[float], List[int], List[int], List[float]]:
#     # HSV color range for green
#     lower_green = np.array([40, 80, 80])
#     upper_green = np.array([75, 255, 255])
#
#     # Accumulators for x and y coordinates of detected green lights
#     green_x, green_y, green_zoom = [], [], []
#
#     # Create an image pyramid with scales 1.0 (original), 0.75, 0.5, 0.25
#     scales = [1.0, 0.75, 0.5, 0.25]


# def calc_max_suppression(image: np.array, threshold: int = 200) -> object:
#     max_filtered = ndimage.maximum_filter(image, size=30)
#
#     mask = np.equal(max_filtered, image)
#
#     img2 = np.uint8(mask) * image
#     y, x = np.where(mask & (img2 >= threshold))
#
#     # cluster_points(points, 20)
#     # x, y = cluster_points(x.tolist(), y.tolist())
#
#     return x.tolist(), y.tolist()


def group_coordinates_by_order(coordinates, tolerance=30):
    # Create a queue to hold the coordinates to be processed
    queue = deque(coordinates)

    # Initialize the groups list
    grouped_lists = []

    while queue:
        # Start a new group with the first coordinate in the queue
        current_group = [queue.popleft()]

        # List to store the indices of coordinates to be removed
        indices_to_remove = []

        # Process all remaining coordinates in the queue
        for i in range(len(queue)):
            x_diff = abs(current_group[-1][0] - queue[i][0])
            y_diff = abs(current_group[-1][1] - queue[i][1])

            if x_diff <= tolerance and y_diff <= tolerance:
                # Add the coordinate to the current group and mark it for removal
                current_group.append(queue[i])
                indices_to_remove.append(i)

        # Remove marked coordinates from the queue in reverse order to avoid index shifting
        for index in reversed(indices_to_remove):
            del queue[index]

        # Add the current group to the grouped_lists
        grouped_lists.append(current_group)

    # Filter out groups with size 1
    # grouped_lists = [group for group in grouped_lists if len(group) > 2]

    return grouped_lists


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

    # If you entered a custom dir to run from or the default dir exists in your project, then:
    directory_path: Path = Path(args.dir or DEFAULT_BASE_DIR)
    if directory_path.exists():
        file_list: List[Path] = []
        for subdirectory in directory_path.iterdir():
            if subdirectory.is_dir():
                # Get a list of image files from each subdirectory
                image_files = subdirectory.glob('*_leftImg8bit.png')
                file_list.extend(image_files)


        with open(CVS_PATH_FILE, 'w', newline='') as csvfile:
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

    # open the attention results csv file with pandas
    df = pd.read_csv(CVS_PATH_FILE)
    crops.create_crops(df)

    plt.show(block=True)


if __name__ == '__main__':
    main()
