import json
import os
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import matplotlib.pyplot as plt
from PIL import Image
import math

SEQ: str = 'seq'  # The image seq number -> for tracing back the original image
IS_TRUE: str = 'is_true'  # Is it a traffic light or not.
IGNOR: str = 'is_ignore'  # If it's an unusual crop (like two tfl's or only half etc.) that you can just ignor it and
# investigate the reason after
CROP_PATH: str = 'data/crops/'
X0: str = 'x0'  # The bigger x value (the right corner)
X1: str = 'x1'  # The smaller x value (the left corner)
Y0: str = 'y0'  # The smaller y value (the lower corner)
Y1: str = 'y1'  # The bigger y value (the higher corner)
COL: str = 'col'
SEQ_IMAG: str = 'seq_imag'  # Serial number of the image
GTIM_PATH: str = 'gtim_path'
X: str = 'x'
Y: str = 'y'
COLOR: str = 'color'
ZOOM: str = 'zoom'
PATH: str = 'path'
CROP_RESULT: List[str] = [SEQ, IS_TRUE, IGNOR, CROP_PATH, X0, X1, Y0, Y1, COL]

# Files path
BASE_SNC_DIR: Path = Path.cwd()
DATA_DIR: Path = (BASE_SNC_DIR / 'data')
CROP_DIR: Path = DATA_DIR / 'crops'
ATTENTION_PATH: Path = DATA_DIR / 'attention_results'

CROP_CSV_NAME: str = 'crop_results.csv'  # result CSV name


def make_crop(df_x, df_y, row_zoom, row_color):
    """
    The function that creates the crops from the image.
    Your return values from here should be the coordinates of the crops in this format (x0, x1, y0, y1, crop content):
    'x0'  The bigger x value (the right corner)
    'x1'  The smaller x value (the left corner)
    'y0'  The smaller y value (the lower corner)
    'y1'  The bigger y value (the higher corner)
    """
    if pd.isna(df_x) or pd.isna(df_y) or pd.isna(row_zoom) or pd.isna(row_color):
        return -1, -1, -1, -1
    width = 10 * row_zoom
    height = 15 * row_zoom
    if row_color == 'red':
        new_x = df_x - 5
        new_y = df_y - height + 5 if df_y >= height else 0
    elif row_color == 'green':
        new_x = df_x - 5
        new_y = df_y - height if df_y >= height else 0
    else:
        return -1, -1, -1, -1

    return new_x - width, new_x + 2 * width + 10, new_y - 10, new_y + height


def check_crop(polygon: List, image_path: str):
    """
    Here you check if your crop contains a traffic light or not.
    Try using the ground truth to do that (Hint: easier than you think for the simple cases, and if you found a hard
    one, just ignore it for now :). )
    """

    cropped_image = Polygon(
        [(polygon[0], polygon[2]), (polygon[0], polygon[3]), (polygon[1], polygon[3]), (polygon[1], polygon[2])])
    image_path = image_path.split('_')[0] + '_' + image_path.split('_')[1] + '_' + \
           image_path.split('_')[2] + '_gtFine_polygons.json'

    image_path = os.path.join('data/attention_results', image_path)
    if not os.path.exists(image_path):
        return False, False

    traffic_light_polygons_json = get_traffic_light_polygons_from_json(image_path)

    traffic_light_contain = False
    for traffic_light in traffic_light_polygons_json:
        is_fully_contain = True
        for point in traffic_light:
            if not cropped_image.contains(Point(point)):
                is_fully_contain = False
                break
            traffic_light_contain = True

        if is_fully_contain:
            # print("found traffic light in crop")
            # x_coords, y_coords = zip(*traffic_light)
            # x_crop, y_crop = zip(*cropped_image.exterior.coords)
            # plt.plot(x_crop + (x_crop[0],), y_crop + (y_crop[0],), color='blue')
            # plt.plot(x_coords + (x_coords[0],), y_coords + (y_coords[0],), color='red')
            # plt.show()
            return True, False

    if not traffic_light_contain:
        return True, True

    return False, False


def get_traffic_light_polygons_from_json(image_path):
    traffic_sign_polygons = []
    with open(image_path, 'r') as json_file:
        data = json.load(json_file)
        for obj in data['objects']:
            if obj['label'] == 'traffic light':
                polygon = obj['polygon']
                traffic_sign_polygons.append(polygon)
    return traffic_sign_polygons

def get_crop_filename(image_path: str, seq: int) -> str:
    """
    *** No need to touch this. ***
    Returns the crop filename according to the image path and the sequence number.
    """
    filename = image_path.split('/')[-1]
    filename = filename.split('.png')[0]
    return f"{filename}_{seq}.png"

def crop_image(path: str, polygon: List) -> Image:
    """
    Crops the image according to the coordinates and saves it in the
     relevant folder under the relevant name.
    """
    # Open the original image using PIL
    original_image = Image.open(path)

    # Crop the image using the calculated coordinates
    cropped_image = original_image.crop((polygon[0], polygon[2], polygon[1], polygon[3]))
    return cropped_image


def save_for_part_2(crops_df: pd.DataFrame):
    """
    *** No need to touch this. ***
    Saves the result DataFrame containing the crops data in the relevant folder under the relevant name for part 2.
    """
    if not ATTENTION_PATH.exists():
        print(f'Creating {ATTENTION_PATH}')
        ATTENTION_PATH.mkdir()
    crops_sorted: pd.DataFrame = crops_df.sort_values(by=SEQ)
    crops_sorted.to_csv(ATTENTION_PATH / CROP_CSV_NAME, index=False)


def create_crops(df: pd.DataFrame) -> pd.DataFrame:
    # Your goal in this part is to take the coordinates you have in the df, run on it, create crops from them, save them
    # in the 'data' folder, then check if crop you have found is correct (meaning the TFL is fully contained in the
    # crop) by comparing it to the ground truth and in the end right all the result data you have in the following
    # DataFrame (for doc about each field and its input, look at 'CROP_RESULT')
    #
    # *** IMPORTANT ***
    # All crops should be the same size or smaller!!!
    # Run this from your 'code' folder so that it will be in the right relative folder from your data folder.

    # creates a folder for you to save the crops in, recommended not must
    if not CROP_DIR.exists():
        CROP_DIR.mkdir()

    # For documentation about each key end what it means, click on 'CROP_RESULT' and see for each value what it means.
    # You wanna stick with this DataFrame structure because its output is the same as the input for the next stages.
    result_df = pd.DataFrame(columns=CROP_RESULT)

    # A dict containing the row you want to insert into the result DataFrame.
    result_template: Dict[Any] = {SEQ: '', IS_TRUE: '', IGNOR: '', CROP_PATH: '', X0: '', X1: '', Y0: '', Y1: '',
                                  COL: ''}
    seq = 0
    image_temp_seq = df.iloc[0][PATH].split('_')[2]

    for index, row in df.iterrows():
        image_seq = row[PATH].split('_')[2]
        if image_seq != image_temp_seq:
            seq = 0
            image_temp_seq = image_seq
        result_template[SEQ] = seq
        seq += 1

        result_template[COL] = row[COL]

        # example code:
        # ******* rewrite ONLY FROM HERE *******

        x0, x1, y0, y1 = make_crop(row[X], row[Y], row[ZOOM], row[COL])
        if x0 == -1 or x1 == -1 or y0 == -1 or y1 == -1:
            continue
        result_template[X0], result_template[X1], result_template[Y0], result_template[Y1] = x0, x1, y0, y1

        # change the crop_path to the path you want to save the crop in.
        result_template[CROP_PATH] = get_crop_filename(row[PATH], seq)

        # Crop the image using the calculated coordinates
        cropped_image = crop_image(row[PATH], [x0, x1, y0, y1])

        cropped_image.save(CROP_DIR / result_template[CROP_PATH])

        crop_polygon = [x0, x1, y0, y1]
        result_template[IS_TRUE], result_template[IGNOR] = check_crop(crop_polygon, row[PATH])
        # ******* TO HERE *******

        # added to current row to the result DataFrame that will serve you as the input to part 2 B).
        result_df = result_df._append(result_template, ignore_index=True)

    # A Short function to help you save the whole thing - your welcome ;)
    save_for_part_2(result_df)
    return result_df


if __name__ == '__main__':
    df = pd.read_csv('data/attention_results/attention_results.csv')
    create_crops(df)
