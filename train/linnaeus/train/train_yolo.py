#Copyright 2015 Yale University - Grablab
#Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:\
#The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# Modified to work with Python 3 by Sebastian Castro, 2020

import numpy as np
import json
import tarfile
import tempfile
import cv2
import pathlib
import random
from tqdm import tqdm
from urllib.request import Request, urlopen

# You can edit this list to only download certain kinds of files.
# 'berkeley_rgbd' contains all of the depth maps and images from the Carmines.
# 'berkeley_rgb_highres' contains all of the high-res images from the Canon cameras.
# 'berkeley_processed' contains all of the segmented point clouds and textured meshes.
# 'google_16k' contains google meshes with 16k vertices.
# 'google_64k' contains google meshes with 64k vertices.
# 'google_512k' contains google meshes with 512k vertices.
# See the website for more details.
#files_to_download = ["berkeley_rgbd", "berkeley_rgb_highres", "berkeley_processed", "google_16k", "google_64k", "google_512k"]
files_to_download = ["berkeley_rgb_highres"]

# objects_to_download = "all"
objects_to_download = ["001_chips_can"]
#                        "002_master_chef_can",
#                        "003_cracker_box",
#                        "004_sugar_box"]

# Extract all files from the downloaded .tgz, and remove .tgz files.
# If false, will just download all .tgz files to output_directory
extract = True

base_url = "http://ycb-benchmarks.s3-website-us-east-1.amazonaws.com/data/"
objects_url = "https://ycb-benchmarks.s3.amazonaws.com/data/objects.json"


def fetch_objects(url):
    """ Fetches the object information before download """
    response = urlopen(url)
    html = response.read()
    objects = json.loads(html)
    return objects["objects"]


def download_file(url):
    """ Downloads files from a given URL """

    file = tempfile.TemporaryFile(mode="w+b")
    u = urlopen(url)
    file_size = int(u.getheader("Content-Length"))    

    block_sz = 65536
    with tqdm(total=file_size) as pbar:
        while True:
            buffer = u.read(block_sz)
            if not buffer:
                break

            file.write(buffer)
            pbar.update(len(buffer))
    file.seek(0)

    return file

def tgz_url(object, type):
    """ Get the TGZ file URL for a particular object and dataset type """
    if type in ["berkeley_rgbd", "berkeley_rgb_highres"]:
        return base_url + "berkeley/{object}/{object}_{type}.tgz".format(object=object,type=type)
    elif type in ["berkeley_processed"]:
        return base_url + "berkeley/{object}/{object}_berkeley_meshes.tgz".format(object=object,type=type)
    else:
        return base_url + "google/{object}_{type}.tgz".format(object=object,type=type)

def check_url(url):
    """ Check the validity of a URL """
    try:
        request = Request(url)
        request.get_method = lambda : 'HEAD'
        response = urlopen(request)
        return response
    except Exception as e:
        return False

def lazy_image_collection():

    # Grab all the object information
    objects = fetch_objects(objects_url)

    # Download each object for all objects and types specified
    for i, object in enumerate(objects):
        if objects_to_download != "all" and object not in objects_to_download:
            continue
        for file_type in files_to_download:
            url = tgz_url(object, file_type)
            if not check_url(url):
                continue
            print("Downloading... ({})".format(url))
            temp_file = download_file(url)

            with tarfile.open(fileobj=temp_file, mode="r:*") as t:
                print("Extracting... ({})".format(t.name))
                for tf in t.getmembers():
                    image_path = pathlib.Path(tf.name)
                    if tf.isfile() and image_path.suffix == '.jpg' and image_path.parent.name == object:
                        # Valid photo
                        mask_path = image_path.parent / 'masks' / (image_path.stem + '_mask.pbm')

                        print("Found image: {}".format(image_path))
                        print("Found mask: {}".format(mask_path))
                        
                        # Yield a tuple of the image file and the image mask
                        yield (i, t.extractfile(tf), t.extractfile(t.getmember(mask_path.as_posix())))
            temp_file.close()

def save_yolo(save_dir, filename, image_and_mask):
    classnumber, image, mask = image_and_mask

    image = np.ndarray(shape=(1, len(image)), dtype=np.uint8, buffer=image.read())
    mask = np.ndarray(shape=(1, len(mask)), dtype=np.uint8, buffer=mask.read())

    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    mask = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)

    # Resize the image to 640x640
    image = image.resize((640, 640))

    # Gather the polygons
    polygons = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create the YOLO annotation filename and the image filename
    image_filename = save_dir / "images" / f"{filename}.jpg"
    annotation_filename = save_dir / "labels" / f"{filename}.txt"
    
    # write the polygons to the annotation file
    with open(annotation_filename, 'w') as annotation_file:
        for polygon in polygons:
            polygon_point_str = " ".join(" ".join(point) for point in polygon)
            annotation_file.write(f"{classnumber} {polygon_point_str}\n")
    
    # Save the image
    image.save(image_filename, format='JPEG', quality=95)

TEST = 0.2
TRAIN = 0.6

if __name__=="__main__":
    path = pathlib.Path("YCB-dataset")

    train_path = path / "train"
    test_path = path / "test"
    validation_path = path / "validation"

    train_path.mkdir(exist_ok=True, parents=True)
    test_path.mkdir(exist_ok=True, parents=True)
    validation_path.mkdir(exist_ok=True, parents=True)

    print(train_path)
    print(test_path)
    print(validation_path)

    for index, image_and_mask in enumerate(lazy_image_collection()):
        choice_machine = random.random()

        if choice_machine < TRAIN:
            save_yolo(train_path, f"img{index}", image_and_mask)
        elif choice_machine < TRAIN + TEST:
            save_yolo(test_path, f"img{index}", image_and_mask)
        else:
            save_yolo(validation_path, f"img{index}", image_and_mask)