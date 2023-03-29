#Copyright 2015 Yale University - Grablab
#Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:\
#The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# Modified to work with Python 3 by Sebastian Castro, 2020

import os
import json
import tarfile
import io
from urllib.request import Request, urlopen
from PIL import Image

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


def download_file(url, file):
    """ Downloads files from a given URL """
    u = urlopen(url)
    file_size = int(u.getheader("Content-Length"))    

    file_size_dl = 0
    block_sz = 65536
    while True:
        buffer = u.read(block_sz)
        if not buffer:
            break

        file_size_dl += len(buffer)
        file.write(buffer)
        status = r"%10d  [%3.2f%%]" % (file_size_dl/1000000.0, file_size_dl * 100. / file_size)
        status = status + chr(8)*(len(status)+1)
        print(status)
    

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
        return True
    except Exception as e:
        return False



def get_bounding_box(filename):
    with Image.open(filename) as mask:
        # Convert to binary format
        binary = mask.convert('1')

        # Find bounding box coordinates
        return binary.getbbox()

def scale_x(x):
    return x * 640 / 4272

def scale_y(y):
    return y * (640 * (2848 / 4272))

def lazy_image_collection():

    # Grab all the object information
    objects = fetch_objects(objects_url)

    # Download each object for all objects and types specified
    for object in objects:
        for file_type in files_to_download:
            temp_file = io.BytesIO()
            url = tgz_url(object, file_type)
            if not check_url(url):
                continue
            download_file(url, temp_file)

            with tarfile.open(fileobj=temp_file, mode="r:*") as t:

                for tf in t.getmembers():
                    if tf.isfile() and (not '/' in tf.name) and tf.name.endswith(".jpg"):
                        # Valid photo

                        yield (tuple(object, *get_bounding_box(t.extractfile(t.getmember(os.path.join('masks', os.path.splitext(tf.name)[0] + '_mask.pbm'))).read())), t.extractfile(tf).read())

def save_yolo_annotation(image_filename, annotations):
    # annotations is a list of ((class_name, bbox_left, bbox_upper, bbox_right, bbox_lower), image_bytes) tuples
    
    # Create the YOLO annotation filename and the image filename
    annotation_filename = os.path.splitext(image_filename)[0] + ".txt"
    image_output_filename = os.path.splitext(image_filename)[0] + ".jpg"
    
    with open(annotation_filename, "w") as f:
        (class_name, bbox_left, bbox_upper, bbox_right, bbox_lower), image_bytes = annotation
        
        # Resize the image to 640x426
        image = Image.open(image_bytes)
        image = image.resize((640, 426))
        
        # Convert the bounding box coordinates to the resized image size
        old_image_width, old_image_height = image.size
        new_image_width, new_image_height = 640, 426
        bbox_left = bbox_left * new_image_width / old_image_width
        bbox_upper = bbox_upper * new_image_height / old_image_height
        bbox_right = bbox_right * new_image_width / old_image_width
        bbox_lower = bbox_lower * new_image_height / old_image_height
        
        # Convert the bounding box coordinates to YOLO format (normalized between 0 and 1)
        x_center = (bbox_left + bbox_right) / 2 / new_image_width
        y_center = (bbox_upper + bbox_lower) / 2 / new_image_height
        width = (bbox_right - bbox_left) / new_image_width
        height = (bbox_lower - bbox_upper) / new_image_height
        
        # Write the YOLO annotation line
        line = "{} {} {} {} {}".format(class_name, x_center, y_center, width, height)
        f.write(line + "\n")
    
    # Save the resized image to disk
    image.save(image_output_filename, format='JPEG', quality=95)

if __name__=="__main__":
    for i, j in lazy_image_collection():
        save_yolo_annotation(f"./yolo/image_{i}", j)
        
        