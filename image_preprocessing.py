import os
import math
import numpy as np
from imageio import imread, imwrite
from skimage import img_as_ubyte
from skimage.transform import resize
from tqdm import tqdm
import numpy as np

import boto3
import re
from get_config import get_config
from image_gather import PlayScraper

config = get_config()

def get_s3_keys(client, bucket=config["bucket"], teams=config["teams"], prefix=config["prefix"], file_type=config["file_type"]):
    """Get a list of keys in an S3 bucket."""
    keys = []
    for team in teams:
        prefix_ = prefix + team
        resp = client.list_objects_v2(Bucket=bucket, Prefix=prefix_)
        for obj in resp['Contents']:
            file = obj['Key']
            if re.findall("(?<=/)[^/]*[\.]"+file_type, file):
                keys.append(file)
    return keys

def preprocess_s3(bucket=config["bucket"],  output_dir=config["output_dir"], output_path=config["full_image_norm_path"], temp_output_path=config["full_image_path"], file_type=config["file_type"]):
    """same as preprocessing, but using data stored on an S3 bucket"""
    if temp_output_path not in os.listdir(output_dir):
        s3 = boto3.client("s3")
        shape = config["shape"]
        files = get_s3_keys(s3, bucket)
        images = np.zeros((len(files),shape[0],shape[1],shape[2]-1))

        for i, file in enumerate(tqdm(files)):
            # get file name
            name = output_dir + "/" + re.findall("(?<=/)[^/]*[\.]"+file_type, file)[-1]
            s3.download_file(bucket, file, name)
            img = imread(name, as_gray=False)
            if img.shape != (shape[0],shape[1],shape[2]):
                final_img = resize(img, (shape[0],shape[1],shape[2]))
            else:
                final_img = img
            final = final_img[:,:,0:shape[2]-1]
            images[i,:,:,:] = final
            os.remove(name)

    else:
        images = np.load(output_dir + "/" + temp_output_path)

    print("Shape of images = " + str(images.shape))

    mean_img = np.mean(images, dtype=np.float64)
    temp_images = images - mean_img
    for i, image in enumerate(tqdm(temp_images)):
        temp_images[i,:,:,:] = temp_images[i,:,:,:] * temp_images[i,:,:,:]
    
    ss = np.sum(temp_images)
    var = ss/len(temp_images)
    std_img = np.sqrt(var)
    
    for i, image in enumerate(tqdm(images)):
        images[i,:,:,:] = images[i,:,:,:] - mean_img
        images[i,:,:,:] = images[i,:,:,:]/std_img
    
    np.save(output_dir + "/" + output_path, images)
    print("Loading to S3")
    s3.upload_file(output_dir + "/" + output_path, bucket, output_path)
    print("Removing Locally")
    os.remove(output_dir + "/" + output_path)
    
if __name__ == '__main__':
    preprocess_s3()