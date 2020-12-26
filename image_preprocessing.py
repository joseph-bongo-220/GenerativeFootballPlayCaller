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
config = config["data_prep"]

def get_s3_keys(client, bucket=config["bucket"], teams=config["teams"], prefix=config["prefix"], file_type=config["file_type"]):
    """Get a list of keys in an S3 bucket."""

    # initialize empty list of keys
    keys = []

    # iterate on each team listed in the config (I could have scraped each team name and done this dynamically,
    # but this was easier given that there are only 32 teams)
    for team in teams:
        # add prefix and team name to get proper S3 file path
        prefix_ = prefix + team

        # lists names of all .png files for each play of the given team
        resp = client.list_objects_v2(Bucket=bucket, Prefix=prefix_)

        # iterate over each file name for that team
        for obj in resp['Contents']:

            # get file name
            file = obj['Key']

            # if the files is a .png file, the we know it is a play (data for our models) and we add it to the
            # keys list
            if re.findall("(?<=/)[^/]*[\.]"+file_type, file):
                keys.append(file)
    return keys

def preprocess_s3(bucket=config["bucket"], output_dir=config["output_dir"], output_path=config["full_image_norm_path"], temp_output_path=config["full_image_path"], file_type=config["file_type"]):
    """same as preprocessing, but using data stored on an S3 bucket"""

    # if we do not have temporary output (output that has not been normalized by formula below)
    # then we must create the numerical data from the .png files
    #
    # Formula:
    #           normalized_x = (x-mean(x))/standard_deviation(x)
    if temp_output_path not in os.listdir(output_dir):
        # get S3 client
        s3 = boto3.client("s3")

        # get desired chape from config file
        shape = config["shape"]

        # get the keys for the play image .png files
        files = get_s3_keys(s3, bucket)

        # initialize an empty images numpy array to the proper sizeto avoid memory error
        # we remove one observation in last dimension because all parts of
        # image have the same level of transparency 
        # Shape Description:
        #   Number of Images, Height of Each Image, Width of Each Image, Parts (Red, Green, Blue) of Each Image
        # Originally read in as parts (Red, Green, Blue, Transparency), but all of the images are completely opaque, 
        # so this was removed
        images = np.zeros((len(files),shape[2]-1,shape[0],shape[1]))

        # iterate over each .png file
        for i, file in enumerate(tqdm(files)):
            # get file name by stripping info from file string
            name = output_dir + "/" + re.findall("(?<=/)[^/]*[\.]"+file_type, file)[-1]

            # download the proper image file from S3
            s3.download_file(bucket, file, name)

            # use imageio library to read each image as a 3-D numpy array
            # This is read in as parts (Red, Green, Blue, Transparency)
            img = imread(name, as_gray=False)
            img = np.swapaxes(img,0,2)
            img = np.swapaxes(img,1,2)

            # if the image is not the desired shape, then user skimage's resize algorithm
            # to make it the proper size
            if img.shape != (shape[2],shape[0],shape[1]):
                final_img = resize(img, (shape[2],shape[0],shape[1]))
            else:
                final_img = img
            
            # remove transparency measure
            final = final_img[0:shape[2]-1,:,:]

            # insert image data into proper position intoimages array
            images[i,:,:,:] = final/255

            # remove file from machine to save memory/disc space
            os.remove(name)

    # if the temporary file is available, then we simply download it
    else:
        images = np.load(output_dir + "/" + temp_output_path)

    # print shape of array
    print("Shape of images = " + str(images.shape))

    # # get mean measure for each image
    # mean_img = np.mean(images, dtype=np.float64)

    # # create a  subtract this from the images so that the mean value for each dimension is 0
    # temp_images = images - mean_img

    # # iterate over each image observation to avoid memory issues
    # for i, image in enumerate(tqdm(temp_images)):

    #     # get squared distance to mean for each observation and dimension
    #     temp_images[i,:,:,:] = temp_images[i,:,:,:] * temp_images[i,:,:,:]
    
    # # sum this over each observation to then get variance, standard deviation
    # ss = np.sum(temp_images)
    # var = ss/len(temp_images)
    # std_img = np.sqrt(var)
    
    # # iterate over each image observation to avoid memory issues
    # for i, image in enumerate(tqdm(images)):
    #     # subtract mean
    #     images[i,:,:,:] = images[i,:,:,:] - mean_img

    #     # divide by standard deviations
    #     images[i,:,:,:] = images[i,:,:,:]/std_img
    
    # data is now normalized. save to new .npy file
    np.save(output_dir + "/" + output_path, images)

    # load file to S3 bucket
    print("Loading to S3")
    s3.upload_file(output_dir + "/" + output_path, bucket, output_path)

    # remove local version of file to save disk space/memory
    print("Removing Locally")
    os.remove(output_dir + "/" + output_path)

def get_cleaned_data(bucket=config["bucket"], output_dir=config["output_dir"], output_path=config["full_image_norm_path"]):
    """function to download the cleaned and normalized data from S3"""
    name = output_dir + "/" + output_path

    if os.path.exists(name):
        return np.load(name, mmap_mode='r')
    else:
        s3 = boto3.client("s3")
        s3.download_file(bucket, output_path, name)
        return np.load(name, mmap_mode='r')
    
if __name__ == '__main__':
    preprocess_s3()