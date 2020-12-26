from bs4 import BeautifulSoup
import requests
import re
import json
import wget
import boto3
import os
from get_config import get_config

config = get_config()
config = config["data_prep"]

class PlayScraper(object):
    """This object scrapes play images from internet and"""
    def __init__(self, bucket_name, prefix, starting_index=0, FORCE=False):
        """Initialize AWS services and URL parameters"""

        # get S3 client
        self.s3 = boto3.client("s3")
        
        # save bucket name
        self.bucket_name = bucket_name

        # save url and S3 prefix
        self.prefix = prefix
        self.url = "https://maddenfocus.com"

        # save other params
        self.starting_index = starting_index
        self.FORCE = FORCE

    def image_to_s3(self, image):
        """Loads image to S3"""
        print(image + " Loaded to s3")

        # downloads image at the given URL
        wget.download(self.url+image)

        # gets name of .png file
        png_file = [x for x in os.listdir() if re.search(".*\.png", x)][0]

        #creates file path for S3
        output_path = self.prefix + self.team + '/' + png_file

        # loads image to S3 and removes local copy
        self.s3.upload_file(png_file, self.bucket_name, output_path)
        os.remove(png_file)

    def create_s3_folder(self, link):
        """creates a folder in S3 for each team"""
        # gets team name from link
        self.team = re.sub('/formation/', '', link)
        print(self.team)

        # creates team folder
        self.s3.put_object(Bucket=self.bucket_name, Key=(self.prefix + self.team + '/'))

    def get_image_links(self, link):
        """gets the links for each image"""
        
        # gets website html
        response = requests.get(self.url+link, timeout=5)

        # if there is a valid response, we parse out the links using BeautifulSoup
        # otherwise we raise an error
        if response.status_code == 200:
            content = BeautifulSoup(response.content, "html.parser")
            block = content.find_all("li", {"class":"bruh"}) # yes this class was actually named "bruh"
            image_links = [b.find("a").find("img").get("src") for b in block]
            return image_links
        else:
            raise ValueError("Error: Request returned non-200 status code of " + str(response.status_code))
    
    def get_play_links(self, link):
        """gets the links for each play"""
        # gets website html
        response = requests.get(self.url+link, timeout=5)

        # if there is a valid response, we parse out the links using BeautifulSoup
        # otherwise we raise an error
        if response.status_code == 200:
            content = BeautifulSoup(response.content, "html.parser")
            block = content.find_all("table")[0].find_all("a") 
            play_links = [b.get("href") for b in block]
            play_links = list(set(play_links))
            play_links.sort()
            return play_links
        else:
            raise ValueError("Error: Request returned non-200 status code of " + str(response.status_code))

    def get_formation_links(self, link):
        """gets the links for each formation"""

        # gets website html
        response = requests.get(self.url+link, timeout=5)

        # if there is a valid response, we parse out the links using BeautifulSoup
        # otherwise we raise an error
        if response.status_code == 200:
            content = BeautifulSoup(response.content, "html.parser")
            block = content.find_all("table")[0].find_all("a") 
            form_links = [b.get("href") for b in block]
            return form_links
        else:
            raise ValueError("Error: Request returned non-200 status code of " + str(response.status_code))

    def get_team_links(self):
        """gets the links for each formation"""

        # gets website html
        response = requests.get(self.url+"/teams", timeout=5)

        # if there is a valid response, we parse out the links using BeautifulSoup
        # otherwise we raise an error
        if response.status_code == 200:
            content = BeautifulSoup(response.content, "html.parser")
            block = content.find_all("table")[0].find_all("a")
            links = [b.get("href") for b in block if 'formation' in b.get("href")]
            self.links = list(set(links))
            self.links.sort()
            self.links = self.links[self.starting_index:]
        else:
            raise ValueError("Error: Request returned non-200 status code of " + str(response.status_code))

    def run(self):
        # list contents of our s3 bucket with the given prefix
        contents = self.s3.list_objects(Bucket=self.bucket_name, Prefix=self.prefix)


        try:
            contents = contents['Contents']
            if len(contents) == 1:
                # if the directory is empty, we gather the photos
                print("Directory specified is empty. Retrieving photos.")
                WRITE = True

            elif self.starting_index != 0:
                # if the user specifies an index at which to start, this automatically triggers photo retrieval
                print("Index Specified. Retrieving photos.")
                WRITE = True

            else:
                # if there are files present and the starting index is 0, we only write the new files if the user
                # forces this via the force flag
                if self.FORCE:
                    print("User forcing rewrite. Retrieving photos.")
                    WRITE = True
                else:
                    print("Directory appears to be full. Will not retrieve photos.")
                    WRITE = False

        # if we get a key error, then we throw an error and do not execute the code below
        except KeyError as e:
            print("Directory does not exist. Will not retrieve photos.")
            WRITE = False

        # if we need to actually gather the images
        if WRITE:
            # get the links for each team, if starting_index=0, gather all teams
            # Otherwise we gather teams starting_index:32
            # The teams are indeed in alphabetical order
            self.get_team_links()

            # iterate over the links for each team
            for link in self.links:

                # create a folder in S3 for the team
                self.create_s3_folder(link=link)

                # get the formation links for each team
                form_links = self.get_formation_links(link=link)
                print(form_links)

                # iterate over every formation for the given team
                for form_link in form_links:

                    # get the play links for each formation
                    play_links = self.get_play_links(link=form_link)
                    print(play_links)

                    # iterate over every play for every formation for every team
                    for play_link in play_links:

                        # get the link(s) for the image of that play
                        # (should only be one image per play)
                        image_links = self.get_image_links(link=play_link)
                        print(len(image_links))

                        # iterate over the image for every play for every formation for every team
                        for image in image_links:
                            # load this image to S3
                            self.image_to_s3(image=image)


def gather_images(bucket_name, prefix, starting_index, FORCE):
    """create simple decorator to execute 'run' method of PlayScraper"""
    ps = PlayScraper(bucket_name=bucket_name, prefix=prefix, starting_index=starting_index, FORCE=FORCE)
    ps.run()

if __name__ == "__main__":
    gather_images(bucket_name=config["bucket"], prefix=config["prefix"], starting_index=0, FORCE=False)
