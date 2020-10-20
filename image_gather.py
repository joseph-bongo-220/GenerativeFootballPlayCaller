from bs4 import BeautifulSoup
import requests
import re
import json
import wget
import boto3
import os
from get_config import get_config

config = get_config()

class PlayScraper(object):
    """This object scrapes Genius for scipts to game of thrones and the office"""
    def __init__(self, bucket_name, prefix, starting_index=0, FORCE=False):
        """Initialize Seasons and URLs with error handling"""

        self.s3 = boto3.client("s3")
        self.bucket_name = bucket_name
        self.prefix = prefix
        self.url = "https://maddenfocus.com"
        self.starting_index = starting_index
        self.FORCE = FORCE

    def image_to_s3(self, image):
        print(image + " Loaded to s3")
        wget.download(self.url+image)
        png_file = [x for x in os.listdir() if re.search(".*\.png", x)][0]
        output_path = self.prefix + self.team + '/' + png_file
        self.s3.upload_file(png_file, self.bucket_name, output_path)
        os.remove(png_file)

    def create_s3_folder(self, link):
        self.team = re.sub('/formation/', '', link)
        print(self.team)
        self.s3.put_object(Bucket=self.bucket_name, Key=(self.prefix + self.team + '/'))

    def get_image_links(self, link):
        response = requests.get(self.url+link, timeout=5)
        if response.status_code == 200:
            content = BeautifulSoup(response.content, "html.parser")
            block = content.find_all("li", {"class":"bruh"})
            image_links = [b.find("a").find("img").get("src") for b in block]
            return image_links
        else:
            raise ValueError("Error: Request returned non-200 status code of " + str(response.status_code))
    
    def get_play_links(self, link):
        response = requests.get(self.url+link, timeout=5)
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
        response = requests.get(self.url+link, timeout=5)
        if response.status_code == 200:
            content = BeautifulSoup(response.content, "html.parser")
            block = content.find_all("table")[0].find_all("a") 
            form_links = [b.get("href") for b in block]
            return form_links
        else:
            raise ValueError("Error: Request returned non-200 status code of " + str(response.status_code))

    def get_team_links(self):
        response = requests.get(self.url+"/teams", timeout=5)
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
        contents = self.s3.list_objects(Bucket=self.bucket_name, Prefix=self.prefix)
        try:
            contents = contents['Contents']
            if len(contents) == 1:
                print("Directory specified is empty. Retrieving photos.")
                WRITE = True
            elif self.starting_index != 0:
                print("Index Specified. Retrieving photos.")
                WRITE = True
            else:
                if self.FORCE:
                    print("User forcing rewrite. Retrieving photos.")
                    WRITE = True
                else:
                    print("Directory appears to be full. Will not retrieve photos.")
                    WRITE = False
        except KeyError as e:
            print("Directory does not exist. Will not retrieve photos.")
            WRITE = False

        if WRITE:
            self.get_team_links()
            for link in self.links:
                self.create_s3_folder(link=link)
                form_links = self.get_formation_links(link=link)
                print(form_links)
                for form_link in form_links:
                    play_links = self.get_play_links(link=form_link)
                    print(play_links)
                    for play_link in play_links:
                        image_links = self.get_image_links(link=play_link)
                        print(len(image_links))
                        for image in image_links:
                            self.image_to_s3(image=image)


def gather_images(bucket_name, prefix, starting_index, FORCE):
    ps = PlayScraper(bucket_name=bucket_name, prefix=prefix, starting_index=starting_index, FORCE=FORCE)
    ps.run()

if __name__ == "__main__":
    gather_images(bucket_name=config["bucket"], prefix=config["prefix"], starting_index=0, FORCE=False)
