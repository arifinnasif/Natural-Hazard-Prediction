import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

def download_nc_files(url, download_folder):
    response = requests.get(url)

    if response.status_code != 200:
        print(f"Failed to access the URL: {url}")
        return

    soup = BeautifulSoup(response.text, 'html.parser')
    links = soup.find_all('a', href=True)

    for link in links:
        href = link['href']
        if href.endswith('.nc'):
            file_url = urljoin(url, href)
            file_name = os.path.join(download_folder, os.path.basename(href))
            download_file(file_url, file_name)
            print(f"Downloaded: {file_name}")

def download_file(url, file_path):
    response = requests.get(url, stream=True)
    with open(file_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

if __name__ == "__main__":
    # Replace the URL with the target webpage containing .nc files
    target_url = "https://download.avl.class.noaa.gov/download/8327807044/001"
    # Replace the folder path where you want to save the downloaded .nc files
    save_folder = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))+ "\\glm\\"

    download_nc_files(target_url, save_folder)
