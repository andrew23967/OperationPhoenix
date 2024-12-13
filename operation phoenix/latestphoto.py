import requests

url = 'http://andrewvalentino.pythonanywhere.com/camera/latest-photo/'

response = requests.get(url)

def get_photo(filename):
    if response.status_code == 200:
        with open(filename, 'wb') as f:
            f.write(response.content)
    else:
        pass
