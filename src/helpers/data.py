import gdown
import subprocess
from pathlib import Path

urls = {
  "genomes":"https://drive.google.com/uc?export=download&id=1K3FQ9-aNmS8Mf-yVQDr0WceeCAbAbP9a",
  "proteomes":"https://drive.google.com/uc?export=download&id=1SviZ9QKdsUSOrrvP2TzAbg9fi29iB0HT",
  "promethion":"https://drive.google.com/uc?export=download&id=1EIN8uUuy98oIqYfHadtc2KzOzRH_E1Cs",
  "url": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/url_combined.bz2",
  "webspam": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/webspam_wc_normalized_trigram.svm.xz"
}

raw = {
  "genomes":"genomes-data.gz",
  "proteomes":"proteomes-data.gz",
  "promethion":"promethion-data.gz",
  "url": "url_combined.bz2",
  "webspam": "webspam_wc_normalized_trigram.svm.xz"
}

Path("../raw").mkdir(parents=True, exist_ok=True)
for name in raw.keys():
  Path(f"../data/{name}").mkdir(parents=True, exist_ok=True)

def download(url, output, is_drive=False):
  if is_drive:
    gdown.download(url, output, quiet=False)
  else:
    subprocess.run(["wget", "-t", "inf", url, "-O", output])

def extract(path, output):
  if path.endswith(".gz"):
    subprocess.run(['gunzip', '-c', path], stdout=open(output, 'wb'))
  elif path.endswith(".bz2"):
    subprocess.run(['bunzip2', '-c', path], stdout=open(output, 'wb'))
  elif path.endswith(".xz"):
    subprocess.run(['xz', '-d', path, '-c'], stdout=open(output, 'wb'))

for name,url in urls.items():
  if not Path(f"../raw/{raw[name]}").is_file():
    is_drive = "drive.google.com" in url
    download(url=url, output=f"../raw/{raw[name]}", is_drive=is_drive)
  else:
    print(f"File {name}-data.gz already exists. Skipping download.")

for name in urls.keys():
  if not Path(f"../data/{name}/data").is_file():
    extract(path=f"../raw/{raw[name]}", output=f"../data/{name}/data")
  else:
    print(f"File data/{name}/data already exists. Skipping extraction.")