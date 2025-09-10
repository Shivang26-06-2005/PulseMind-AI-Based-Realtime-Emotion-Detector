import os

TESS_PATH = r"D:\DataSet\Sound Dataset\TESS Toronto emotional speech set data"

for root, dirs, files in os.walk(TESS_PATH):
    if dirs:  # only print folder names
        for d in dirs:
            print(d)
