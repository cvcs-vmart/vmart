import json
import os
import argparse
from typing import TextIO

def dict_const(path):

    dictionary = {}
    i = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.jpg'):
                dictionary[f"{i}"] = root + '/' + file
                i += 1
                print(file)

    with open('all_paintings.json', 'w') as f:
        json.dump(dictionary, f) # type: ignore

if __name__ == '__main__':
    # parser per i parametri
    parser = argparse.ArgumentParser()
    parser.add_argument("--dirpath", type=str, default="/work/cvcs2025/bilardello_melis_prato/wikiart",)
    args = parser.parse_args()

    dict_const(args.dirpath)