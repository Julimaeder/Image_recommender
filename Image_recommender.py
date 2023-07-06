import pathlib, os
filepath = str(pathlib.Path(__file__).parent.resolve())
os.chdir(os.path.join(filepath, 'Scripts'))
from Image_recommender_Predictions import main

main("path/to/filename.jpg")