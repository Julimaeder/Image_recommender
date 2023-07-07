import pathlib, os
filepath = str(pathlib.Path(__file__).parent.resolve())
filepath2 = os.path.join(filepath, 'Scripts')
os.chdir(filepath2)
from Image_recommender_Predictions import main

main("Path/to/image.jpg")