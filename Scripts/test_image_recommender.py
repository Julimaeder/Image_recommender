from Scripts.Image_recommender_Predictions import real_image_label, extract_image_embeddings
from Scripts.Image_recommender_Vorverarbeitung import load_and_compress_image
from Scripts.Color_Scheme import Image_to_rgb_scheme, Get_color_scheme, Full_Prediction
from PIL import Image
import numpy as np
import pandas as pd

def test_image_recommender():
    #Test Label:
    path = "../.github/Test_bild/standing-german-shepherd.jpg"
    expected_output_label = 'German_shepherd' #genaue Werte können nur schelcht verglichen werden, da die Berechnungen teils minimal von einander abweichen
    image = load_and_compress_image(path, target_size=(224, 224))
    output_label = real_image_label(path,image)[1]
    assert output_label == expected_output_label

    #Test Embeddings:
    expected_output_embeddings = 1000 #genaue Werte können nur schelcht verglichen werden, da die Berechnungen teils minimal von einander abweichen
    output_embeddings = len(extract_image_embeddings(image))
    assert output_embeddings == expected_output_embeddings

    #Test scheme:
    path = "../.github/Test_bild/standing-german-shepherd.jpg"
    input_scheme = [  [1,1,'E:\\images\\weather_image_recognition\\dew\\2208.jpg', 1, 9315, 0, 0, 0, 0, 0, 31859, 0, 0, 0, 0, 0, 3307, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2229, 2, 0, 0, 0, 0, 34805, 17, 0, 0, 0, 0, 2626, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 327, 43, 0, 0, 0, 0, 3735, 277, 21, 0, 0, 0, 50, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 40, 297, 96, 1, 0, 0, 124, 85, 174, 34, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 14, 21, 92, 132, 56, 5, 1, 5, 11, 5, 60, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 11, 15, 0, 1, 2, 6, 53, 25],
                                [1,1,'E:\\images\\weather_image_recognition\\dew\\1.jpg', 1, 93156, 0, 0, 0, 0, 0, 31859, 0, 0, 0, 0, 0, 3307, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2229, 2, 0, 0, 0, 0, 34805, 17, 0, 0, 0, 0, 2626, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 327, 43, 0, 0, 0, 0, 3735, 277, 21, 0, 0, 0, 50, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 40, 297, 96, 1, 0, 0, 124, 85, 174, 34, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 14, 21, 92, 132, 56, 5, 1, 5, 11, 5, 60, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 11, 15, 0, 1, 2, 6, 53, 25],
                                [1,1,'E:\\images\\weather_image_recognition\\dew\\2.jpg', 1, 9317, 0, 0, 0, 0, 0, 31859, 0, 0, 0, 0, 0, 3307, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2229, 2, 0, 0, 0, 0, 34805, 17, 0, 0, 0, 0, 2626, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 327, 43, 0, 0, 0, 0, 3735, 277, 21, 0, 0, 0, 50, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 40, 297, 96, 1, 0, 0, 124, 85, 174, 34, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 14, 21, 92, 132, 56, 5, 1, 5, 11, 5, 60, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 11, 15, 0, 1, 2, 6, 53, 25],
                                [1,1,'E:\\images\\weather_image_recognition\\dew\\3.jpg', 1, 9318, 0, 0, 0, 0, 0, 31859, 0, 0, 0, 0, 0, 3307, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2229, 2, 0, 0, 0, 0, 34805, 17, 0, 0, 0, 0, 2626, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 327, 43, 0, 0, 0, 0, 3735, 277, 21, 0, 0, 0, 50, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 40, 297, 96, 1, 0, 0, 124, 85, 174, 34, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 14, 21, 92, 132, 56, 5, 1, 5, 11, 5, 60, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 11, 15, 0, 1, 2, 6, 53, 25],
                                [1,1,'E:\\images\\weather_image_recognition\\dew\\2208.jpg', 1, 9319, 0, 0, 0, 0, 0, 31859, 0, 0, 0, 0, 0, 3307, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2229, 2, 0, 0, 0, 0, 34805, 17, 0, 0, 0, 0, 2626, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 327, 43, 0, 0, 0, 0, 3735, 277, 21, 0, 0, 0, 50, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 40, 297, 96, 1, 0, 0, 124, 85, 174, 34, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 14, 21, 92, 132, 56, 5, 1, 5, 11, 5, 60, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 11, 15, 0, 1, 2, 6, 53, 25]
                            ]
    output_scheme = [   'E:\\images\\weather_image_recognition\\dew\\2208.jpg',
                        'E:\\images\\weather_image_recognition\\dew\\1.jpg',
                        'E:\\images\\weather_image_recognition\\dew\\2.jpg',
                        'E:\\images\\weather_image_recognition\\dew\\3.jpg',
                        'E:\\images\\weather_image_recognition\\dew\\2208.jpg']
    c = ["iID", "sID"] + [f"spalte_{i}" for i in range(len(input_scheme[0]))]
    df = pd.DataFrame(input_scheme, columns=c)
    output_paths = Full_Prediction(path, df)
    # image = Image.open(path)
    # vectors = Image_to_rgb_scheme(image)
    # color_scheme, count_scheme = Get_color_scheme(vectors)
    # rgb_values = np.zeros((6, 6, 6), dtype=int)
    # rgb_values[color_scheme[:, 0].astype('int32'), color_scheme[:, 1].astype('int32'), color_scheme[:, 2].astype('int32')] = count_scheme
    # np_i = rgb_values.flatten().astype('int32')
    # output_scheme = list(np_i[:10])
    assert output_paths == output_scheme
    
 #   #Test Daten auslesen aus der Datenbank
 #   expected_output_db = 'E:\\images\\coco2017_unlabeled\\unlabeled2017\\000000075102.jpg'
 #   output_db = label_vergleich(r,3)[0][0]
 #   assert output_db == expected_output_db

