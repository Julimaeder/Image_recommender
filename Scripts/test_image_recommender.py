from Scripts.Image_recommender_Predictions import real_image_label, extract_image_embeddings
from Scripts.Image_recommender_Vorverarbeitung import load_and_compress_image
from Scripts.Color_Scheme import Image_to_rgb_scheme, Get_color_scheme
from PIL import Image
import numpy as np

def test_image_recommender():
    path = "../.github/Test_bild/standing-german-shepherd.jpg"
    #Test Label:
    expected_output_label = 'German_shepherd' #genaue Werte können nur schelcht verglichen werden, da die Berechnungen teils minimal von einander abweichen
    image = load_and_compress_image(path, target_size=(224, 224))
    output_label = real_image_label(path,image)[1]
    assert output_label == expected_output_label

    #Test Embeddings:
    expected_output_embeddings = 1000 #genaue Werte können nur schelcht verglichen werden, da die Berechnungen teils minimal von einander abweichen
    output_embeddings = len(extract_image_embeddings(image))
    assert output_embeddings == expected_output_embeddings

    #Test scheme:
    expected_output_scheme = [5500, 595, 0, 0, 0, 0, 140, 88, 0, 0]
    image = Image.open(path)
    vectors = Image_to_rgb_scheme(image)
    color_scheme, count_scheme = Get_color_scheme(vectors)
    rgb_values = np.zeros((6, 6, 6), dtype=int)
    rgb_values[color_scheme[:, 0].astype('int32'), color_scheme[:, 1].astype('int32'), color_scheme[:, 2].astype('int32')] = count_scheme
    np_i = rgb_values.flatten().astype('int32')
    output_scheme = list(np_i[:10])
    assert output_scheme == expected_output_scheme
    
 #   #Test Daten auslesen aus der Datenbank
 #   expected_output_db = 'E:\\images\\coco2017_unlabeled\\unlabeled2017\\000000075102.jpg'
 #   output_db = label_vergleich(r,3)[0][0]
 #   assert output_db == expected_output_db

