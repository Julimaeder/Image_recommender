from Scripts.Image_recommender_Predictions import real_image_label, label_vergleich, extract_image_embeddings
from Scripts.Image_recommender_Vorverarbeitung import load_and_compress_image

def test_image_recommender():
    path = "../.github/Test_bild/standing-german-shepherd.jpg"
    #Test Label:
    expected_output_label = 'German_shepherd' #genaue Werte können nur schelcht verglichen werden, da die Berechnungen teils minimal von einander abweichen
    image = load_and_compress_image(path, target_size=(224, 224))
    output_label = real_image_label(path,image)[1]
    assert output_label == expected_output_label

    #Test Embeddings:
    expected_output_embeddings = 1000
    output_embeddings = len(extract_image_embeddings(image))
    assert output_embeddings == expected_embeddings_label

 #   #Test Daten auslesen aus der Datenbank
 #   expected_output_db = 'E:\\images\\coco2017_unlabeled\\unlabeled2017\\000000075102.jpg'
 #   output_db = label_vergleich(r,3)[0][0]
 #   assert output_db == expected_output_db


