import pandas as pd
import pathlib
import os
import sqlite3

filepath = str(pathlib.Path(__file__).parent.resolve()) 
os.chdir(filepath)
from Image_recommender_Predictions import real_image_label, extract_image_embeddings
from Image_recommender_Vorverarbeitung import load_and_compress_image
from Color_Scheme import Full_Prediction

def test_image_recommender():
    global output_label,output_embeddings,output_paths,output_db
    #Test Label:
    #path = "../.github/Test_bild/standing-german-shepherd.jpg"
    path = "..\\.github\\Test_bild\\standing-german-shepherd.jpg"
    expected_output_label = 'German_shepherd' #genaue Werte können nur schelcht verglichen werden, da die Berechnungen teils minimal von einander abweichen
    image = load_and_compress_image(path, target_size=(224, 224))
    output_label = real_image_label(path,image)[1]
    assert output_label == expected_output_label

    #Test Embeddings:
    expected_output_embeddings = 1000 #genaue Werte können nur schelcht verglichen werden, da die Berechnungen teils minimal von einander abweichen
    output_embeddings = len(extract_image_embeddings(image))
    assert output_embeddings == expected_output_embeddings

    #Test scheme:
    input_scheme = [  [1,1,'E:\\images\\weather_image_recognition\\dew\\2208.jpg', 1, 9315, 0, 0, 0, 0, 0, 31859, 0, 0, 0, 0, 0, 3307, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2229, 2, 0, 0, 0, 0, 34805, 17, 0, 0, 0, 0, 2626, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 327, 43, 0, 0, 0, 0, 3735, 277, 21, 0, 0, 0, 50, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 40, 297, 96, 1, 0, 0, 124, 85, 174, 34, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 14, 21, 92, 132, 56, 5, 1, 5, 11, 5, 60, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 11, 15, 0, 1, 2, 6, 53, 25],
                                [1,1,'E:\\images\\weather_image_recognition\\dew\\1.jpg', 1, 93156, 0, 0, 0, 0, 0, 31859, 0, 0, 0, 0, 0, 3307, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2229, 2, 0, 0, 0, 0, 34805, 17, 0, 0, 0, 0, 2626, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 327, 43, 0, 0, 0, 0, 3735, 277, 21, 0, 0, 0, 50, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 40, 297, 96, 1, 0, 0, 124, 85, 174, 34, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 14, 21, 92, 132, 56, 5, 1, 5, 11, 5, 60, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 11, 15, 0, 1, 2, 6, 53, 25],
                                [1,1,'E:\\images\\weather_image_recognition\\dew\\2.jpg', 1, 9317, 0, 0, 0, 0, 0, 31859, 0, 0, 0, 0, 0, 3307, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2229, 2, 0, 0, 0, 0, 34805, 17, 0, 0, 0, 0, 2626, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 327, 43, 0, 0, 0, 0, 3735, 277, 21, 0, 0, 0, 50, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 40, 297, 96, 1, 0, 0, 124, 85, 174, 34, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 14, 21, 92, 132, 56, 5, 1, 5, 11, 5, 60, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 11, 15, 0, 1, 2, 6, 53, 25],
                                [1,1,'E:\\images\\weather_image_recognition\\dew\\3.jpg', 1, 9318, 0, 0, 0, 0, 0, 31859, 0, 0, 0, 0, 0, 3307, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2229, 2, 0, 0, 0, 0, 34805, 17, 0, 0, 0, 0, 2626, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 327, 43, 0, 0, 0, 0, 3735, 277, 21, 0, 0, 0, 50, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 40, 297, 96, 1, 0, 0, 124, 85, 174, 34, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 14, 21, 92, 132, 56, 5, 1, 5, 11, 5, 60, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 11, 15, 0, 1, 2, 6, 53, 25],
                                [1,1,'E:\\images\\weather_image_recognition\\dew\\2208.jpg', 1, 9319, 0, 0, 0, 0, 0, 31859, 0, 0, 0, 0, 0, 3307, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2229, 2, 0, 0, 0, 0, 34805, 17, 0, 0, 0, 0, 2626, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 327, 43, 0, 0, 0, 0, 3735, 277, 21, 0, 0, 0, 50, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 40, 297, 96, 1, 0, 0, 124, 85, 174, 34, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 14, 21, 92, 132, 56, 5, 1, 5, 11, 5, 60, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 11, 15, 0, 1, 2, 6, 53, 25]
                            ]
    output_scheme = ['E:\\images\\weather_image_recognition\\dew\\2208.jpg', 'E:\\images\\weather_image_recognition\\dew\\2.jpg', 'E:\\images\\weather_image_recognition\\dew\\3.jpg', 'E:\\images\\weather_image_recognition\\dew\\2208.jpg', 'E:\\images\\weather_image_recognition\\dew\\1.jpg']
    c = ["iID", "sID"] + [f"spalte_{i}" for i in range(len(input_scheme[0])-2)]
    df = pd.DataFrame(input_scheme, columns=c)
    output_paths = Full_Prediction(path, df)
    assert output_paths == output_scheme
    
    #Test DB Abfrage
    expected_output_db = 281 #die Länge der Liste abzufragen ist einfacher, als 281 Elemente zu vergleichen
    conn = sqlite3.connect("Data\\databases\\Big_data_database.db")
    curs = conn.cursor()
    sql_string = "SELECT * FROM Labels WHERE Label1 = 'bubble'"
    curs.execute(sql_string)
    output_sql = curs.fetchall()
    output_db = len(output_sql)
    conn.close()
    assert output_db == expected_output_db
    
test_image_recommender()


