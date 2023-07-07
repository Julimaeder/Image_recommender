"""
Bitte das Scripts. in Zeile 14 entfernen, dient nur zu testzwecken.
"""
import struct # zum konvertieren von binary zu float beim auslesen der db
import sqlite3
import os
from PIL import Image
import numpy as np
import pandas as pd
import pathlib
import matplotlib.pyplot as plt
filepath = str(pathlib.Path(__file__).parent.resolve())
os.chdir(filepath)
from Scripts.Image_recommender_Vorverarbeitung import load_and_compress_image, extract_image_embeddings, label_image, model

def find_nearest_images_embeddings(embeddings_image, df1, num_images=5):
    global embeddings2, distances, indices
    distances = []
    nearest_images_embeddings = []
    for i in range(len(df1)):
        embeddings2 = df1.iloc[i].values[0:]
        distance = np.linalg.norm(embeddings_image - embeddings2)
        distances.append(distance)
    indices = np.argsort(distances)[:num_images]
    
    nearest_images_embeddings = df1.iloc[indices].index.values
    
    return nearest_images_embeddings

def real_image_label(real_image_path):
    image_label = []
    labels = label_image(model, image, target_size=(224, 224), top_labels=5)
    image_label = [real_image_path]
    for label in labels:
        image_label.append(label[1])
        image_label.append(label[2])
    return image_label

def bytes_to_float(b):
    return struct.unpack('f', b)[0]

def label_vergleich(image_label,sql_string_nr = 1): #label_nummer = 1,3,5,7,9
    conn = sqlite3.connect("Data\\databases\\Big_data_database.db")
    curs = conn.cursor()
    if sql_string_nr == 1:
        sql_string = f"SELECT * FROM Labels WHERE Label1 = '{image_label[1]}'"
    elif sql_string_nr == 2:
        sql_string = f"SELECT * FROM Labels WHERE Label1 = '{image_label[1]}' and Label2 = '{image_label[3]}'"
    elif sql_string_nr == 3:
        sql_string = f"SELECT * FROM Labels WHERE Label1 = '{image_label[1]}' and Label2 = '{image_label[3]}' and Label3 = '{image_label[5]}'"
    elif sql_string_nr == 4:
        sql_string = f"SELECT * FROM Labels WHERE Label1 = '{image_label[1]}' and Label2 = '{image_label[3]}' and Label3 = '{image_label[5]}' and Label4 = '{image_label[7]}'"
    elif sql_string_nr == 5:
        sql_string = f"SELECT * FROM Labels WHERE Label1 = '{image_label[1]}' and Label2 = '{image_label[3]}' and Label3 = '{image_label[5]}' and Label4 = '{image_label[7]}' and Label5 = '{image_label[9]}'"
    else:
        print('sql_string_nr must be 1-5')
        return ValueError
    curs.execute(sql_string)
    similar_images = curs.fetchall()
    conn.close()
    converted_images = []
    for image in similar_images:
        converted_image = list(image)
        for i in range(2, len(image), 2):  # Bei Index 2 beginnen und jedes zweite Element überspringen
            if isinstance(image[i], bytes):
                converted_image[i] = bytes_to_float(image[i])
        converted_images.append(tuple(converted_image))
    return converted_images

# wenn über 5 Treffer bestehen, zuerst die Bilder nehmen, welche alle Labels gleich haben. Davon die, bei denen das erste Label am stärksten ist
# Es ist nicht besonders schön, aber mit ganz viel Liebe gemacht :)
def find_nearest_images_label(result_first_1,result_first_2,result_first_3,result_first_4,result_first_5):
    list_nearest_images_label = []
    nearest_images_needed = 5 
    while nearest_images_needed > 0:
        if len(result_first_5) > 0:
            result_first_5_max = 0
            result_first_5_i = 0
            result_first_5_max_path = ''
            for i in range(0,len(result_first_5)):
                if result_first_5[i][2] > result_first_5_max:
                    result_first_5_i = i
                    result_first_5_max = result_first_5[i][2]
                    result_first_5_max_path = result_first_5[i][0]
            list_nearest_images_label.append(result_first_5_max_path)
            result_first_5.remove(result_first_5[result_first_5_i])
            nearest_images_needed -= 1
        if len(result_first_4) > 0:
            result_first_4_max = 0
            result_first_4_i = 0
            result_first_4_max_path = ''
            for i in range(0,len(result_first_4)):
                if result_first_4[i][2] > result_first_4_max:
                    result_first_4_i = i
                    result_first_4_max = result_first_4[i][2]
                    result_first_4_max_path = result_first_4[i][0]
            list_nearest_images_label.append(result_first_4_max_path)
            result_first_4.remove(result_first_4[result_first_4_i])
            nearest_images_needed -= 1
        if len(result_first_3) > 0:
            result_first_3_max = 0
            result_first_3_i = 0
            result_first_3_max_path = ''
            for i in range(0,len(result_first_3)):
                if result_first_3[i][2] > result_first_3_max:
                    result_first_3_i = i
                    result_first_3_max = result_first_3[i][2]
                    result_first_3_max_path = result_first_3[i][0]
            list_nearest_images_label.append(result_first_3_max_path)
            result_first_3.remove(result_first_3[result_first_3_i])
            nearest_images_needed -= 1
        if len(result_first_2) > 0:
            result_first_2_max = 0
            result_first_2_i = 0
            result_first_2_max_path = ''
            for i in range(0,len(result_first_2)):
                if result_first_2[i][2] > result_first_2_max:
                    result_first_2_i = i
                    result_first_2_max = result_first_2[i][2]
                    result_first_2_max_path = result_first_2[i][0]
            list_nearest_images_label.append(result_first_2_max_path)
            result_first_2.remove(result_first_2[result_first_2_i])
            nearest_images_needed -= 1
        if len(result_first_1) > 0:
            result_first_1_max = 0
            result_first_1_i = 0
            result_first_1_max_path = ''
            for i in range(0,len(result_first_1)):
                if result_first_1[i][2] > result_first_1_max:
                    result_first_1_i = i
                    result_first_1_max = result_first_1[i][2]
                    result_first_1_max_path = result_first_1[i][0]
            list_nearest_images_label.append(result_first_1_max_path)
            result_first_1.remove(result_first_1[result_first_1_i])
            nearest_images_needed -= 1
    return list_nearest_images_label

def display_combined_image(image_paths_embeddings,image_paths_label,real_image_path):
    img1 = Image.open(real_image_path)
    plt.imshow(img1)
    plt.axis('off')
    plt.show()
    fig = plt.figure(figsize=(10, 2))
    columns = 5
    rows = 2
    for i in range(1, 6):
        img = Image.open(image_paths_embeddings[i-1])
        fig.add_subplot(rows, columns, i)
        plt.imshow(img)
        plt.axis('off')
    for i in range(1, 6):
        img = Image.open(image_paths_label[i-1])
        fig.add_subplot(rows, columns, i+5)
        plt.imshow(img)
        plt.axis('off')
    plt.show()

def main(real_image_path):
    global nearest_images_embeddings,image
    df1 = pd.read_pickle("Data\\Pickle_embeddings_test.pkl")
    image = load_and_compress_image(real_image_path, target_size=(224, 224))
    embeddings_image = extract_image_embeddings(image)
    nearest_images_embeddings  = find_nearest_images_embeddings(embeddings_image,df1)
    image_label = real_image_label(real_image_path)
    result_first_1 = label_vergleich(image_label,sql_string_nr = 1)
    result_first_2 = label_vergleich(image_label,sql_string_nr = 2)
    result_first_3 = label_vergleich(image_label,sql_string_nr = 3)
    result_first_4 = label_vergleich(image_label,sql_string_nr = 4)
    result_first_5 = label_vergleich(image_label,sql_string_nr = 5)
    for e in result_first_5:
        result_first_4.remove(e)
        result_first_3.remove(e)
        result_first_2.remove(e)
        result_first_1.remove(e)
    for e in result_first_4:
        result_first_3.remove(e)
        result_first_2.remove(e)
        result_first_1.remove(e)
    for e in result_first_3:
        result_first_2.remove(e)
        result_first_1.remove(e)
    for e in result_first_2:
        result_first_1.remove(e)
    nearest_images_label = find_nearest_images_label(result_first_1,result_first_2,result_first_3,result_first_4,result_first_5)
    #Plot Images
    display_combined_image(nearest_images_embeddings,nearest_images_label,real_image_path)

"""
Da das modell mit dem ImageNet Datensatz trainiert wurde, hat es nur 1000 verschiedene Klassen.
So kommen z.B. bei bildern mit einem Regenbogen, Bilder mit Seifenvlasen raus, da es kein Rainbow kennt.
Das erste Label ist hierbei 'bubble'
"""
