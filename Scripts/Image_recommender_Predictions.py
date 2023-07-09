import struct # zum konvertieren von binary zu float beim auslesen der db
import sqlite3
import os
from PIL import Image
import numpy as np
import pandas as pd
import pathlib
import matplotlib.pyplot as plt
#cwd zum speicherort der Datei wechseln, kam sonst zu Problemen mit dem import
filepath = str(pathlib.Path(__file__).parent.resolve()) 
os.chdir(filepath)
# Funktionen aus den anderen Dateien importieren
from Image_recommender_Vorverarbeitung import load_and_compress_image, extract_image_embeddings, label_image, model
from Color_Scheme import Path_generator, Full_Prediction, readAllImages

# Sucht die 5 nächsten embeddings aus dem df zu den imput embeddings von einem Bild
def find_nearest_images_embeddings(embeddings_image, df1, num_images=5):
    global embeddings2, distances, indices
    distances = []
    nearest_images_embeddings = []
    # geht das ganze df durch
    for i in range(len(df1)):
        # mit iloc kann man die indices eines df abfragen, so kann man durch das df loopen
        embeddings2 = df1.iloc[i].values[0:]
        distance = np.linalg.norm(embeddings_image - embeddings2)
        distances.append(distance)
    # Nimmt die indices ersten 5 items aus distances 
    indices = np.argsort(distances)[:num_images]
    # gibt die values, also filenames, von den indices wieder
    nearest_images_embeddings = df1.iloc[indices].index.values
    #Array aus den 5 Paths\\filenames der ähnlichsten Embeddings 
    return nearest_images_embeddings

# Labelt das Input image; siehe main() von Image_recommender_Vorverarbeitung
def real_image_label(real_image_path,image):
    image_label = []
    labels = label_image(model, image, target_size=(224, 224), top_labels=5)
    # Die Label werden als tupel, bestehend aus dem path//filename und dann abwechselnd das Label und der Wert des Labels, gespeichert
    image_label = [real_image_path]
    for label in labels:
        image_label.append(label[1]) # Das sind die Label Namen, Label[0] ist die cryptische Bezeichnung der Label
        image_label.append(label[2]) # Wert des Labels
    return image_label

# Die eigentlichen floats werden beim auslesen der Datenbank in bytes ausgelesen und müssen konvertiert werden
def bytes_to_float(b):
    return struct.unpack('f', b)[0]

# Hier weden die Label des input images mit allen Labeln aus der Datenbank vergliche
def label_vergleich(image_label,sql_string_nr = 1): 
    conn = sqlite3.connect("Data\\databases\\Big_data_database.db")
    curs = conn.cursor()
    # Je nach sql_string_nr werden die Bilder mit den selben ersten 1,2,3,4 oder 5 Labeln rausgesucht
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
    # Jeder float, also jedes 2. Element, beginnend ab dem 3. Element, muss von byte zu float konvertiert werden
    converted_images = []
    for image in similar_images:
        converted_image = list(image)
        for i in range(2, len(image), 2):  # Bei Index 2 beginnen und jedes zweite Element überspringen
            if isinstance(image[i], bytes):
                converted_image[i] = bytes_to_float(image[i])
        converted_images.append(tuple(converted_image))
    # Liste aus allen treffern
    return converted_images

# wenn über 5 Treffer bestehen, zuerst die Bilder nehmen, welche alle Labels gleich haben. Davon die, bei denen das erste Label am stärksten ist
def find_nearest_images_label(result_first_1,result_first_2,result_first_3,result_first_4,result_first_5):
    list_nearest_images_label = []
    nearest_images_needed = 5 
    while nearest_images_needed > 0: #so lange, bis man 5 Bilder hat
        if len(result_first_5) > 0: #wenn etwas in der Liste mit 5 gleichen Labeln ist
            result_first_5_max = 0
            result_first_5_i = 0
            result_first_5_max_path = ''
            for i in range(0,len(result_first_5)):
                if result_first_5[i][2] > result_first_5_max: #Das Bild finden, bei dem das erste Label am stärksten ist und alles zu dem Bild speichern
                    result_first_5_i = i #Index in der Liste, damit es removed werden kann
                    result_first_5_max = result_first_5[i][2] # Element mit dem Index 2 ist der Wert des ersten Labels
                    result_first_5_max_path = result_first_5[i][0] # Element mit dem Index 0 ist der Pfad des ersten Labels
            list_nearest_images_label.append(result_first_5_max_path)
            result_first_5.remove(result_first_5[result_first_5_i]) # Aus der Liste entfernen, damit es nicht zu dopplungen kommt
            nearest_images_needed -= 1
        if len(result_first_4) > 0: #wenn etwas un der Liste mit 4 gleichen Labeln ist
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
        if len(result_first_3) > 0:#wenn etwas un der Liste mit 3 gleichen Labeln ist
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
        if len(result_first_2) > 0: #wenn etwas un der Liste mit 2 gleichen Labeln ist
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
        if len(result_first_1) > 0: #wenn etwas un der Liste mit einem gleichen Label ist
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
    # Paths der 5 ähnlcihsten Bilder
    return list_nearest_images_label

# Alle 3x5 Bilder plotten
def display_combined_image(image_paths_embeddings,image_paths_label,real_image_path,color_schemes):
    # Input Bild plotten
    img1 = Image.open(real_image_path)
    plt.imshow(img1)
    plt.axis('off')
    plt.title('Input image')
    plt.show()
    fig = plt.figure(figsize=(10, 6))
    columns = 5
    rows = 3
    # Bilder mit den nächsten Embeddings
    for i in range(1, 6):
        img = Image.open(image_paths_embeddings[i-1])
        ax = fig.add_subplot(rows, columns, i) # Subplot, damit alle Bilder zusammen angezeigt werden können
        if i == 3:  # Das mittlere Bild bekommt den Titel. So hat man keine Probleme mit der Positionierung des Textes im Plot
            ax.set_title('Vergleich mittels Embeddings', fontsize = 10)
        plt.imshow(img)
        plt.axis('off')
    # Bilder mit den ähnlichsten Labeln
    for i in range(1, 6):
        img = Image.open(image_paths_label[i-1])
        ax = fig.add_subplot(rows, columns, i+5) # i+5, weil i hier wieder von vorne anfängt, aber schon 5 Bilder in der Liste sind. Die würden sonst überschrieben werden
        if i == 3:  # Das mittlere Bild bekommt den Titel. So hat man keine Probleme mit der Positionierung des Textes im Plot
            ax.set_title('Vergleich mittels Label', fontsize = 10)
        plt.imshow(img)
        plt.axis('off')
    # Bilder mit den ähnlichsten Farbwerten
    for i in range(1, 6):
        img = Image.open(color_schemes[i-1])
        ax = fig.add_subplot(rows, columns, i+10)
        if i == 3:  # Das mittlere Bild bekommt den Titel. So hat man keine Probleme mit der Positionierung des Textes im Plot
            ax.set_title('Vergleich mittels Farbschema', fontsize = 10)
        plt.imshow(img)
        plt.axis('off')
    fig.subplots_adjust(hspace=0.3)  # Platz zwischen den Zeilen erhöhen
    fig.subplots_adjust(wspace=0.4)  # Platz zwischen den Spalten erhöhen
    plt.show()

def main(image_path):
    global nearest_images_embeddings,image,result_first_1,result_first2,result_first_3,result_first_4,result_first_5
    df1 = pd.read_pickle("Data\\Pickle_embeddings_test.pkl") # Pickle Datei mit den Embeddings auslesen
    df = readAllImages() # Datenbank mit den Farbschemen auslesen
    image_generator = Path_generator(image_path) #Generator, der durch alle images im Ordner geht. So können mehrere direkt hintereinander gemacht werden
    # Jedes Bild durch alle Funktionen schicken
    for real_image_path in image_generator:
        image = load_and_compress_image(real_image_path, target_size=(224, 224))
        color_schemes = Full_Prediction(real_image_path, df)
        embeddings_image = extract_image_embeddings(image)
        nearest_images_embeddings  = find_nearest_images_embeddings(embeddings_image,df1)
        image_label = real_image_label(real_image_path,image)
        result_first_1 = label_vergleich(image_label,sql_string_nr = 1)
        result_first_2 = label_vergleich(image_label,sql_string_nr = 2)
        result_first_3 = label_vergleich(image_label,sql_string_nr = 3)
        result_first_4 = label_vergleich(image_label,sql_string_nr = 4)
        result_first_5 = label_vergleich(image_label,sql_string_nr = 5)
        # Um bei den Labels Bilddopplungen zu vermeiden, werden alle Bilder gelöscht, die schon in der Liste mit mehr gleichen Labeln sind
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
        display_combined_image(nearest_images_embeddings,nearest_images_label,real_image_path,color_schemes)