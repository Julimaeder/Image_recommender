"""
Infos:
- Jedes Bild hat 2 Progress Bars: Die erste ist extract_image_embeddings und die zweite label_image
"""
import sqlite3
import os
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, decode_predictions
from PIL import Image
import numpy as np
import logging
import time
import pandas as pd
from tqdm import tqdm

# Datenbank für die Labels erstellen
# Überprüfen, ob der Ordner 'databases' existiert, wenn nicht erstellen
if not os.path.exists("databases"):
    os.makedirs("databases")
conn = sqlite3.connect("databases/Big_data_database.db")
curs = conn.cursor()
curs.execute("""
CREATE TABLE IF NOT EXISTS Labels
(Pfad TEXT, 
Label1 TEXT, Label1_Wert REAL, 
Label2 TEXT, Label2_Wert REAL,
Label3 TEXT, Label3_Wert REAL,
Label4 TEXT, Label4_Wert REAL,
Label5 TEXT, Label5_Wert REAL)
""")
conn.commit()
conn.close()

#logging, um zu schauen, welche Bilder nicht funktioniert haben
#logging.basicConfig(filename='Logging_test.log', level=logging.ERROR)

folder_path = "E:\\images"
model = MobileNetV2(weights="imagenet")

# Bilder komprimieren, damit MobileNet damit arbeiten kann
def load_and_compress_image(image_path, target_size):
    image = Image.open(image_path)
    image = image.resize(target_size)
    image_array = np.array(image)
    preprocessed_image = tf.keras.applications.mobilenet.preprocess_input(image_array)
    return preprocessed_image

# Embeddings extrahieren mittels ImageNetV2
def extract_image_embeddings(image):
    image = np.expand_dims(image, axis=0)
    embeddings = model.predict(image)
    return embeddings.flatten()

# generator, welcher alle Pfade in einem Ordner (mit Unterordnern) wiedergibt
def image_generator(folder_path):
    global dirpath,filename
    image_extensions = ('.jpg', '.png', '.jpeg')
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for filename in filenames:
            if filename.lower().endswith(image_extensions):
                yield os.path.join(dirpath, filename)

# Checkpoint in checkpoint.txt speichern
def save_checkpoint(data, filename='checkpoint.txt'):
    with open(filename, 'w') as f:
        f.write(str(data))

# Checkpoint laden
def load_checkpoint(filename='checkpoint.txt'):
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return int(f.read())
    else:
        return None
    
# Bilder mittels ImageNetV2 Labeln und die 5 größten Label, uinklusive Werte zurückgeben
def label_image(model, compressed_image, target_size=(224, 224), top_labels=5):
    image = compressed_image
    if len(image.shape) == 3:
        image = np.expand_dims(image, axis=0)
    predictions = model.predict(image)
    labels = decode_predictions(predictions, top=top_labels)
    return labels[0]
    
# Embeddings in einer pickle Datei speichern (kleiner und schneller als eine DB, wenn man auf alle Werte zugreifen muss;
# Bei den Labels kann man mit einem WHERE statement nur die Daten, die man braucht auslesen, deswegen werden nur die
# Embeddings in der Pickle gespeichert)
def save_pickle(embeddings_list,filenames_list,i):
    embeddings_df = pd.DataFrame(embeddings_list, index=filenames_list)
    pickle_filename = 'Pickle_embeddings_test.pkl'
    save_checkpoint(i)  # nur alle 500, weil checkpoint sonst ggf weiter ist als das Speichern 
                        #--> dann würden bis zu 499 Bilder verloren gehen
    if os.path.exists(pickle_filename):
        # Daten aus der pickle auslesen, und vor das df packen, damit die alten DAten nicht überschrieben werden
        with open(pickle_filename, "rb"):
            existing_data = pd.read_pickle(pickle_filename)
        embeddings_df = pd.concat([existing_data, embeddings_df])
    with open(pickle_filename, "wb"):
        pd.to_pickle(embeddings_df, pickle_filename)

def main():
    global i #Um zu sehen, wie weit man schon ist
    batch_size = 500 #in 500er Schritten speichern
    embeddings_list = []
    filenames_list = []
    data = []
    error_counter = 0
    start_time = time.time()
    generator = image_generator(folder_path)
    checkpoint = load_checkpoint()
    i = 0
    start = checkpoint + 1 if checkpoint is not None else 0
    # Generator so lange durch laufen lassen, bis er den Chekpoint erreicht hat, damit nicht jedes mal von vorne angefangen wird
    for i in tqdm(range(0,start)):
        next(generator)
    # Jedes Bild durch die Funktionen schicken
    for image_path in generator:
        try:  
            temp2 = load_and_compress_image(image_path, target_size=(224, 224))
            embeddings = extract_image_embeddings(temp2)
            filenames_list.append(image_path)
            embeddings_list.append(embeddings)
            labels = label_image(model, temp2, target_size=(224, 224), top_labels=5)
            # Die Label werden immer als tupel, bestehend aus dem path//filename und dann abwechselnd die Label und die Werte der Label, gespeichert
            row = [image_path]
            for label in labels:
                row.append(label[1]) # Das sind die Label Namen, Label[0] ist die cryptische Bezeichnung der Label
                row.append(label[2]) # Wert des Labels
            data.append(tuple(row))
            i +=1
        # Fehler aufschreiben und weiter machen
        except Exception as e:
            logging.error(f"Error at point {i}: {e}")
            error_counter = error_counter +1
            continue
        # Alle 500 iterationen speichern
        if i % batch_size == 0:
            save_pickle(embeddings_list,filenames_list,i)
            embeddings_list = [] # Listen leeren, damit es nicht zu dopplungen kommt
            filenames_list = []
            conn = sqlite3.connect("databases/Big_data_database.db")
            curs = conn.cursor()
            curs.executemany("""
                             INSERT INTO Labels 
                             (Pfad, Label1, Label1_Wert, Label2, Label2_Wert, Label3, Label3_Wert, Label4, Label4_Wert, Label5, Label5_Wert)
                             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                             """, data)
            conn.commit()
            conn.close()
            data = []  # Liste leeren, damit es nicht zu dopplungen kommt
            print(i)
    # noch ein mal, sobald der generator durch gelaufen ist
    save_pickle(embeddings_list,filenames_list,i)
    embeddings_list = []
    filenames_list = []
    conn = sqlite3.connect("databases/Big_data_database.db")
    curs = conn.cursor()
    curs.executemany("""
                     INSERT INTO Labels 
                     (Pfad, Label1, Label1_Wert, Label2, Label2_Wert, Label3, Label3_Wert, Label4, Label4_Wert, Label5, Label5_Wert)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                     """, data)
    conn.commit()
    conn.close()
    data = []
    print(i)
    end_time = time.time() - start_time
    print(f"time: {end_time}   errors: {error_counter}")
        
#%%
if __name__ == '__main__':
    main() #aufpassen, dass nicht ausversehen gestartet wird
