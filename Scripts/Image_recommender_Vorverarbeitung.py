"""
Infos:
Im non-live code: Die erste Progress Bar ist extract_image_embeddings und die zweite label_image
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


#%%
os.chdir('C:\\Users\\maede\\Desktop\\Big_Data\\Test_Dateien\\final_test')
#%%
# Überprüfen Sie, ob der Ordner 'databases' nicht existiert
if not os.path.exists('databases'):
    # Erstellen Sie den Ordner
    os.makedirs('databases')

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

#%%
logging.basicConfig(filename='Logging_test.log', level=logging.ERROR)
#%%
folder_path = 'C:\\Users\\maede\\Desktop\\Big_Data\\train'#'E:\\images'
model = MobileNetV2(weights='imagenet')

# Schritt 3: Lade und komprimiere die Bilder
def load_and_compress_image(image_path, target_size):
    image = Image.open(image_path)
    image = image.resize(target_size)
    image_array = np.array(image)
    preprocessed_image = tf.keras.applications.mobilenet.preprocess_input(image_array)
    return preprocessed_image

# Schritt 4: Extrahiere Bild-Einbettungen
def extract_image_embeddings(image):
    image = np.expand_dims(image, axis=0)
    embeddings = model.predict(image)
    return embeddings.flatten()

def image_generator(folder_path):
    global dirpath,filename
    image_extensions = ('.jpg', '.png', '.jpeg')
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for filename in filenames:
            if filename.lower().endswith(image_extensions):
                yield os.path.join(dirpath, filename)


def save_checkpoint(data, filename='checkpoint.txt'):
    with open(filename, 'w') as f:
        f.write(str(data))

# load checkpoint
def load_checkpoint(filename='checkpoint.txt'):
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return int(f.read())
    else:
        return None
    

def label_image(model, compressed_image, target_size=(224, 224), top_labels=1):
    image = compressed_image
    if len(image.shape) == 3:
        image = np.expand_dims(image, axis=0)
    predictions = model.predict(image)
    labels = decode_predictions(predictions, top=top_labels)
    return labels[0]
    
def save_pickle(embeddings_list,filenames_list,i):
    embeddings_df = pd.DataFrame(embeddings_list, index=filenames_list)
    
    pickle_filename = 'Pickle_embeddings_test.pkl'
    save_checkpoint(i)  # nur alle 500, weil checkpoint sonst ggf weiter ist als das Speichern 
                        #--> dann würden bis zu 499 Bilder verloren gehen
    
    if os.path.exists(pickle_filename):
        with open(pickle_filename, "rb"):
            existing_data = pd.read_pickle(pickle_filename)
        embeddings_df = pd.concat([existing_data, embeddings_df])

    with open(pickle_filename, "wb"):
        pd.to_pickle(embeddings_df, pickle_filename)
        

#%%

def main():
    global i
    batch_size = 500
    embeddings_list = []
    filenames_list = []
    data = []
    error_counter = 0
    start_time = time.time()
    generator = image_generator(folder_path)
    checkpoint = load_checkpoint()
    i = 0
    start = checkpoint + 1 if checkpoint is not None else 0
    for i in tqdm(range(0,start)):
        next(generator)

    for image_path in generator:
        try:  
            temp2 = load_and_compress_image(image_path, target_size=(224, 224))
            embeddings = extract_image_embeddings(temp2)
            filenames_list.append(image_path)
            embeddings_list.append(embeddings)
            labels = label_image(model, temp2, target_size=(224, 224), top_labels=5)
            row = [image_path]
            for label in labels:
                row.append(label[1]) # Das sind die Label Namen, Label[0] ist die cryptische Bezeichnung der Label
                row.append(label[2]) # Wert des Labels
            data.append(tuple(row))
            i +=1
            
        except Exception as e:
            logging.error(f"Error at point {i}: {e}")
            error_counter = error_counter +1
            continue
        if i % batch_size == 0:
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
            data = []  # Leeren Sie die Datenliste für die nächste Charge
            print(i)
    # noch ein mal, sobald der generator durch gelaufen ist
    save_pickle(embeddings_list,filenames_list,i)
    embeddings_list = []
    filenames_list = []
    print(i)
        
    end_time = time.time() - start_time
    print(f"time: {end_time}   errors: {error_counter}")
        

#%%
if __name__ == '__main__':
    main() #aufpassen, dass nicht ausversehen gestartet wird
