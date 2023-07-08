# Image_recommender
## Inhaltsverzeichnis
- [Beschreibung](#Beschreibung)
- [Installation](#Installation)
- [Anwendung](#Anwendung)
- [Mitwirkende](#Mitwirkende)

## Beschreibung

## Installation
```shell
git clone https://github.com/Julimaeder/Image_recommender.git
cd Image_recommender
pip install -r dependencies.txt
```

## Anwendung
- Nach installation muss die Pickle Datei für die Embeddings in Image_recommender/Scripts/Data 
- und die Datenbanken für Labels und Scheme in Image_recommender/Scripts/Data/databases
- aus Image_recommender_Predictions.py muss in Zeile 12 & 15 das Scripts. vor den imports entfernt werden (Dient nur zu testzwecken)
- in Image_recommender.py muss der main() Funktion der Ordner mit den Bildern drin übergeben werden, von denen man die Ergebnisse wünscht
- --> so kann das Programm mit beliebig vielen Bildern genutzt werden. Die Ergebnisse werden nacheinander angezeigt
- Image_recommender.py ausführen

## Mitwirkende
- Julian Mäder - Github: @Julimaeder
- Shazil Khan - Github: @Bro-tec
