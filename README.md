# Image_recommender
## Inhaltsverzeichnis
- [Beschreibung](#Beschreibung)
- [Installation](#Installation)
- [Anwendung](#Anwendung)
- [Mitwirkende](#Mitwirkende)

## Beschreibung

Image_recommender ist dazu dar um das ähnlichste Bild vorzuschlagen, zu den Bildern, welche vorhergesagt werden sollen.
Dafür kann eine eigene, oder eine heruntergelade Bildersammlung auf der Festplatte mit dem Buchstaben E verwendet werden.
Zu den Bildern werden die jeweiligen datenbanken und die pickeldatei benötigt, Diese wurden entweder im vorhinein erstellt, oder mit Image_recommender selbst eingeladen.
Wenn dir deine Bildersammlung zu groß ist, du aber ähnliche Bilder zu einem bestimmten Bild benötigst, kannst du Diese mit dem Image recommender finden. 


## Installation
```shell
git clone https://github.com/Julimaeder/Image_recommender.git
cd Image_recommender
pip install -r dependencies.txt
```

## Anwendung
- Nach installation muss die Pickle Datei für die Embeddings in Image_recommender/Scripts/Data 
- und die Datenbanken für Labels und Scheme in Image_recommender/Scripts/Data/databases
- in Image_recommender.py muss der main() Funktion der Ordner mit den Bildern drin übergeben werden, von denen man die Ergebnisse wünscht
- --> so kann das Programm mit beliebig vielen Bildern genutzt werden. Die Ergebnisse werden nacheinander angezeigt
- Image_recommender.py ausführen

## Mitwirkende
- Julian Mäder - Github: @Julimaeder
- Shazil Khan - Github: @Bro-tec
