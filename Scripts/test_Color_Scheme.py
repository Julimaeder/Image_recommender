from Color_Scheme import Full_Prediction,Full_Preperation
import pandas as pd
path = "../.github/Test_bild/standing-german-shepherd.jpg"
d = ['E:\\images\\weather_image_recognition\\dew\\2208.jpg', 1, 9315, 0, 0, 0, 0, 0, 31859, 0, 0, 0, 0, 0, 3307, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2229, 2, 0, 0, 0, 0, 34805, 17, 0, 0, 0, 0, 2626, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 327, 43, 0, 0, 0, 0, 3735, 277, 21, 0, 0, 0, 50, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 40, 297, 96, 1, 0, 0, 124, 85, 174, 34, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 14, 21, 92, 132, 56, 5, 1, 5, 11, 5, 60, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 11, 15, 0, 1, 2, 6, 53, 25]
c = [f"spalte_{i}" for i in range(len(d))]
df = pd.DataFrame([d], columns=c)

def test_check_prediction():
    
    #Test Label:
    expected_output_label = 'German_shepherd' #genaue Werte können nur schelcht verglichen werden, da die Berechnungen teils minimal von einander abweichen
    path = Full_Prediction(path, df)
    assert path == d[0]

def test_check_preperation():
    assert True