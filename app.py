from flask import Flask,render_template, request

#from tensorflow.keras.utils import load_img
#from keras.preprocessing.image import load_img
from keras_preprocessing.image import load_img
from keras_preprocessing.image import img_to_array
#import tensorflow as tf
from keras.applications.vgg16 import preprocess_input,decode_predictions,VGG16
from keras.models import load_model
from keras_preprocessing import image
#MODEL = tf.keras.models.load_model("1")
import numpy as np

app= Flask(__name__)
disease_dict={"white nail":"Possible Diseases - Jaundiceii, liver trouble,  Anemia",
             "yellow nails" :"Possible Diseases - lung disease ,diabetes ,psoriasis, thyroid disease",
             "pale nail" : "Possible Diseases - Anemia Congestive ,heart failure ,Liver disease, Malnutrition",
              "beau's lines" : "Possible Diseases -  systematic disease",
              "bluish nail" : "Possible Diseases -  heart problems , emphysema",
              "terry's nail" : "Possible Diseases -  . Hepatic failure , Cirrhosis, Diabetes, Mellitus, Congestive Heart failure , Hyperthyroidism.",
             }
model_p='nails.h5'

model=load_model(model_p)


@app.route('/', methods=['GET'])
def hello_world():
    return render_template('index.html')

@app.route('/',methods=['POST'])
def predict():
    imagefile=request.files['imagefile']
    image_path="./images/"+imagefile.filename
    imagefile.save(image_path)

    image=load_img(image_path,target_size=(256, 256))
    image=img_to_array(image)
    image=image.reshape(1,image.shape[0],image.shape[1],image.shape[2])
    image=preprocess_input(image)
    #yhat=model.predict(image)

    pred= model.predict(image)
    print(pred)
    itemindex = np.where(pred==np.max(pred))
    reverse_mapping = ["Darier's disease","Muehrck-e's lines","aloperia areata","beau's lines",
    "bluish nail", "clubbing" ,"eczema",
    "half and half nailes (Lindsay's nails)" ,"koilonychia", "leukonychia",
    'onycholycis', 'pale nail' ,'red lunula', 'splinter hemmorrage',
    "terry's nail" ,'white nail', 'yellow nails']
    prediction_name = reverse_mapping[itemindex[1][0]]
    print(prediction_name)
    value=np.max(pred)*100
    print(value)
    # if (value >= 95.0):
    #     itemindex = np.where(pred==np.max(pred))
    #     print("Final Diagonasis result : " +prediction_name)
    #     print("Probability of " + prediction_name + "  is: " +str(value))
    #     print(disease_dict[prediction_name])
    # else:
    #     print("No disease is identified " + str(value) +" ; nail type: "+prediction_name)


    

    return render_template('index.html', prediction_name=prediction_name, value=value)

if __name__=='__main__':
    app.run(port=3000, debug=True)    



    