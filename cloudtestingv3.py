import numpy as np
from PIL import Image
from keras.models import load_model


from flask import Flask, jsonify

import pyrebase

import urllib

# offical firestore 
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore



app = Flask(__name__)


model =load_model('effnet.h5')
print('Model loaded.')


config = {
  "apiKey": "AIzaSyBaBdap1BjftQiRdniZ5AQ6jkqcAfHu2SQ",
  "authDomain": "brain-tumor-detector-6be52.firebaseapp.com",
  "projectId": "brain-tumor-detector-6be52",
  "storageBucket": "brain-tumor-detector-6be52.appspot.com",
  "messagingSenderId": "130383918562",
  "appId": "1:130383918562:web:9cce14aecd9c35a1a8f5bb",
  "databaseURL": ""
}

firebase = pyrebase.initialize_app(config)
storage = firebase.storage()

path_on_cloud = "images/sample.jpg"
path_local = "images/toupload.jpg"

# credintials for firestore
cred = credentials.Certificate("brain-tumor-detector-6be52-firebase-adminsdk-rjegi-dfd3f47967.json")
firebase_admin.initialize_app(cred)


def firebase_upload(path_on_cloud,path_local,storage):
    storage.child(path_on_cloud).put(path_local)

def firebase_get_image_url(path_on_cloud,storage):
    url = storage.child(path_on_cloud).get_url(None)
    if url:
        print(url)
        getResultV2(url)
        return url
    else:
        return "url is empty"


def send_result_to_firebase(result):
    db = firestore.client()
    doc_ref = db.collection(u'result').document(u'output')
    doc_ref.set({
        u'predicationresult': u'' + str(result),
    })
    # db.child("result").child("output").update({"predicationresult": "str(result)" })


# send_result_to_firebase()
    


# def getResult(url):
    
#     print(f"url : {url}")

    
#     # url_response = urllib.request.urlopen(url)
#     # img_array = np.array(bytearray(url_response.read()), dtype=np.uint8)
#     # img = cv2.imdecode(img_array, -1)
#     # # cv2.imshow('URL Image', img)

#     # image=cv2.imread(img)
#     # image = Image.fromarray(image, 'RGB')

#     # urllib.request.urlretrieve(url,"gfg.png")
  
#     image = Image.open("image(1).jpg")
#     # image.show()
#     image = image.resize((150, 150))
#     image=np.array(image)
#     input_img = np.expand_dims(image, axis=0)
#     result=model.predict(input_img)
#     result = np.argmax(result,axis=1)[0]
#     print(result)
#     # Removing the brackets [[1]]
#     print(np.ndarray.item(result))
#     result = np.ndarray.item(result)

#     if result == 1.0:
#         result = "Positive"
#     else:
#         result = "Negative"
    
    
#     # send_result_to_firebase(result)
#     # return result




def getResultV2(url):
    
    print(f"url : {url}")
    
    # url_response = urllib.request.urlopen(url)
    # img_array = np.array(bytearray(url_response.read()), dtype=np.uint8)
    # img = cv2.imdecode(img_array, -1)
    # # cv2.imshow('URL Image', img)

    # image=cv2.imread(img)
    # image = Image.fromarray(image, 'RGB')

    urllib.request.urlretrieve(url,"gfg.jpg")
  
    image = Image.open("gfg.jpg")
    # image.show()
    image = image.resize((150, 150))
    image=np.array(image)
    input_img = np.expand_dims(image, axis=0)
    result=model.predict(input_img)
    print("Predict result : ", result)
    result = np.argmax(result,axis=1)[0]
    print(result)

    if result==0:
        result = 'Glioma Tumor'
    elif result==1:
        result = "No Tumor"
    elif result==2:
        result = 'Meningioma Tumor'
    else:
        result = 'Pituitary Tumor'
    
    
    send_result_to_firebase(result)
    return result
    



@app.route('/',methods=['POST','GET'])
def index():    
    response = jsonify({'some': "data"})
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response
     
# for testing purpose only 
@app.route('/classv2',methods=['POST','GET'])
def classv2():    
    getResultV2()
    return "All OK"



@app.route('/modelpredict/', methods=['POST','GET'])
def modelpredict():
    storage = firebase.storage()
    # img = request.args['imgurl']
    # its not a url but the location of the image images/y14.jpg
    # imgurl = request.values["imgurl"]

    # print(type(imgurl))
    # print(imgurl)
    url =  firebase_get_image_url(path_on_cloud,storage)
    # url = ""
    
    if url:
        response = jsonify({"imageurl": url,"responseStatus": "200"})
    else:
        response = jsonify({"imageurl": "empty","resonseStatus": "500"})
    
    response.headers.add('Access-Control-Allow-Origin', '*')
    
    return response


# firebase_upload(path_on_cloud,path_local,storage)
# firebase_get_image_url(path_on_cloud,storage)









if __name__ == '__main__':
    app.run()


# getResult("C:\Users\richa\Downloads\BrainTumor Classification DL\uploads\y14.jpg")





