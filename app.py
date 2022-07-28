import numpy as np
from PIL import Image

import os 



from flask import Flask, jsonify

import pyrebase

import urllib

# offical firestore 
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

from tflite_runtime import interpreter as tflite





app = Flask(__name__)


interpreter = tflite.Interpreter(model_path="effnet.tflite")
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


#firebasecred
firebasecred = {
  "type": "service_account",
  "project_id": "brain-tumor-detector-6be52",
  "private_key_id": "dfd3f47967e13604bd6bc6354c39ab56e81be2c6",
  "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwggSkAgEAAoIBAQDaIFd4D34XSD/J\nxkzKgjqlOtfxSwd5K1DvDF9kds3x8VbJYweZ75EzgrWo1+0BXH/rHvWOuvXSSZ8R\nNqx6iqNIq76Y2pM4440tcsPsqJa5yQcVGADizxTn0bxJdo0qCSeTqmdp9QlsInBL\nxKZN3wg7sctXn7HZGSckVrxmfnR73q2uKnM+NDDVWQYHpP+srvV6Vfc8HTauY8Pf\neXtxACWL2QXGQ83brCmTkMFsrGciqu+8/ZqshFKSAZWgZ+e5KY+JVkHJWeJ0nrNT\nfGO1twk0T2tsjuT5Beaho33TzD69AGaskFROlnpaWt+wy2IHbNZGO/7kn4enBqXB\nhPBR2utlAgMBAAECggEABOdsq3BZKIVqdSaWLj9f3in7qFH7OwgrIoJw/OIZMPML\nPVIk8C1WYqpfpOWTmIbPk7hAd+50tGOfWDAD646FPdgqEtUeDRbOUOVD7mS/XHsV\naEBJC6vMxoYM6V1W97av1WLs0Np1MKF5AyQ4Z9BcJr21/zadWdx2DRv09q6ptg4M\nGu3BANymjE+Dg8hCF11mAxRpvENCh92v5dVeOayuIz7VAhvbN0585PmGbNAnLqDb\nvzJu3WOh83gdaTn0c46sL8XYfoxwrzlR8eSHmR/U8if4mJJa4s/D4Cj5/Lf2vfRV\n6pn/W0uPlgwvS4RiTjS3NIjex0dpRun77FmIvGmYSQKBgQDvUCePliwOk+mMwJic\nKYbQceIlw0l42RCUiT8et815cgYXTVSGKRUvcY5xSfSjI+u6aZKpcbnu4dn6ICcp\n44MLEauaBabWxDzajFLW7AxE8wMcYgXaB5B9ihuKoemXZB/1tyl4QpzaY1BmNPHt\neQQp9Q5xH1EwV5sckewWJSE8OwKBgQDpVf5u8SnxV0RQ8TVdX2eLqF9S3l5/+7Wd\nbNfQZ623SHeY4FYONaT9mhFosuyvG/qDmbpoQCopqiL8nU3NtIsC04wHpwr+ivOw\nNXtlx7sWHZh6CRRUwXESVPzhFEIPAyhYH8LAt9KaRk7lKGiaFPU1523PRXJiIAqw\ncZ1t3eMc3wKBgQDZUda7I8paaqO6N+PeXC+a9vBBDriXz7aozIHPaWZklNFHM+g/\n9OrSLLMH2fsYczRMEjcnPKl2bw69f5lRBtQnpyJIOj2p5obEiI1psu3pZy06ByH9\nPsVN267rE+HGoxwKzQwRs5wxDeMjDY1s82p+l5VH0QKvfb7UEQdtjMZDpQKBgQCz\nIgKwF1MU5eMbpOJMKbcrn3p2+yJfbNVT40CXzVCu+eJfKjLGu+ZLj2E4GMzd7kPX\njkhuSnxT+jrb5sPZXXavF8tUAKjPG8vThmuSitCVPOlXHutN2ig9Y6O0BEJmlgz+\nAnwYScdUCw/8m5YaXGaYGHDUBEO5E1JBfSfYNdVqzwKBgD4dkP0sfc0bcvRC+btH\nJS8jlcf2CJYBty+tXgApALRnlkWT/VjJsRk0Kla+235rCQb2x68mMlDwnsZ3R5ZS\nWcUOiSMcXPaZAaRsVdBd4G/AY2BlJbT+3DRSRxg+mlzSMP4PVh+MFErdqEEMpssX\niPbUYSro5sP9tLc81Y8TdpAK\n-----END PRIVATE KEY-----\n",
  "client_email": "firebase-adminsdk-rjegi@brain-tumor-detector-6be52.iam.gserviceaccount.com",
  "client_id": "111732052885284013437",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/firebase-adminsdk-rjegi%40brain-tumor-detector-6be52.iam.gserviceaccount.com" 
}
# credintials for firestore
cred = credentials.Certificate(firebasecred)
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
    
        

    
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_data_type = input_details[0]["dtype"]
    image = np.array(Image.open("gfg.jpg").resize((150, 150)), dtype=input_data_type)
    image=np.array(image)
    image = np.expand_dims(image, axis=0)

    print(input_details[0])
    interpreter.set_tensor(input_details[0]["index"], image)
    interpreter.invoke()
    tflite_interpreter_output = interpreter.get_tensor(output_details[0]["index"])
    print(tflite_interpreter_output )
    result = np.argmax(tflite_interpreter_output,axis=1)[0]
    print(result)

    if result==0:
      result = 'Glioma Tumor'
    elif result==1:
      result = "No Tumor"
    elif result==2:
      result = 'Meningioma Tumor'
    else:
      result = 'Pituitary Tumor'

    print(result)
    
    
    
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

    app.run(debug=False,host='0.0.0.0')








# getResult("C:\Users\richa\Downloads\BrainTumor Classification DL\uploads\y14.jpg")




