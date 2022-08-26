
from flask import Flask, render_template, request, session, redirect, url_for, flash
import os
from werkzeug.utils import secure_filename
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import numpy as np

Image_folder = './gui/assets/Images/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
# Create Database if it doesnt exist

app = Flask(__name__,static_url_path='/assets',
            static_folder='./gui/assets', 
            template_folder='./gui')
app.config['Image_folder'] = Image_folder


@app.route('/')
def root():
   return render_template('index.html')

@app.route('/index.html')
def index():
   return render_template('index.html')

@app.route('/graph.html')
def graph():
   return render_template('graph.html')





@app.route('/prevention.html')
def prevention():
   return render_template('prevention.html')

@app.route('/upload.html')
def upload():
   return render_template('upload.html')

@app.route('/upload_chest.html')
def upload_chest():
   return render_template('upload_chest.html')



@app.route('/uploaded_chest', methods = ['POST', 'GET'])
def uploaded_chest():
   if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            # filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['Image_folder'], 'test_image.jpg'))

  
   cnn_model = tf.keras.models.load_model('F:/Project/saved_models/cnn5_model.h5')

   image = cv2.imread('/gui/assets/images/test_image.jpg') # read file 
   image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # arrange format as per keras
   image = cv2.resize(image,(64,64))
   image = np.array(image) / 255
   image = np.expand_dims(image, axis=0)
   


   cnn_pred = cnn_model.predict(image)
   probability = cnn_pred[0]
   print("CNN Predictions:")
   if probability[0] > 0.5:
      cnn_pred = str('There is '+'%.2f' % (probability[0]*100) + '% chance of having TUMOR') 
   else:
      cnn_pred = str('There is '+'%.2f' % ((1-probability[0])*100) + '% chance of NOT having TUMOR')
   print(cnn_pred)

   

   return render_template('results_chest.html',cnn_pred=cnn_pred)



if __name__ == '__main__':
   app.secret_key = ".."
   app.run(debug=True,port=8000)