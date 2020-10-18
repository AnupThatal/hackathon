from PIL import Image
import os
import integrate
import pickle
from flask import Flask, flash, request, redirect, url_for,render_template
from werkzeug.utils import secure_filename
from keras.preprocessing import image
import numpy as np



UPLOAD_FOLDER = '/path/to/the/uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

classifier = pickle.load(open("model.h5", 'rb'))
@app.route("/")
def index():
    return render_template('upload.html')


@app.route("/upload", methods=["GETS","POST"])
def upload():
    if request.method == 'POST':
      f = request.files['file']
      f.save(secure_filename(f.filename))
    return 'file uploaded successfully'
    # folder_name = request.form['superhero']
    # '''
    # # this is to verify that folder to upload to exists.
    # if os.path.isdir(os.path.join(APP_ROOT, 'files/{}'.format(folder_name))):
    #     print("folder exist")
    # '''
    # target = os.path.join(APP_ROOT, 'files/{}'.format(folder_name))
    # print(target)
    # if not os.path.isdir(target):
    #     os.mkdir(target)
    # print(request.files.getlist("file"))
    # for upload in request.files.getlist("file"):
    #     print(upload)
    #     print("{} is the file name".format(upload.filename))
    #     filename = upload.filename
    #     # This is to verify files are supported
    #     ext = os.path.splitext(filename)[1]
    #     if (ext == ".jpg") or (ext == ".png"):
    #         print("File supported moving on...")
    #     else:
    #         render_template("Error.html", message="Files uploaded are not supported...")
    #     destination = "/".join([target, filename])
    #     print("Accept incoming file:", filename)
    #     print("Save it to:", destination)
    #     upload.save(destination)

    # # return send_from_directory("images", filename, as_attachment=True)
    # return render_template("complete.html", image_name=filename)





@app.route('/upload/<filename>')
def send_image(filename):
    return send_from_directory("images", filename)


@app.route('/gallery')
def get_gallery():
    image_names = os.listdir('./images')
    print(image_names)
    return render_template("gallery.html", image_names=image_names)

def predict():
    test_image = image.load_img(f,target_size = (64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = classifier.predict(test_image)          
    return result

if __name__ == "__main__":
    app.run(port=4555, debug=True)
