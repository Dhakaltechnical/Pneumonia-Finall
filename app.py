from flask import Flask, render_template, request, session, redirect, url_for, flash
import os
import cv2
import numpy as np
import tensorflow as tf
from werkzeug.security import generate_password_hash
from flask_mysqldb import MySQL
from tensorflow.keras.models import load_model

UPLOAD_FOLDER = './flask app/assets/images'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__, static_url_path='/assets',
            static_folder='./flask app/assets',
            template_folder='./flask app')
app.secret_key = os.urandom(24)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER



@app.route('/')
def root():
   return render_template('index.html')

# @app.route('/index.html')
# def index():
#    return render_template('index.html')

@app.route('/contact.html')
def contact():
   return render_template('contact.html')

@app.route('/news.html')
def news():
   return render_template('news.html')

@app.route('/about.html')
def about():
   return render_template('about.html')

@app.route('/faqs.html')
def faqs():
   return render_template('faqs.html')

@app.route('/prevention.html')
def prevention():
   return render_template('prevention.html')

@app.route('/upload.html')
def upload():
   return render_template('upload.html')

@app.route('/upload_chest.html')
def upload_chest():
   return render_template('upload_chest.html')



def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def is_chest_xray(image):
    height, width = image.shape[:2]
    aspect_ratio = width / height
    if height < 200 or width < 200 or not (1.0 < aspect_ratio < 1.5):
        return False

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_gray_value = np.mean(gray_image)
    
    if 0 < mean_gray_value < 160:
        return True
    return False




# Convolution operation (manual implementation)
def convolve2d(image, kernel):
    kernel_height, kernel_width = kernel.shape
    image_height, image_width = image.shape
    output = np.zeros((image_height - kernel_height + 1, image_width - kernel_width + 1))
    
 # Mathematical representation of convolution
    for y in range(output.shape[0]):
        for x in range(output.shape[1]):
            output[y, x] = np.sum(image[y:y + kernel_height, x:x + kernel_width] * kernel)
    return output

# ReLU activation function
def relu(feature_map):
    # Mathematical representation: A(x) = max(0, F(x, y))
    return np.maximum(0, feature_map)

# Max pooling operation
def max_pooling(feature_map, size=2):
    pool_height, pool_width = feature_map.shape
    output_height = pool_height // size
    output_width = pool_width // size
    output = np.zeros((output_height, output_width))

    for y in range(0, pool_height, size):
        for x in range(0, pool_width, size):
            output[y // size, x // size] = np.max(feature_map[y:y + size, x:x + size])
    return output

@app.route('/uploaded_chest', methods=['POST', 'GET'])
def uploaded_chest():
   

    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'upload_chest.jpg')
            file.save(file_path)

            image = cv2.imread(file_path)
            if not is_chest_xray(image):
                flash('Uploaded image does not appear to be a chest X-ray. Please upload a valid X-ray image.')
                return render_template('upload_chest.html', prediction="Uploaded image does not appear to be a chest X-ray.")
            
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
            image_gray = cv2.resize(image_gray, (500, 500))  # Resize

            # Example kernel (3x3 edge detection kernel)
            kernel = np.array([[1, 0, -1],
                               [1, 0, -1],
                               [1, 0, -1]])

            # Manual convolution operation
            convolved_image = convolve2d(image_gray, kernel)
            convolved_image = relu(convolved_image)  # Apply ReLU
            convolved_image = max_pooling(convolved_image)  # Max pooling   

            # For simplicity, here we just continue with model prediction
            image_resized = cv2.resize(image_gray, (500, 500))  # Resize for model
            image_resized = np.expand_dims(image_resized, axis=-1)  # Add channel dimension
            image_resized = np.expand_dims(image_resized, axis=0)  # Add batch dimension
            image_resized = image_resized.astype('float32') / 255.0  # Normalize pixel values

            # Load model and make prediction
            cnn_chest = load_model('models/pneu_cnn_model.h5')
            cnn_pred = cnn_chest.predict(image_resized)
            probability = cnn_pred[0]

            # Set a confidence threshold
            confidence_threshold = 0.5
            if probability[0] > confidence_threshold:
                cnn_chest_pred = 'Uploaded image is detected PNEUMONIA'
            else:
                cnn_chest_pred = 'Uploaded image is detected NORMAL'
            print(cnn_chest_pred)

            
            return render_template('upload_chest.html', prediction=cnn_chest_pred)

        else:
            flash('File type not allowed. Please upload a valid image (png, jpg, jpeg).')
            return redirect(request.url)

    return render_template('upload_chest.html') 



# MySQL Configuration
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'users'

mysql = MySQL(app)

@app.context_processor
def inject_logged_in():
    return dict(logged_in='username' in session, username=session.get('username'))
@app.route('/index.html')
def index():
    return render_template('index.html')  

@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        pwd = request.form['password']
        cur = mysql.connection.cursor()
        cur.execute(f"select username, password from tbl_users where username = '{username}'")
        user = cur.fetchone()
        cur.close()
        if user and pwd == user[1]:
            session['username'] = user[0]
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error='Invalid username or password')
    return render_template('login.html')

@app.route('/register' ,methods=['GET','POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        contact = request.form['contact']
        pwd = request.form['password']


        cur = mysql.connection.cursor()
        cur.execute("INSERT INTO tbl_users (username, email, contact, password) VALUES (%s, %s, %s, %s)", 
                    (username, email, contact, pwd))
        mysql.connection.commit()
        cur.close()

        return redirect(url_for('login'))
    
    return render_template('register.html')
@app.route('/logout')
def logout():
    session.pop('username',None)
    return redirect (url_for('index'))

@app.route('/history')
def history():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    # Example history data, replace with actual database queries
    history = [
        {'image_url': 'assets/images/upload_chest.jpg', 'prediction': 'Pneumonia'},
        {'image_url': 'assets/images/upload_chest.jpg', 'prediction': 'No Pneumonia'},
        # Add more records as needed
    ]

    return render_template('history.html', history=history)


if __name__ == '__main__':
    app.debug = True
    app.run(port=5001)
