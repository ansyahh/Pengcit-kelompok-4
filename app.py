import heapq
from io import BytesIO
from flask import Flask, redirect, render_template, request, url_for, send_file
import numpy as np
from werkzeug.utils import secure_filename
import os
import matplotlib.pyplot as plt
import cv2
import rembg
from collections import defaultdict

app = Flask(__name__)

upload_folder = os.path.join('static', 'uploads')

app.config['UPLOAD'] = upload_folder


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/histogram', methods=['GET', 'POST'])
def histogram_equ():
    if request.method == 'POST':
        file = request.files['img']
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD'], filename))
        img_path = os.path.join(app.config['UPLOAD'], filename)

        # Membaca gambar dengan OpenCV
        img = cv2.imread(img_path)

        # Menghitung histogram untuk masing-masing saluran (R, G, B)
        hist_r = cv2.calcHist([img], [0], None, [256], [0, 256])
        hist_g = cv2.calcHist([img], [1], None, [256], [0, 256])
        hist_b = cv2.calcHist([img], [2], None, [256], [0, 256])

        # Normalisasi histogram
        hist_r /= hist_r.sum()
        hist_g /= hist_g.sum()
        hist_b /= hist_b.sum()

        # Simpan histogram sebagai gambar PNG
        hist_image_path = os.path.join(app.config['UPLOAD'], 'histogram.png')
        plt.figure()
        plt.title("RGB Histogram")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.plot(hist_r, color='red', label='Red')
        plt.plot(hist_g, color='green', label='Green')
        plt.plot(hist_b, color='blue', label='Blue')
        plt.legend()
        plt.savefig(hist_image_path)

        # Hasil equalisasi
        # Ubah ke ruang warna YCrCb
        img_equalized = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        img_equalized[:, :, 0] = cv2.equalizeHist(
            img_equalized[:, :, 0])  # Equalisasi komponen Y (luminance)
        # Kembalikan ke ruang warna BGR
        img_equalized = cv2.cvtColor(img_equalized, cv2.COLOR_YCrCb2BGR)

        # Menyimpan gambar hasil equalisasi ke folder "static/uploads"
        equalized_image_path = os.path.join(
            'static', 'uploads', 'img-equalized.jpg')
        cv2.imwrite(equalized_image_path, img_equalized)

        # Menghitung histogram untuk gambar yang sudah diequalisasi
        hist_equalized_r = cv2.calcHist(
            [img_equalized], [0], None, [256], [0, 256])
        hist_equalized_g = cv2.calcHist(
            [img_equalized], [1], None, [256], [0, 256])
        hist_equalized_b = cv2.calcHist(
            [img_equalized], [2], None, [256], [0, 256])

        # Normalisasi histogram
        hist_equalized_r /= hist_equalized_r.sum()
        hist_equalized_g /= hist_equalized_g.sum()
        hist_equalized_b /= hist_equalized_b.sum()

        # Simpan histogram hasil equalisasi sebagai gambar PNG
        hist_equalized_image_path = os.path.join(
            app.config['UPLOAD'], 'histogram_equalized.png')
        plt.figure()
        plt.title("RGB Histogram (Equalized)")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.plot(hist_equalized_r, color='red', label='Red')
        plt.plot(hist_equalized_g, color='green', label='Green')
        plt.plot(hist_equalized_b, color='blue', label='Blue')
        plt.legend()
        plt.savefig(hist_equalized_image_path)

        return render_template('histogram_equalization.html', img=img_path, img2=equalized_image_path, histogram=hist_image_path, histogram2=hist_equalized_image_path)

    return render_template('histogram_equalization.html')


def edge_detection(img):
    # Menerapkan deteksi tepi menggunakan algoritma Canny
    edges = cv2.Canny(img, 100, 200)

    # Menyimpan gambar hasil deteksi tepi ke folder "static/uploads"
    edge_image_path = os.path.join(app.config['UPLOAD'], 'edge_detected.jpg')
    cv2.imwrite(edge_image_path, edges)

    return edge_image_path


@app.route('/edge', methods=['GET', 'POST'])
def edge():
    if request.method == 'POST':
        file = request.files['img']
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD'], filename))
        img_path = os.path.join(app.config['UPLOAD'], filename)

        # Membaca gambar dengan OpenCV
        img = cv2.imread(img_path)

        # Memanggil fungsi edge_detection
        edge_image_path = edge_detection(img)

        return render_template('edge.html', img=img_path, edge=edge_image_path)

    return render_template('edge.html')


def blur_faces(image_path, blur_level):
    # Membaca gambar dengan OpenCV
    img = cv2.imread(image_path)

    # Menggunakan Cascade Classifier untuk mendeteksi wajah
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Menerapkan deteksi wajah dengan parameter yang diatur
    faces = face_cascade.detectMultiScale(
        gray_img, scaleFactor=1.1, minNeighbors=5, minSize=[30, 30])

    # Menerapkan efek blur ke setiap wajah yang terdeteksi
    for (x, y, w, h) in faces:
        # Ambil bagian wajah dari gambar
        face = img[y:y+h, x:x+w]
        # Hitung ukuran kernel berdasarkan tingkat blur yang diatur
        kernel_size = (blur_level, blur_level)
        # Terapkan efek blur Gaussian dengan kernel yang sesuai
        blurred_face = cv2.GaussianBlur(face, kernel_size, 0)
        img[y:y+h, x:x+w] = blurred_face

    # Menyimpan gambar dengan wajah-wajah yang telah di-blur
    blurred_image_path = os.path.join(
        app.config['UPLOAD'], 'blurred_image.jpg')
    cv2.imwrite(blurred_image_path, img)

    return blurred_image_path


@app.route('/faceBlur', methods=['GET', 'POST'])
def face_blur():
    error = None
    if request.method == 'POST':
        # Check if the 'img' file is in the request
        if 'img' not in request.files:
            error = 'Please Select a Picture'
            return render_template('blur.html', error=error)

        file = request.files['img']

        # Check if the file name is empty
        if file.filename == '':
            error = 'Please Select a Picture'
            return render_template('blur.html', error=error)

        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD'], filename))
        image_path = os.path.join(app.config['UPLOAD'], filename)

        # Get blur level from the form
        blur_level = int(request.form.get('tingkatan', 1))

        # Call the function to blur faces
        blurred_image_path = blur_faces(image_path, blur_level)

        return render_template('blur.html', img=image_path, img2=blurred_image_path)

    return render_template('blur.html')


def cartoonize_image(image_path, cartoonized_image_path):
    # Baca gambar
    img = cv2.imread(image_path)

    # Konversi gambar ke grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)

    # Deteksi tepi dengan menggunakan operator Canny
    edges = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
    # Dilasi pada gambar tepi
    # edges = cv2.dilate(edges, None, iterations=2)

    # Thresholding gambar untuk mendapatkan gambar kartun
    color = cv2.bilateralFilter(img, 9, 250, 250)
    cartoon = cv2.bitwise_and(color, color, mask=edges)

    # Simpan gambar kartun ke path yang sesuai
    cv2.imwrite(cartoonized_image_path, cartoon)

    # Kembalikan path file gambar kartun
    return cartoonized_image_path


@app.route('/background_remove', methods=['GET', 'POST'])
def background_remove():
    if request.method == 'POST':
        if 'img' not in request.files:
            return render_template('background_remove.html', error='No image file uploaded')

        img_file = request.files['img']

        if img_file.filename == '':
            return render_template('background_remove.html', error='No selected image')

        if img_file:
            input_image_path = os.path.join(
                app.config['UPLOAD'], 'background_input.png')
            output_image_path = os.path.join(
                app.config['UPLOAD'], 'background_output.png')

            img_file.save(input_image_path)
            remove_background(input_image_path, output_image_path)

            return render_template('background_remove.html', input_img=input_image_path, output_img=output_image_path)

    return render_template('background_remove.html')


def remove_background(input_image_path, output_image_path):
    with open(input_image_path, 'rb') as input_file:
        input_data = input_file.read()
        output_data = rembg.remove(input_data)

        with open(output_image_path, 'wb') as output_file:
            output_file.write(output_data)

# Route untuk halaman kartunize


@app.route('/cartoonize', methods=['GET', 'POST'])
def cartoonize():
    if request.method == 'POST':
        file = request.files['img']
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD'], filename)
        file.save(file_path)

        # Memproses gambar menjadi greyscale
        img = cv2.imread(file_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        gray_image_path = os.path.join(app.config['UPLOAD'], 'greyscale.jpg')
        cv2.imwrite(gray_image_path, gray)

        # Melakukan deteksi tepi menggunakan operator Threshold
        edges = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
        edges_image_path = os.path.join(app.config['UPLOAD'], 'edges.jpg')
        cv2.imwrite(edges_image_path, edges)

        # Mengaplikasikan efek kartun dengan bitwise_and
        color = cv2.bilateralFilter(img, 9, 250, 250)
        cartoon = cv2.bitwise_and(color, color, mask=edges)
        cartoon_image_path = os.path.join(
            app.config['UPLOAD'], 'cartoon_image.jpg')
        cv2.imwrite(cartoon_image_path, cartoon)

        # Mengirim semua hasil gambar ke halaman HTML
        return render_template('cartoonize.html',
                               original=file_path,
                               gray=gray_image_path,
                               edges=edges_image_path,
                               cartoon=cartoon_image_path)
    return render_template('cartoonize.html')


# Tambahan
def detect_and_apply_sticker(image_path):
    # Baca gambar
    img = cv2.imread(image_path)

    # Inisialisasi detektor wajah
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Deteksi wajah
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Loop melalui setiap wajah yang terdeteksi
    for (x, y, w, h) in faces:
        # Baca gambar kacamata
        sticker = cv2.imread('kacamata.png', -1)

        # Sesuaikan ukuran kacamata dengan wajah
        sticker = cv2.resize(sticker, (w, h))

        # Tempelkan kacamata pada wajah
        for i in range(sticker.shape[0]):
            for j in range(sticker.shape[1]):
                if sticker[i, j, 3] != 0:
                    img[y+i, x+j] = sticker[i, j, 0:3]

    return img


@app.route('/filter_face', methods=['GET', 'POST'])
def filter_face():
    original_file = None
    filtered_file = None

    if request.method == "POST":
        file = request.files['img']
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD'], filename)
        file.save(file_path)

        # Deteksi wajah dan tempelkan kacamata
        result_image = detect_and_apply_sticker(file_path)
        filter_path = os.path.join(app.config['UPLOAD'], 'filtered.jpg')
        cv2.imwrite(filter_path, result_image)

        # original_file = os.path.join(app.config['UPLOAD'], filename)

        return render_template('filter_face.html', original=file_path, filtered=filter_path)
    return render_template('filter_face.html')


@app.route('/erosion', methods=['GET', 'POST'])
def erosion():

    original_file = None
    filtered_file = None

    if request.method == "POST":
        from io import BytesIO
        file = request.files['img']
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD'], filename)
        file.save(file_path)

        img = cv2.imread(file_path, 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        kernel = np.ones((5, 5), 'uint8')
        erosion = cv2.erode(gray, kernel, iterations=1)

        # Deteksi wajah dan tempelkan kacamata
        result_image = BytesIO()
        plt.imsave(result_image, erosion, format='jpg', cmap=plt.cm.gray)
        result_image.seek(0)
        filter_path = os.path.join(app.config['UPLOAD'], 'erosion.jpg')
        with open(os.path.join(app.config['UPLOAD'], 'erosion.jpg'), 'wb') as f:
            f.write(result_image.read())

        return render_template('erosion.html', original=file_path, erosion=filter_path)

    return render_template('erosion.html')


@app.route('/dilatation', methods=['GET', 'POST'])
def dilatation():

    original_file = None
    filtered_file = None

    if request.method == "POST":
        from io import BytesIO
        file = request.files['img']
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD'], filename)
        file.save(file_path)

        img = cv2.imread(file_path, 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        kernel = np.ones((5, 5), 'uint8')
        dilate_img = cv2.dilate(gray, kernel, iterations=1)

        # Deteksi wajah dan tempelkan kacamata
        result_image = BytesIO()
        plt.imsave(result_image, dilate_img, format='jpg', cmap=plt.cm.gray)
        result_image.seek(0)
        filter_path = os.path.join(app.config['UPLOAD'], 'dilatation.jpg')
        with open(os.path.join(app.config['UPLOAD'], 'dilatation.jpg'), 'wb') as f:
            f.write(result_image.read())

        return render_template('dilatation.html', original=file_path, dilatation=filter_path)

    return render_template('dilatation.html')


@app.route('/opening', methods=['GET', 'POST'])
def opening():

    original_file = None
    filtered_file = None

    if request.method == "POST":
        from io import BytesIO
        file = request.files['img']
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD'], filename)
        file.save(file_path)

        img = cv2.imread(file_path, 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
        opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)

        # Deteksi wajah dan tempelkan kacamata
        result_image = BytesIO()
        plt.imsave(result_image, opening, format='jpg', cmap=plt.cm.gray)
        result_image.seek(0)
        filter_path = os.path.join(app.config['UPLOAD'], 'opening.jpg')
        with open(os.path.join(app.config['UPLOAD'], 'opening.jpg'), 'wb') as f:
            f.write(result_image.read())

        return render_template('opening.html', original=file_path, opening=filter_path)

    return render_template('opening.html')


@app.route('/closing', methods=['GET', 'POST'])
def closing():

    original_file = None
    filtered_file = None

    if request.method == "POST":
        from io import BytesIO
        file = request.files['img']
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD'], filename)
        file.save(file_path)

        img = cv2.imread(file_path, 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
        closing = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

        # Deteksi wajah dan tempelkan kacamata
        result_image = BytesIO()
        plt.imsave(result_image, closing, format='jpg', cmap=plt.cm.gray)
        result_image.seek(0)
        filter_path = os.path.join(app.config['UPLOAD'], 'closing.jpg')
        with open(os.path.join(app.config['UPLOAD'], 'closing.jpg'), 'wb') as f:
            f.write(result_image.read())

        return render_template('closing.html', original=file_path, closing=filter_path)

    return render_template('closing.html')


@app.route('/rotation', methods=['GET', 'POST'])
def rotation():
    original_file = None
    filtered_file = None

    if request.method == "POST":
        # from io import BytesIO
        file = request.files['img']
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD'], filename)
        file.save(file_path)

        img = cv2.imread(file_path, 1)
        # gray = cv2.cvtColor(img, 1)
        height, width = img.shape[:2]
        degrees = int(request.form['degrees'])
        if degrees in [225, 270]:
            new_height = width
            new_width = height
            center = (height/2, width/3)
            M = cv2.getRotationMatrix2D(center, degrees, 1.0)
            rotated = cv2.warpAffine(img, M, (new_width, new_height))
        elif degrees in [90, 135]:
            new_height = width
            new_width = height
            center = (height/1.335, width/2)
            M = cv2.getRotationMatrix2D(center, degrees, 1.0)
            rotated = cv2.warpAffine(img, M, (new_width, new_height))
        else:
            new_height = height
            new_width = width

            center = (new_width / 2, new_height / 2)
            M = cv2.getRotationMatrix2D(center, degrees, 1.0)
            rotated = cv2.warpAffine(img, M, (new_width, new_height))

        # Deteksi wajah dan tempelkan kacamata

        rotated_path = os.path.join(app.config['UPLOAD'], 'rotate.png')
        cv2.imwrite(rotated_path, rotated)
        return render_template('rotation.html', original=file_path, rotation=rotated_path)
    return render_template('rotation.html')


@app.route('/scale', methods=['GET', 'POST'])
def scale():
    original_file = None
    filtered_file = None

    if request.method == "POST":
        # from io import BytesIO
        file = request.files['img']
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD'], filename)
        file.save(file_path)
        percentage = int(request.form['size_value'])

        img = cv2.imread(file_path, 1)
        # gray = cv2.cvtColor(img, 1)

        height, width = img.shape[:2]

        # Hitung ukuran baru berdasarkan persentase
        new_height = int(height * (percentage / 100))
        new_width = int(width * (percentage / 100))

        # Resize gambar
        resized = cv2.resize(img, (new_width, new_height))

        # Deteksi wajah dan tempelkan kacamata

        resized_path = os.path.join(app.config['UPLOAD'], 'resized.png')
        cv2.imwrite(resized_path, resized)
        return render_template('scale.html', original=file_path, scale=resized_path, width=width/5, height=height/5, new_height=new_height/5, new_width=new_width/5)
    return render_template('scale.html')

@app.route('/bilinear', methods=['GET', 'POST'])
def bilinear():
    original_file = None
    filtered_file = None

    if request.method == "POST":
        file = request.files['img']
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD'], filename)
        file.save(file_path)

        img = cv2.imread(file_path, 1)
        percentage = int(request.form.get('size_value'))
        if percentage < 0:
            percentage = 0
        elif percentage > 200:
            percentage = 200

        # Hitung ukuran baru berdasarkan persentase
        new_height = int(img.shape[0] * (percentage / 100))
        new_width = int(img.shape[1] * (percentage / 100))

        # Resize gambar dengan interpolasi linear
        bilinear = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

        # Simpan gambar hasil perubahan skala
        bilinear_path = os.path.join(app.config['UPLOAD'], 'bilinear.png')
        cv2.imwrite(bilinear_path, bilinear)

        return render_template('bilinear.html', original=file_path, bilinear=bilinear_path)

    return render_template('bilinear.html')

@app.route('/bicubic', methods=['GET', 'POST'])
def bicubic():
    original_file = None
    filtered_file = None

    if request.method == "POST":
        file = request.files['img']
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD'], filename)
        file.save(file_path)

        img = cv2.imread(file_path, 1)
        percentage = int(request.form.get('size_value'))
        if percentage < 0:
            percentage = 0
        elif percentage > 200:
            percentage = 200

        # Hitung ukuran baru berdasarkan persentase
        new_height = int(img.shape[0] * (percentage / 100))
        new_width = int(img.shape[1] * (percentage / 100))

        # Resize gambar dengan interpolasi bicubic
        bicubic = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

        # Simpan gambar hasil perubahan skala
        bicubic_path = os.path.join(app.config['UPLOAD'], 'bicubic.png')
        cv2.imwrite(bicubic_path, bicubic)

        return render_template('bicubic.html', original=file_path, bicubic=bicubic_path)

    return render_template('bicubic.html')

@app.route('/lowpass', methods=['GET', 'POST'])
def lowpass():

    original_file = None
    filtered_file = None

    if request.method == "POST":
        file = request.files['img']
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD'], filename)
        file.save(file_path)

        img = cv2.imread(file_path, 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Ganti kernel dengan Gaussian blur
        # Anda dapat menyesuaikan nilai (5, 5) sesuai kebutuhan
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        result_image = BytesIO()
        plt.imsave(result_image, blurred, format='jpg', cmap=plt.cm.gray)
        result_image.seek(0)
        filter_path = os.path.join(app.config['UPLOAD'], 'lowpass.jpg')
        with open(os.path.join(app.config['UPLOAD'], 'lowpass.jpg'), 'wb') as f:
            f.write(result_image.read())

        return render_template('lowpass.html', original=file_path, lowpass=filter_path)

    return render_template('lowpass.html')    

@app.route('/median', methods=['GET', 'POST'])
def median():

    original_file = None
    filtered_file = None

    if request.method == "POST":
        file = request.files['img']
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD'], filename)
        file.save(file_path)

        img = cv2.imread(file_path, 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply Median filter
        blurred = cv2.medianBlur(gray, 5)

        result_image = BytesIO()
        plt.imsave(result_image, blurred, format='jpg', cmap=plt.cm.gray)
        result_image.seek(0)
        filter_path = os.path.join(app.config['UPLOAD'], 'medaian.jpg')
        with open(os.path.join(app.config['UPLOAD'], 'median.jpg'), 'wb') as f:
            f.write(result_image.read())

        return render_template('median.html', original=file_path, median=filter_path)

    return render_template('median.html')

# A class used to implement a Binary Tree consisting of Nodes!
class Node(object):
    left = None
    right = None
    item = None
    weight = 0

    def __init__(self, symbol, weight, l=None, r=None):
        self.symbol = symbol
        self.weight = weight
        self.left = l
        self.right = r

    # Called when outputting/printing the node
    def __repr__(self):
        return '("%s", %s, %s, %s)' % (self.symbol, self.weight, self.left, self.right)

def sortByWeight(node):    
    return (node.weight * 1000000 + ord(node.symbol[0]))  # Sort by weight and alphabetical order if same weight

# A Class used to apply the Huffman Coding algorithm to encode / compress a message
class HuffmanEncoder:
    def __init__(self):
        self.symbols = {}
        self.codes = {}
        self.tree = []
        self.message = ""

    def frequencyAnalysis(self):
        self.symbols = {}
        for symbol in self.message:
            self.symbols[symbol] = self.symbols.get(symbol, 0) + 1

    def preorder_traverse(self, node, path=""):
        if node.left == None:
            self.codes[node.symbol] = path
        else:
            self.preorder_traverse(node.left, path + "0")
            self.preorder_traverse(node.right, path + "1")

    def encode(self, message):
        self.message = message
        # Identify the list of symbols and their weights / frequency in the message
        self.frequencyAnalysis()

        # Convert list of symbols into a binary Tree structure
        # Step 1: Generate list of Nodes...
        self.tree = []
        for symbol in self.symbols.keys():
            self.tree.append(Node(symbol, self.symbols[symbol], None, None))

        # Step 2: Sort list of nodes per weight
        self.tree.sort(key=sortByWeight)

        # Step 3: Organize all nodes into a Binary Tree.
        while len(self.tree) > 1:  # Carry on till the tree has only one root node!
            leftNode = self.tree.pop(0)
            rightNode = self.tree.pop(0)
            newNode = Node(leftNode.symbol + rightNode.symbol, leftNode.weight + rightNode.weight, leftNode, rightNode)
            self.tree.append(newNode)
            self.tree.sort(key=sortByWeight)

        # Generate List of Huffman Code for each symbol used...
        self.codes = {}
        self.preorder_traverse(self.tree[0])

        # Encode Message:
        encodedMessage = ""
        for symbol in message:
            encodedMessage = encodedMessage + self.codes[symbol]

        return encodedMessage

@app.route('/huffman', methods=['GET', 'POST'])
def huffman():
    result = None

    if request.method == 'POST':
        message = request.form['message']
        encoder = HuffmanEncoder()
        compressedMessage = encoder.encode(message)
        result = {"message": message, "compressedMessage": compressedMessage}

        return render_template('huffman.html', result=result)

    return render_template('huffman.html')




if __name__ == '__main__':
    app.run(debug=True, port=8001)
