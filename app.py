from flask import Flask, redirect, render_template, request, url_for, send_file
import streamlit as st
import numpy as np
from werkzeug.utils import secure_filename
import os
import matplotlib.pyplot as plt
import cv2
import rembg

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
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
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
            input_image_path = os.path.join(app.config['UPLOAD'], 'background_input.png')
            output_image_path = os.path.join(app.config['UPLOAD'], 'background_output.png')

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
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
        edges_image_path = os.path.join(app.config['UPLOAD'], 'edges.jpg')
        cv2.imwrite(edges_image_path, edges)

        # Mengaplikasikan efek kartun dengan bitwise_and
        color = cv2.bilateralFilter(img, 9, 250, 250)
        cartoon = cv2.bitwise_and(color, color, mask=edges)
        cartoon_image_path = os.path.join(app.config['UPLOAD'], 'cartoon_image.jpg')
        cv2.imwrite(cartoon_image_path, cartoon)

        # Mengirim semua hasil gambar ke halaman HTML
        return render_template('cartoonize.html', 
                               original=file_path, 
                               gray=gray_image_path, 
                               edges=edges_image_path, 
                               cartoon=cartoon_image_path)
    return render_template('cartoonize.html')



############################################## Tambahan
def detect_and_apply_sticker(image_path):
    # Baca gambar
    img = cv2.imread(image_path)

    # Inisialisasi detektor wajah
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
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

@app.route('/opening', methods=['GET','POST'])
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

        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))
        opening = cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel)

        # Deteksi wajah dan tempelkan kacamata
        result_image = BytesIO()
        plt.imsave(result_image, opening, format='png', cmap=plt.cm.gray)
        result_image.seek(0)
        filter_path = os.path.join(app.config['UPLOAD'], 'opening.jpg')
        with open(os.path.join(app.config['UPLOAD'], 'opening.jpg'), 'wb') as f:
            f.write(result_image.read())

        return render_template('opening.html', original=file_path, opening=filter_path)

    return render_template('opening.html')


@app.route('/closing', methods=['GET','POST'])
def closing():

    original_file = None
    filtered_file = None

    if request.method == "POST":
        from io import BytesIO
        file = request.files['img']
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD'], filename)
        file.save(file_path)
        
        img = cv2.imread(file_path, 0)

        img = cv2.threshold(img, 0, 255, 
                     cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
        
        kernel = np.ones((3, 3), np.uint8)
        closing = cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel)

        # Deteksi wajah dan tempelkan kacamata
        result_image = BytesIO()
        plt.imsave(result_image, closing, format='png', cmap=plt.cm.gray)
        result_image.seek(0)
        filter_path = os.path.join(app.config['UPLOAD'], 'closing.jpg')
        with open(os.path.join(app.config['UPLOAD'], 'closing.jpg'), 'wb') as f:
            f.write(result_image.read())

        return render_template('closing.html', original=file_path, closing=filter_path)

    return render_template('closing.html')
#



if __name__ == '__main__': 
    app.run(debug=True,port=8001)
