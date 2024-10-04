from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import base64
import pywt
import os
import scipy.fftpack as fftpack

app = Flask(__name__)

def embed_watermark(original_img, watermark_img, alpha=0.05):
    channels_o = cv2.split(original_img)
    channels_wm = cv2.split(watermark_img)
    watermarked_channels = []

    for channel_o, channel_wm in zip(channels_o, channels_wm):
        coeffs_o = pywt.dwt2(channel_o, 'haar')
        LL_o, (LH_o, HL_o, HH_o) = coeffs_o

        coeffs_wm = pywt.dwt2(channel_wm, 'haar')
        LL_wm, (LH_wm, HL_wm, HH_wm) = coeffs_wm

        dct_LL_o = fftpack.dct(fftpack.dct(LL_o.T, norm='ortho').T, norm='ortho')
        dct_LL_wm = fftpack.dct(fftpack.dct(LL_wm.T, norm='ortho').T, norm='ortho')

        watermarked_dct = dct_LL_o + alpha * dct_LL_wm
        watermarked_LL = fftpack.idct(fftpack.idct(watermarked_dct.T, norm='ortho').T, norm='ortho')

        watermarked_channel = pywt.idwt2((watermarked_LL, (LH_o, HL_o, HH_o)), 'haar')
        watermarked_channels.append(np.uint8(np.clip(watermarked_channel, 0, 255)))

    return cv2.merge(watermarked_channels)

def extract_watermark(watermarked_img, alpha=0.05):
    channels_wm = cv2.split(watermarked_img)
    extracted_channels = []

    for channel_wm in channels_wm:
        coeffs_wm = pywt.dwt2(channel_wm, 'haar')
        LL_wm, (LH_wm, HL_wm, HH_wm) = coeffs_wm

        dct_LL_wm = fftpack.dct(fftpack.dct(LL_wm.T, norm='ortho').T, norm='ortho')

        extracted_watermark_dct = dct_LL_wm / alpha
        extracted_watermark = fftpack.idct(fftpack.idct(extracted_watermark_dct.T, norm='ortho').T, norm='ortho')

        extracted_channel = pywt.idwt2((extracted_watermark, (LH_wm, HL_wm, HH_wm)), 'haar')
        extracted_channels.append(np.uint8(np.clip(extracted_channel, 0, 255)))

    return cv2.merge(extracted_channels)

def img_to_base64(img):
    _, buffer = cv2.imencode('.png', img)
    return base64.b64encode(buffer).decode('utf-8')

def base64_to_img(base64_str):
    nparr = np.frombuffer(base64.b64decode(base64_str), np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

@app.route('/')
def index():
    return render_template('test.html')

@app.route('/extract', methods=['GET', 'POST'])
def extract():
    if request.method == 'POST':
        # โหลดภาพจากที่เก็บ
        watermarked_img = cv2.imread(os.path.join('static', 'watermarked_image.png'))
        extracted_watermark_img = extract_watermark(watermarked_img)

        return jsonify({
            'extracted_watermark': img_to_base64(extracted_watermark_img)
        })

    return render_template('extract.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    original = request.files['original']
    watermark = request.files['watermark']

    original_img = cv2.imdecode(np.frombuffer(original.read(), np.uint8), cv2.IMREAD_COLOR)
    watermark_img = cv2.imdecode(np.frombuffer(watermark.read(), np.uint8), cv2.IMREAD_COLOR)

    # Resize watermark to match original image size
    watermark_img = cv2.resize(watermark_img, (original_img.shape[1], original_img.shape[0]))

    watermarked_img = embed_watermark(original_img, watermark_img)
    extracted_watermark_img = extract_watermark(watermarked_img)

    return jsonify({
        'watermarked_image': img_to_base64(watermarked_img),
        'extracted_watermark': img_to_base64(extracted_watermark_img)
    })

if __name__ == '__main__':
    app.run(debug=True)
