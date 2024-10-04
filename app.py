from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import pywt
import base64

app = Flask(__name__)

# ฟังก์ชัน Resize ลายน้ำให้ตรงกับขนาดของ LL จาก DWT
def resize_watermark_to_LL(LL, watermark_img):
    LL_height, LL_width = LL.shape
    watermark_resized = cv2.resize(watermark_img, (LL_width, LL_height), interpolation=cv2.INTER_CUBIC)
    return watermark_resized

# ฟังก์ชันการทำ DCT
def apply_dct(block):
    block_float = np.float32(block)
    return cv2.dct(block_float)

# ฟังก์ชันการทำ inverse DCT
def inverse_dct(block):
    return cv2.idct(block)

# ฟังก์ชันการปรับสีของลายน้ำ
def adjust_watermark_color(watermark_img, alpha=1.0):
    # ปรับความเข้มของสีด้วยการคูณค่าของพิกเซล
    adjusted_watermark = cv2.convertScaleAbs(watermark_img, alpha=alpha)
    return adjusted_watermark

# ฟังก์ชันการฝังลายน้ำด้วย DWT-DCT
def embed_watermark_dwt_dct(original_img, watermark_img, alpha=0.075):
    channels = cv2.split(original_img)
    watermarked_channels = []

    # ปรับสีของลายน้ำก่อนฝัง
    watermark_img_adjusted = adjust_watermark_color(watermark_img)

    for channel in channels:
        coeffs = pywt.dwt2(channel, 'haar')
        LL, (LH, HL, HH) = coeffs

        dct_LL = apply_dct(LL)
        watermark_img_resized = resize_watermark_to_LL(LL, watermark_img_adjusted)
        watermark_dct = apply_dct(cv2.cvtColor(watermark_img_resized, cv2.COLOR_RGB2GRAY))

        watermarked_dct_LL = dct_LL + alpha * watermark_dct
        watermarked_LL = inverse_dct(watermarked_dct_LL)
        watermarked_channel = pywt.idwt2((watermarked_LL, (LH, HL, HH)), 'haar')
        watermarked_channels.append(np.uint8(np.clip(watermarked_channel, 0, 255)))

    return cv2.merge(watermarked_channels)

# ฟังก์ชันการถอดลายน้ำ
def extract_watermark_dwt_dct(watermarked_img, original_img, alpha=0.075):
    channels_w = cv2.split(watermarked_img)
    channels_o = cv2.split(original_img)
    extracted_channels = []

    for channel_w, channel_o in zip(channels_w, channels_o):
        coeffs_w = pywt.dwt2(channel_w, 'haar')
        coeffs_o = pywt.dwt2(channel_o, 'haar')
        LL_w, (LH_w, HL_w, HH_w) = coeffs_w
        LL_o, (LH_o, HL_o, HH_o) = coeffs_o
        
        dct_LL_w = apply_dct(LL_w)
        dct_LL_o = apply_dct(LL_o)
        
        watermark_dct = (dct_LL_w - dct_LL_o) / alpha
        extracted_watermark = inverse_dct(watermark_dct)

        extracted_watermark = cv2.normalize(extracted_watermark, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        
        extracted_channels.append(np.uint8(np.clip(extracted_watermark, 0, 255)))

    return cv2.merge(extracted_channels)

# ฟังก์ชันการถอดลายน้ำแบบ Blind โดยไม่มีภาพต้นฉบับ
def blind_watermark_extraction(watermarked_img, alpha=0.05):
    channels = cv2.split(watermarked_img)
    extracted_channels = []

    for channel in channels:
        coeffs = pywt.dwt2(channel, 'haar')
        LL, (LH, HL, HH) = coeffs

        # วิเคราะห์ HH เพื่อหาสัญญาณลายน้ำ (Blind extraction)
        dct_HH = apply_dct(HH)
        extracted_watermark_dct = dct_HH / alpha  # วิเคราะห์สัญญาณลายน้ำ
        extracted_watermark = inverse_dct(extracted_watermark_dct)

        extracted_watermark = cv2.normalize(extracted_watermark, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        extracted_channels.append(np.uint8(np.clip(extracted_watermark, 0, 255)))

    return cv2.merge(extracted_channels)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/extract', methods=['POST'])
def extract_watermark():
    if 'watermarked' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    watermarked_file = request.files['watermarked']
    if watermarked_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    watermarked_img = cv2.imdecode(np.frombuffer(watermarked_file.read(), np.uint8), cv2.IMREAD_COLOR)
    watermarked_img_rgb = cv2.cvtColor(watermarked_img, cv2.COLOR_BGR2RGB)

    # ถอดลายน้ำ
    extracted_watermark = blind_watermark_extraction(watermarked_img_rgb)

    # Convert extracted watermark back to bytes
    _, extracted_watermark_bytes = cv2.imencode('.png', extracted_watermark)

    # Convert to base64
    extracted_watermark_base64 = base64.b64encode(extracted_watermark_bytes).decode('utf-8')

    return jsonify({'extracted_watermark': extracted_watermark_base64})

@app.route('/test')
def test_page():
    return render_template('extract.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    if 'original' not in request.files and 'watermark' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    # โหลดไฟล์ภาพ
    original_file = request.files.get('original')
    watermark_file = request.files['watermark']

    watermark_img = cv2.imdecode(np.frombuffer(watermark_file.read(), np.uint8), cv2.IMREAD_COLOR)
    watermark_img_rgb = cv2.cvtColor(watermark_img, cv2.COLOR_BGR2RGB)

    if original_file:
        original_img = cv2.imdecode(np.frombuffer(original_file.read(), np.uint8), cv2.IMREAD_COLOR)
        original_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

        # ฝังลายน้ำ
        watermarked_img = embed_watermark_dwt_dct(original_img_rgb, watermark_img_rgb)

        # ถอดลายน้ำแบบปกติ (มีภาพต้นฉบับ)
        extracted_watermark = extract_watermark_dwt_dct(watermarked_img, original_img_rgb)

    else:
        watermarked_img = watermark_img_rgb  # สมมุติว่าเป็นภาพที่ฝังลายน้ำแล้ว

        # ถอดลายน้ำแบบ Blind (ไม่มีภาพต้นฉบับ)
        extracted_watermark = blind_watermark_extraction(watermarked_img)

    # Convert images back to bytes
    _, watermarked_img_bytes = cv2.imencode('.png', watermarked_img)
    _, extracted_watermark_bytes = cv2.imencode('.png', extracted_watermark)

    # Convert to base64
    watermarked_image_base64 = base64.b64encode(watermarked_img_bytes).decode('utf-8')
    extracted_watermark_base64 = base64.b64encode(extracted_watermark_bytes).decode('utf-8')

    return jsonify({
        'watermarked_image': watermarked_image_base64,
        'extracted_watermark': extracted_watermark_base64
    })


if __name__ == '__main__':
    app.run(debug=True)
