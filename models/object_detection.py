import sys
import os
import json
from inference_sdk import InferenceHTTPClient, InferenceConfiguration
from PIL import Image, ImageDraw, ImageFont

# # Configure paths
# RESULT_FOLDER = "static/results"
# os.makedirs(RESULT_FOLDER, exist_ok=True)

# Configure paths
RESULT_FOLDER = "static/results"

# 🔹 Delete the folder if it exists, then recreate it
if os.path.exists(RESULT_FOLDER):
    import shutil
    shutil.rmtree(RESULT_FOLDER)  # Deletes the entire folder and its contents

os.makedirs(RESULT_FOLDER, exist_ok=True)  # Recreate an empty folder


# Initialize the client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="rgD0iFCgR5ZLKLZ1wI70"
)

custom_configuration = InferenceConfiguration(confidence_threshold=0.1, iou_threshold=0.6)

from PIL import Image, ImageDraw, ImageFont

def draw_predictions(image, predictions):
    draw = ImageDraw.Draw(image)
    
    # Try to load font, else use default
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default()

    for prediction in predictions:
        x, y, width, height = prediction['x'], prediction['y'], prediction['width'], prediction['height']
        label, confidence = prediction['class'], prediction['confidence']

        x1, y1 = x - width / 2, y - height / 2
        x2, y2 = x + width / 2, y + height / 2

        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        text = f"{label} {confidence:.2f}"

        # 🔥 FIXED: Use textbbox instead of textsize
        bbox = draw.textbbox((x1, y1), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # Draw label background
        draw.rectangle([x1, y1 - text_height, x1 + text_width, y1], fill="red")

        # Draw text
        draw.text((x1, y1 - text_height), text, fill="white", font=font)

    return image


def main(image_path):
    if not os.path.exists(image_path):
        print(json.dumps({"error": "Image file not found"}))
        return

    # Run inference
    result = CLIENT.infer(image_path, model_id="pimples-detection/4")

    image = Image.open(image_path)
    annotated_image = draw_predictions(image, result['predictions'])

    result_filename = "processed_image.png"
    result_path = os.path.join(RESULT_FOLDER, result_filename)
    annotated_image.save(result_path)

    print(json.dumps({"processed_image": result_filename}))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No image path provided"}))
        sys.exit(1)

    main(sys.argv[1])


# from flask import Flask, render_template, request, send_from_directory
# import os
# from werkzeug.utils import secure_filename
# from inference_sdk import InferenceHTTPClient, InferenceConfiguration
# from PIL import Image, ImageDraw, ImageFont

# app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = 'uploads'
# app.config['RESULT_FOLDER'] = 'static/results'
# app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'webp'}

# # Initialize the client
# CLIENT = InferenceHTTPClient(
#     api_url="https://detect.roboflow.com",
#     api_key="rgD0iFCgR5ZLKLZ1wI70"
# )

# # Set up custom configuration
# custom_configuration = InferenceConfiguration(confidence_threshold=0.1,
#                                                iou_threshold=0.6)

# def allowed_file(filename):
#     return '.' in filename and \
#            filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# def draw_predictions(image, predictions):
#     draw = ImageDraw.Draw(image)
#     try:
#         # Load a font (you can use a TTF font file if available)
#         font = ImageFont.truetype("arial.ttf", 20)
#     except IOError:
#         # Fallback to default font if the specified font is not available
#         font = ImageFont.load_default()

#     for prediction in predictions:
#         # Extract prediction details
#         x = prediction['x']
#         y = prediction['y']
#         width = prediction['width']
#         height = prediction['height']
#         confidence = prediction['confidence']
#         label = prediction['class']

#         # Calculate bounding box coordinates
#         x1 = x - width / 2
#         y1 = y - height / 2
#         x2 = x + width / 2
#         y2 = y + height / 2

#         # Draw bounding box
#         draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

#         # Add label and confidence
#         text = f"{label} {confidence:.2f}"
#         # Use getbbox to calculate text size
#         text_bbox = font.getbbox(text)
#         text_width = text_bbox[2] - text_bbox[0]
#         text_height = text_bbox[3] - text_bbox[1]

#         # Draw background rectangle for text
#         draw.rectangle([x1, y1 - text_height, x1 + text_width, y1], fill="red")
#         # Draw text
#         draw.text((x1, y1 - text_height), text, fill="white", font=font)

#     return image

# @app.route('/')
# def index():
#     return render_template('upload.html')

# @app.route('/upload', methods=['POST'])
# def upload_file():
#     if 'file' not in request.files:
#         return 'No file uploaded', 400
    
#     file = request.files['file']
#     if file.filename == '':
#         return 'No selected file', 400
    
#     if file and allowed_file(file.filename):
#         filename = secure_filename(file.filename)
#         upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         file.save(upload_path)

#         # Process the image
#         with CLIENT.use_configuration(custom_configuration):
#             result = CLIENT.infer(upload_path, model_id="pimples-detection/4")

#         image = Image.open(upload_path)
#         annotated_image = draw_predictions(image, result['predictions'])
        
#         # Save result
#         result_filename = f"annotated_{filename}"
#         result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
#         annotated_image.save(result_path)

#         return render_template('result.html', result_image=result_filename)
    
#     return 'Invalid file type', 400

# @app.route('/results/<filename>')
# def send_result(filename):
#     return send_from_directory(app.config['RESULT_FOLDER'], filename)

# if __name__ == '__main__':
#     # Create folders if they don't exist
#     os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
#     os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)
#     app.run(debug=True)