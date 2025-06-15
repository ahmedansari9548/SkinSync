from flask import Flask, request, jsonify
import cv2
import os

app = Flask(__name__)

@app.route('/validate-face', methods=['POST'])
def validate_face():
    try:
        data = request.json
        image_path = data.get("image_path")
        
        if not image_path or not os.path.exists(image_path):
            return jsonify({"valid": False, "error": "Image not found"})
        
        # Load the image
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Load OpenCV's pre-trained face detection model
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        # If at least one face is found, return valid
        return jsonify({"valid": len(faces) > 0})
    
    except Exception as e:
        return jsonify({"valid": False, "error": str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
