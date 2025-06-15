import subprocess
import os
import sys
import json

# Paths
SCRIPT_PATH = os.path.join(os.path.dirname(__file__), "models/skin_disease_detection.py")
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "models/chartData.json")

def predict_disease(image_path):
    """Runs skin disease detection and saves results."""
    try:
        # Run skin_disease_detection.py using subprocess
        subprocess.run(["python", SCRIPT_PATH, image_path], check=True)

        # Read results from chartData.json
        with open(RESULTS_PATH, "r") as json_file:
            result = json.load(json_file)

        return result

    except Exception as e:
        print(f"Error in model prediction: {str(e)}")
        return {"error": str(e)}

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Error: No image path provided")
        sys.exit(1)

    image_path = sys.argv[1]
    result = predict_disease(image_path)
    print(json.dumps(result, indent=2))
