import json
import sys
import os
from inference_sdk import InferenceHTTPClient, InferenceConfiguration

# Initialize Roboflow client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="rgD0iFCgR5ZLKLZ1wI70"
)

custom_configuration = InferenceConfiguration(confidence_threshold=0.1, iou_threshold=0.1)
CLIENT.use_configuration(custom_configuration)

def preprocess_for_radar(detection_data):
    """Convert model predictions into structured JSON format."""
    class_stats = {}

    for prediction in detection_data.get('predictions', []):
        class_name = prediction['class']
        confidence = prediction['confidence']

        if class_name not in class_stats:
            class_stats[class_name] = {"total_confidence": 0, "count": 0}

        class_stats[class_name]['total_confidence'] += confidence
        class_stats[class_name]['count'] += 1

    radar_data = {
        "labels": list(class_stats.keys()),
        "series": [
            {
                "values": [
                    round(stats['total_confidence'] / stats['count'], 4)
                    for stats in class_stats.values()
                ],
                "text": "Predicted Confidence"
            }
        ]
    }

    return radar_data

def main(image_path):
    """Run inference and save results to JSON."""
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return

    result = CLIENT.infer(image_path, model_id="skn-1/2")
    processed_data = preprocess_for_radar(result)

    output_path = "chartData.json"
    with open(output_path, 'w') as f:
        json.dump(processed_data, f, indent=2)

    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Error: No image path provided")
        sys.exit(1)

    main(sys.argv[1])


# import json
# from inference_sdk import InferenceHTTPClient, InferenceConfiguration

# # Initialize the inference client
# CLIENT = InferenceHTTPClient(
#     api_url="https://detect.roboflow.com",
#     api_key="rgD0iFCgR5ZLKLZ1wI70"
# )

# # Assume you have a configuration for the client
# custom_configuration = InferenceConfiguration(confidence_threshold=0.1, iou_threshold=0.1)

# CLIENT.use_configuration(custom_configuration)

# def preprocess_for_radar(detection_data):
#     # Initialize aggregation dictionary
#     class_stats = {}

#     # Loop through predictions and aggregate confidence values for each class
#     for prediction in detection_data['predictions']:
#         class_name = prediction['class']
#         confidence = prediction['confidence']

#         # Initialize if this class hasn't been encountered yet
#         if class_name not in class_stats:
#             class_stats[class_name] = {
#                 'total_confidence': 0,
#                 'count': 0,
#                 'average_confidence': 0
#             }

#         # Aggregate the confidence for each class
#         class_stats[class_name]['total_confidence'] += confidence
#         class_stats[class_name]['count'] += 1

#     # Calculate average confidence for each class
#     for class_name, stats in class_stats.items():
#         class_stats[class_name]['average_confidence'] = stats['total_confidence'] / stats['count']

#     # Extract labels and values for radar chart
#     labels = list(class_stats.keys())  # The class names will be the labels
#     values = [stats['average_confidence'] for stats in class_stats.values()]  # The average confidence will be the values

#     # Structure the data into the required format
#     radar_data = {
#         "labels": labels,  # The class names
#         "series": [
#             {
#                 "values": values,  # The list of average confidence values
#                 "text": "Predicted Confidence"  # Optional, can be any label you prefer
#             }
#         ]
#     }

#     return radar_data

# # Function to simulate inference from an image
# def simulate_inference(image_path):
#     # Replace this with actual inference code from the Roboflow API
#     result = CLIENT.infer(image_path, model_id="skn-1/2")
#     return result

# def main(image_path):
#     # Process the inference result
#     result = simulate_inference(image_path)

#     # Preprocess the result into the desired format
#     processed_data = preprocess_for_radar(result)

#     # Save the processed radar data to a JSON file
#     output_path = "chartData.json"  # Change the path as needed
#     with open(output_path, 'w') as f:
#         json.dump(processed_data, f, indent=2)

#     print(f"Radar chart data saved to {output_path}")

# # Replace with your image path
# image_path = r"c:\Users\talha\OneDrive\Desktop\front.jpg"

# # Run the main function
# main(image_path)
