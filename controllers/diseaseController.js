const { exec } = require("child_process");
const path = require("path");
const fs = require("fs");

// Path to the Python script
const PYTHON_SCRIPT_PATH = path.join(__dirname, "../model.py");

// Function to detect disease using the Python model
const detectDisease = (imagePath) => {
    return new Promise((resolve, reject) => {
        // Run the Python script with the image path as an argument
        exec(`python "${PYTHON_SCRIPT_PATH}" "${imagePath}"`, (error, stdout, stderr) => {
            if (error) {
                console.error("Error executing Python script:", stderr);
                reject({ error: "Model execution failed" });
                return;
            }

            // Read the results.json file
            const resultsFile = path.join(__dirname, "../results/results.json");
            fs.readFile(resultsFile, "utf8", (err, data) => {
                if (err) {
                    console.error("Error reading results.json:", err);
                    reject({ error: "Failed to read model output" });
                    return;
                }

                try {
                    const result = JSON.parse(data);
                    resolve(result);
                } catch (parseError) {
                    console.error("Error parsing results.json:", parseError);
                    reject({ error: "Invalid JSON format in results.json" });
                }
            });
        });
    });
};

module.exports = { detectDisease };
