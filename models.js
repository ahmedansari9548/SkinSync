const { exec } = require("child_process");
const path = require("path");
const fs = require("fs");

// Ensure processed & results folders exist
const processedDir = path.join(__dirname, "../processed");
const resultsDir = path.join(__dirname, "../results");

if (!fs.existsSync(processedDir)) fs.mkdirSync(processedDir, { recursive: true });
if (!fs.existsSync(resultsDir)) fs.mkdirSync(resultsDir, { recursive: true });

// Object Detection Model
async function runObjectDetection(imagePath) {
    return new Promise((resolve, reject) => {
        const outputPath = path.join(processedDir, `processed_${path.basename(imagePath)}`);

        exec(`python models/object_detection.py --input ${imagePath} --output ${outputPath}`, (err) => {
            if (err) return reject(err);
            resolve(outputPath);
        });
    });
}

// Skin Disease Detection Model
async function runSkinDiseaseDetection(imagePath) {
    return new Promise((resolve, reject) => {
        exec(`python models/skin_disease_detection.py --input ${imagePath}`, (err, stdout) => {
            if (err) return reject(err);

            try {
                const resultData = JSON.parse(stdout);
                resolve(resultData);  // Returning JSON directly instead of writing to file
            } catch (parseError) {
                reject(parseError);
            }
        });
    });
}

module.exports = { runObjectDetection, runSkinDiseaseDetection };


// const { exec } = require("child_process");
// const path = require("path");
// const fs = require("fs");

// // Ensure results folder exists
// const resultsDir = path.join(__dirname, "results");
// if (!fs.existsSync(resultsDir)) {
//     fs.mkdirSync(resultsDir, { recursive: true });
// }

// // Object Detection Model
// async function runObjectDetection(imagePath) {
//     return new Promise((resolve, reject) => {
//         const outputPath = path.join(__dirname, "processed", `processed_${path.basename(imagePath)}`);

//         exec(`python models/object_detection.py --input ${imagePath} --output ${outputPath}`, (err) => {
//             if (err) return reject(err);
//             resolve(outputPath);
//         });
//     });
// }

// // Skin Disease Detection Model
// async function runSkinDiseaseDetection(imagePath) {
//     return new Promise((resolve, reject) => {
//         const resultPath = path.join(__dirname, "results", "skin_analysis.json");

//         exec(`python models/skin_disease_detection.py --input ${imagePath}`, (err, stdout) => {
//             if (err) return reject(err);

//             try {
//                 const resultData = JSON.parse(stdout);
//                 fs.writeFileSync(resultPath, JSON.stringify(resultData, null, 2)); // Overwrite the result
//                 resolve(resultPath);
//             } catch (parseError) {
//                 reject(parseError);
//             }
//         });
//     });
// }

// module.exports = { runObjectDetection, runSkinDiseaseDetection };
