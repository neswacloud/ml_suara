const modelPath = 'content/model.json';
let model;

async function loadModel() {
    try {
        model = await tf.loadLayersModel(modelPath);
        document.getElementById('output').innerText = 'Model loaded successfully.';
    } catch (error) {
        document.getElementById('output').innerText = 'Error loading model: ' + error.message;
        console.error('Error loading the model', error);
    }
}

async function handleImageUpload(event) {
    const file = event.target.files[0];
    if (!file) {
        return;
    }

    const reader = new FileReader();
    reader.onload = async (e) => {
        const img = document.getElementById('preview');
        img.src = e.target.result;
        img.style.display = 'block';

        if (!model) {
            document.getElementById('output').innerText = 'Model not loaded yet.';
            return;
        }

        try {
            // Preprocess the image (adjust as per your model's input requirements)
            const tensor = tf.browser.fromPixels(img)
                .resizeNearestNeighbor([224, 224]) // Example: Resize to 224x224
                .toFloat()
                .expandDims(); // Add batch dimension

            // Example: Normalize image data (adjust based on your model's training)
            // const offset = tf.scalar(127.5);
            // const divided = tensor.div(offset);
            // const normalized = divided.sub(tf.scalar(1));
            // const prediction = await model.predict(normalized);


            // Make a prediction
            const prediction = await model.predict(tensor);

            // Display the results (adjust based on your model's output)
            const outputDiv = document.getElementById('output');
            outputDiv.innerText = 'Prediction: ' + prediction.toString(); // Example: Display tensor string
            console.log('Prediction:', prediction.dataSync()); // Log prediction data

            // Dispose the tensor to free up memory
            tensor.dispose();
            prediction.dispose();


        } catch (error) {
            document.getElementById('output').innerText = 'Error during prediction: ' + error.message;
            console.error('Error during prediction', error);
        }
    };
    reader.readAsDataURL(file);
}

// Load the model when the page loads
window.onload = loadModel;

// Add event listener for image upload
document.getElementById('imageUpload').addEventListener('change', handleImageUpload);
