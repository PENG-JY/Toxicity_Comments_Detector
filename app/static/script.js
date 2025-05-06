// In static/js/script.js

function analyzeToxicity() {
    const userId = document.getElementById("userId").value;
    const commentText = document.getElementById("commentText").value;
    const modelType = document.getElementById("model").value;

    const requestData = {
        id: userId,
        comment_text: commentText,
        model: modelType
    };

    fetch("http://localhost:5000/predict", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify(requestData)
    })
    .then(response => response.json())
    .then(data => {
        // Check if the data is an array with predictions
        if (Array.isArray(data) && data.length > 0) {
            const toxicityScore = data[0].Prediction;  // Access the Prediction value
            document.getElementById("result").innerHTML = `Toxicity Score: ${toxicityScore}`;
        } else {
            console.error("Error: Expected an array with prediction but received:", data);
            alert("An error occurred. Please check the console for details.");
        }
    })
    .catch(error => {
        console.error("Error:", error);
        alert("An error occurred while processing your request. Please check the console for details.");
    });
}
