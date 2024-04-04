document.getElementById('upload-btn').addEventListener('click', function () {
    document.getElementById('file-input').click();
});

document.getElementById('file-input').addEventListener('change', function (event) {
    const file = event.target.files[0];

    if (file) {
        const reader = new FileReader();

        reader.onload = function (e) {
            const imageUrl = e.target.result;

            // Display the image
            const imgElement = document.createElement('img');
            imgElement.src = imageUrl;
            imgElement.style.maxWidth = '300px'; // Adjust the max width as needed
            imgElement.style.maxHeight = '300px'; // Adjust the max height as needed
            document.getElementById('result-container').innerHTML = '';
            document.getElementById('result-container').appendChild(imgElement);

            // Use FormData to send the file to the server
            const formData = new FormData();
            formData.append('image', file);

            fetch('/predict', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                // Display the prediction result
                const predictionResult1 = document.createElement('p');
                predictionResult1.innerText = `Prediction (Random Forest): ${data.prediction1}, Probability: ${data.probability1}`;
                document.getElementById('result-container').appendChild(predictionResult1);

                const predictionResult2 = document.createElement('p');
                predictionResult2.innerText = `Prediction (SVM): ${data.prediction2}, Probability: ${data.probability2}`;
                document.getElementById('result-container').appendChild(predictionResult2);
            })
            .catch(error => {
                console.error('Error during prediction:', error);
            });
        };

        reader.readAsDataURL(file);
    } else {
        alert('Please select a valid image file.');
    }
});

// async function prediction(){
//     let response = await fetch("/predict").then((response)=>response.json())
//     // console.log(response['otp'])

//     const para = document.createElement("p")
//     para.id = "result"
//     para.innerHTML = response['predicted_class']

//     // document.getElementById("result-container").innerHTML = `<p id="result" >${response['otp']}</p>`
//     document.getElementById("result-container").appendChild(para)


//     setTimeout(() => {
//         para.style.opacity = '0';
//     }, 20000);
//     errorBox.addEventListener('transitionend', () => errorBox.remove());


// }