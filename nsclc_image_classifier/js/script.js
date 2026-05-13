// global variable to hold the model
let model;

window.addEventListener("load", init);

async function init() {
  // add event listener and function to fileInput
  fileInput = document.getElementById("file-input")
  fileInput.addEventListener("change",previewImage)
  // add event listerner and function to classify button
  classifyBTN= document.getElementById("classify-image-btn")
  classifyBTN.addEventListener("click",classifyImage)
  resetBTN=document.getElementById("reset-btn")
  resetBTN.addEventListener("click",reset)
  // load the model
  model = await loadModel();
}

async function loadModel() {
  // load the model
  const model = await tf.loadGraphModel("tfjs_model/model.json");
  return model;
}

function previewImage(event){
  console.log(event)
  // get the preview image element
  const previewImage = document.getElementById("preview-image")
  // get the image that was slected
  const fileInput = event.target.files[0]
  // update the preview image with the selected file
  previewImage.src=URL.createObjectURL(fileInput)

}

function processImage(img) {
  const imgWidth = 256;
  const imgHeight = 256;
  // create image tensor from selected image, resizing it ready for predictions
  const imgTensor = tf.browser
    .fromPixels(img)
    .resizeBilinear([imgHeight, imgWidth])
    .expandDims();
  console.log(imgTensor.shape);
  return imgTensor;
}

function getModelPrediction(imgTensor) {
  // get the model predictions for the provided image tensor
  const prediction = model.predict(imgTensor);
  return prediction;
}

function displayPrediction(prediction) {
    const classNames = ["Adenocarcinoma", "Benign ", "Squamous cell carcinoma"];
    const resultDisplay = document.getElementById("result-display");
    // get the values of the prediction
    const data = prediction.dataSync();
    // get the predicted class for the image, by finding the index of the highest value
    const predictedClass = classNames[data.indexOf(Math.max(...data))];
    console.log(predictedClass)
    // update the page with the results of the prediction
    resultDisplay.innerHTML=`
                    <h2>Classification Result</h2>
                    <p>The predicted class for the image is ${predictedClass}</p>
                    <h3>Prediction breakdown</h3>
                    <p>The model's prediction for the classification are:</p>`;
    for (let i = 0; i < data.length; i++) {
      // display the probability values of each class for this prediction, rounded to 2 decimal places
      let value = (data[i] * 100).toFixed(2);
      resultDisplay.insertAdjacentHTML("beforeend",`
        <p>${classNames[i]} : ${value}%</p>`)
      console.log(classNames[i] + ": " + value + "%");
    }
}

async function classifyImage() {
  const fileInput = document.getElementById("file-input");
  // check that a file has been selected, if not advise user to select a file
  if(fileInput.files.length===0){
    const resultDisplay = document.getElementById("result-display");
    resultDisplay.innerHTML=`<h3>Please select a file</h3>`
  }else{
    // get user selected image
    const img = document.getElementById("preview-image");
    // create image tensor
    const imgTensor = processImage(img);
    // get the model's prediction
    const prediction = await getModelPrediction(imgTensor);

    // display the model's prediction
    displayPrediction(prediction);
  }
}

function reset(){
  const resultDisplay = document.getElementById("result-display")
  const fileInput = document.getElementById("file-input");
  const previewImage = document.getElementById("preview-image")
  resultDisplay.innerHTML="";
  fileInput.value="";
  previewImage.src="icon/photo.png";
}
