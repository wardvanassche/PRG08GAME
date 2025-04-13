import poses from '/public/data.json' with { type: "json" };

const trainButton = document.getElementById('trainButton')
const testButton = document.getElementById('testButton')
const saveButton = document.getElementById('saveButton')

ml5.setBackend("webgl");
const nn = ml5.neuralNetwork({ task: 'classification', debug: true })

let trainData = []
let testData = []

// train NN
function train() {

    let data = poses
    data = data.toSorted(() => Math.random() - 0.5)

    trainData = data.slice(0, Math.floor(data.length * 0.8))
    testData = data.slice(0, Math.floor(data.length * 0.8) + 1)

    for(const {points, label} of trainData) {
        nn.addData(points, {label: label})
    }

    nn.normalizeData()
    nn.train({ epochs: 20 }, () => console.log("Training complete"))
}

// test accuracy using testdata
async function testAccuracy() {
    let correct = 0

    for (const {points, label} of testData) {
        const prediction = await nn.classify(points)

        if(prediction[0].label === label) {
            correct++
        }
    }
    console.log("Accuracy:", correct / testData.length)
}

// save model
function saveModel() {
    nn.save("model", () => console.log("Model was saved"))
}

function handler(){
    trainButton.addEventListener("click", (e) => train(e))
    testButton.addEventListener("click", (e) => testAccuracy(e))
    saveButton.addEventListener("click", (e) => saveModel(e))
}


/********************************************************************
 // START THE APP
 ********************************************************************/
if (navigator.mediaDevices?.getUserMedia) {
    handler()
}