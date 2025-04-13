import {
    HandLandmarker,
    FilesetResolver,
    DrawingUtils
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18";

const enableWebcamButton = document.getElementById("webcamButton")
const trainingDataButton = document.getElementById('trainingDataButton')

const video = document.getElementById("webcam")
const canvasElement = document.getElementById("output_canvas")
const canvasCtx = canvasElement.getContext("2d")

// empty array for all training data
const trainingData = []

const drawUtils = new DrawingUtils(canvasCtx)
let handLandmarker = undefined;
let webcamRunning = false;
let results = undefined;


/********************************************************************
 // CREATE THE POSE DETECTOR
 ********************************************************************/
const createHandLandmarker = async () => {
    const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm");
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
            delegate: "GPU"
        },
        runningMode: "VIDEO",
        numHands: 2
    });
    console.log("model loaded, you can start webcam")

    enableWebcamButton.addEventListener("click", (e) => enableCam(e))
    trainingDataButton.addEventListener("click", (e) => logTrainingData(e))

    document.addEventListener('keydown', function(event) {
        if (event.key === '1') {
            logHand()
        }
        if (event.key === '2') {
            logFist()
        }
        if (event.key === '3') {
            logThumbsUp()
        }
        if (event.key === '4') {
            logPeace()
        }
        if (event.key === '5') {
            logHorns()
        }
    });
}


/********************************************************************
 // START THE WEBCAM
 ********************************************************************/
async function enableCam() {
    webcamRunning = true;
    try {
        const stream = await navigator.mediaDevices.getUserMedia({video: true, audio: false});
        video.srcObject = stream;
        video.addEventListener("loadeddata", () => {
            canvasElement.style.width = video.videoWidth;
            canvasElement.style.height = video.videoHeight;
            canvasElement.width = video.videoWidth;
            canvasElement.height = video.videoHeight;
            document.querySelector(".videoView").style.height = video.videoHeight + "px";
            predictWebcam()
        });
    } catch (error) {
        console.error("Error accessing webcam:", error);
    }
}


/********************************************************************
 // START PREDICTIONS
 ********************************************************************/
async function predictWebcam() {
    results = await handLandmarker.detectForVideo(video, performance.now())

    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    for (let hand of results.landmarks) {
        drawUtils.drawConnectors(hand, HandLandmarker.HAND_CONNECTIONS, {color: "#00FF00", lineWidth: 5});
        drawUtils.drawLandmarks(hand, {radius: 4, color: "#FF0000", lineWidth: 2});
    }

    if (webcamRunning) {
        window.requestAnimationFrame(predictWebcam)
    }
}


/********************************************************************
 // LOG ROCK, PAPER, SCISSORS IN THE CONSOLE AND ADD TO TRAININGDATA
 ********************************************************************/
function logHand() {
    console.log('log hand')
    for (let hand of results.landmarks) {
        // flatMap data and simplify
        const result = hand.flatMap(point => [point.x, point.y, point.z])
        // log simplified data
        // console.log(result)

        // push to trainingData
        trainingData.push({points: result, label: "hand"})

        // log trainingData
        // console.log(`Poses: ${trainingData}`)
    }
}

function logFist() {
    console.log('log fist')
    for (let hand of results.landmarks) {
        // flatMap data and simplify
        const result = hand.flatMap(point => [point.x, point.y, point.z])
        // log simplified data
        // console.log(result)

        // push to trainingData
        trainingData.push({points: result, label: "fist"})

        // log trainingData
        // console.log(`Poses: ${trainingData}`)
    }
}

function logThumbsUp() {
    console.log('log thumbs up')
    for (let hand of results.landmarks) {
        // flatMap data and simplify
        const result = hand.flatMap(point => [point.x, point.y, point.z])
        // log simplified data
        // console.log(result)

        // push to trainingData
        trainingData.push({points: result, label: "thumbsup"})

        // log trainingData
        // console.log(`Poses: ${trainingData}`)
    }
}

function logPeace() {
    console.log('log peace')
    for (let hand of results.landmarks) {
        // flatMap data and simplify
        const result = hand.flatMap(point => [point.x, point.y, point.z])
        // log simplified data
        // console.log(result)

        // push to trainingData
        trainingData.push({points: result, label: "peace"})

        // log trainingData
        // console.log(`Poses: ${trainingData}`)
    }
}

function logHorns() {
    console.log('log horns')
    for (let hand of results.landmarks) {
        // flatMap data and simplify
        const result = hand.flatMap(point => [point.x, point.y, point.z])
        // log simplified data
        // console.log(result)

        // push to trainingData
        trainingData.push({points: result, label: "horns"})

        // log trainingData
        // console.log(`Poses: ${trainingData}`)
    }
}


/********************************************************************
 // LOG TRAININGDATA TO COPY PASTE IN JSON FILE
 ********************************************************************/
function logTrainingData() {
    const trainingDataToString = JSON.stringify(trainingData)
    console.log(trainingDataToString)
}


/********************************************************************
 // START THE APP
 ********************************************************************/
if (navigator.mediaDevices?.getUserMedia) {
    createHandLandmarker()
}