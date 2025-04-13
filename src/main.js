    import './style.css'
    import {
        HandLandmarker,
        FilesetResolver,
        DrawingUtils
    } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18";
    import poses from '/public/data.json' with {type: "json"};
    import kNear from "./knear.js";

    const enableWebcamButton = document.getElementById("webcamButton")
    const startButton = document.getElementById("startButton")
    const restartButton = document.getElementById("restartButton")

    const video = document.getElementById("webcam")
    const canvasElement = document.getElementById("output_canvas")
    const canvasCtx = canvasElement.getContext("2d")

    const drawUtils = new DrawingUtils(canvasCtx)
    let handLandmarker = undefined;
    let webcamRunning = false;
    let results = undefined;

    const k = 3
    const knn = new kNear(k)

    const prompt = document.getElementById("prompt")
    const scoreView = document.getElementById("score")
    const livesView = document.getElementById("lives")

    let score = 0
    let lives = 3
    let hasScoredThisRound = false; // Flag to prevent multiple scoring in a round

    let prediction = []
    let input = []
    let hand = []

    let currentPrompt = null
    let previousPrompt = null
    let gameActive = false

    let timerInterval = null
    let timeLimit = 2000


    const emojis = [
        {
            img: "/public/hand.png",
            label: "hand"
        },
        {
            img: "/public/fist.png",
            label: "fist"
        },
        {
            img: "/public/thumbsup.png",
            label: "thumbsup"
        },
        {
            img: "/public/peace.png",
            label: "peace"
        },
        {
            img: "/public/horns.png",
            label: "horns"
        },
    ]

    ml5.setBackend("webgl");
    const nn = ml5.neuralNetwork({ task: 'classification', debug: true })
    const modelDetails = {
        model: 'model/model.json',
        metadata: 'model/model_meta.json',
        weights: 'model/model.weights.bin'
    }

    async function loadModel() {
        nn.load(modelDetails, () => {
            console.log("het model is geladen!");
        });
    }


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
        startButton.addEventListener("click", (e) => startGame(e))
        restartButton.addEventListener("click", (e) => restartGame(e))
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

        hand = results.landmarks[0]

        if (hand) {
            for (const {points, label} of poses) {
                knn.learn(points, label)
            }

            input = hand.flatMap(point => [point.x, point.y, point.z])

            await predictHand()
        }

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
     // START GAME
     ********************************************************************/
    async function generateRandomEmoji() {
        // generate random emoji
        let randomNumber = Math.floor(Math.random() * emojis.length)

        // create randomNumber for prompt that isn't the same as previous prompt
        do {
            randomNumber = Math.floor(Math.random() * emojis.length);
        } while (emojis[randomNumber] === previousPrompt);

        // Set new random emoji
        currentPrompt = emojis[randomNumber];

        // update previous prompt for next round
        previousPrompt = currentPrompt

        // give random emoji as html prompt to user
        prompt.innerHTML = `
        <img src="${currentPrompt.img}" alt="${currentPrompt.label}" style="width:100px;">
            <p>${currentPrompt.label}</p>
        `;

        // allow scoring in new round
        hasScoredThisRound = false
    }


    async function predictHand() {
        // predict with knn
        let knnPrediction = knn.classify(input)
        console.log(`KNN: ${knnPrediction}`)

        let nnPrediction = await nn.classify(input)
        prediction = nnPrediction[0].label // get first value of prediction which is most likely to be right
        console.log(`NN: ${prediction}`)

        // if prediction is same as prompt score +1
        if(prediction === currentPrompt.label && !hasScoredThisRound) {
            score++
            scoreView.innerText = `Score: ${score}`;

            // allow only one point per good answer
            hasScoredThisRound = true

            // clear timer and set new prompt
            clearInterval(timerInterval);
            generateRandomEmoji();
            startTimer();
        }
    }

    function startGame() {
        // webcam needs to be on to start
        if(webcamRunning) {
            gameActive = true
            generateRandomEmoji()
            predictWebcam()
            startTimer()
        } else {
            console.log('webcam moet eerst aanstaan!')
            prompt.innerText = 'De webcam moet eerst aanstaan!'
        }
    }

    function restartGame() {
        // reload game
        location.reload()
    }

    function gameOver() {
        prompt.innerText = `Game Over! Score: ${score}`
        webcamRunning = false
        clearInterval(timerInterval)
    }

    function startTimer() {
        // set timeinterval
        timerInterval = setInterval(() => {
            // if not in time live -1
            if(!hasScoredThisRound) {
                lives--
                livesView.innerText = `LIVES ${lives}`

                // if lives 0 or below game over
                if(lives === 0) {
                    console.log('game over!')
                    gameOver()
                } else {
                    generateRandomEmoji() // else create new prompt
                }
            } else {
                // restart timer and allow to score in next round
                clearInterval(timerInterval)
                hasScoredThisRound = false
                startTimer()
            }
        }, timeLimit)
    }

    /********************************************************************
     // START THE APP
     ********************************************************************/
    ( async () => {
        await loadModel()
        await createHandLandmarker()
    })();