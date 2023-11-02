// Load package from CDN URL for demo/test or install using NPM/Yarn for dev/prod
import vision from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0";  
const {ImageSegmenter, FaceLandmarker, FilesetResolver } = vision;
let imageSegmenter;
let faceLandmarker;
let labels;
let runningMode = "VIDEO";
let enableFaceMakeup = true;
let enableHairColor = false;
let gpuStatus = false;
let hairModelPath;


// CHECK DEVICE GPU/CAMERA/BROWSER COMPATABILITY ETC.
async function gpuInit() {
    if (!navigator.gpu) {
        hairModelPath = "https://raw.githubusercontent.com/srishti-buyume/vto-models/main/hair_small.tflite";
        document.getElementById("gpuMsg").textContent = "GPU not supported, Loading Small Hair Model ";
        throw Error("WebGPU not supported. Reload App.");  
    } else {
        gpuStatus = true;
        document.getElementById("gpuMsg").textContent = "GPU supported, Loading Large Hair Model ";
        hairModelPath = "https://raw.githubusercontent.com/srishti-buyume/vto-models/main/hair_large.tflite";
    }
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
      throw Error("Couldn't request WebGPU adapter. Reload App.");
    }
  }


// LOAD IMAGE SEGMENTATION MODEL WITH SPECIFIED PARAMETERS 
const createImageSegmenter = async () => {
    const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm"); // Again this CDN URL is only for demo purpose, load the package locally when building the app
    imageSegmenter = await ImageSegmenter.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath: hairModelPath,
            delegate: "GPU"
        },
        runningMode: runningMode,
        outputCategoryMask: true,
        outputConfidenceMasks: true
    });
    labels = imageSegmenter.getLabels();
};



// LOAD FACE LANDMARKER MODEL WITH SPECIFIED PARAMETERS 
const createFaceLandmarker = async () => {
    const ultron = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm");
    faceLandmarker = await FaceLandmarker.createFromOptions(ultron, {
        baseOptions: {
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`, // Load the model locally from assets, this CDN URL is only for demo
            delegate: "GPU"
        },
        runningMode: "IMAGE",
        numFaces: 1
    });
};



// MAIN FUNCTION FOR ALL VIRTUAL TRY ON FEATURES
function virtualTryon() 
{
    let video = document.getElementById("webcam");
    let canvasElement = document.getElementById("canvas1");
    let canvasElement2 = document.getElementById("canvas2");
    const canvasCtx = canvasElement.getContext("2d", { willReadFrequently: true })
    const canvasCtx2 = canvasElement2.getContext("2d", { willReadFrequently: true })
    const ctx = canvasElement.getContext("2d", { willReadFrequently: true })
    let enableWebcamButton;
    let webcamRunning = false;
    let legendColors = [ 
        [90, 30, 31, 200], // Default Hair Color- RGBA
        [90, 30, 31, 255], // Default Lip Color 
        [201, 15, 40, 255], // Default Blush Color
        [228, 93, 125, 255], // Default Shadow Color
        [0, 0, 0, 255],  // Default Liner Color
    ]; 

    // Run hair coloring on live webcam feed
    function hairColorForVideo(result) {
        canvasElement.style.display = 'block';
        canvasElement2.style.display = 'block';
        let imageData = canvasCtx.getImageData(0, 0, video.videoWidth, video.videoHeight).data;
        const mask = result.categoryMask.getAsUint8Array();
        for (let i in mask) {
            if(labels[mask[i]] == "hair"){
            const legendColor = legendColors[0];
            imageData[i * 4 + 0] = (legendColor[0] + imageData[i * 4 + 0]) / 2;
            imageData[i * 4 + 1] = (legendColor[1] + imageData[i * 4 + 1]) / 2;
            imageData[i * 4 + 2] = (legendColor[2] + imageData[i * 4 + 2]) / 2;
            imageData[i * 4 + 3] = (legendColor[3] + imageData[i * 4 + 3]) / 2;
            }
        }
        const uint8Array = new Uint8ClampedArray(imageData.buffer);
        const dataNew = new ImageData(uint8Array, video.videoWidth, video.videoHeight);
        canvasCtx.imageSmoothingEnabled = true;
        canvasCtx.putImageData(dataNew, 0, 0);
        canvasElement.style.filter = "blur(0px) brightness(110%) contrast(120%)"; 
        if (webcamRunning === true) {
            window.requestAnimationFrame(predictWebcam);
        }   
    }


    // Run hair coloring for image input
    function hairColorForImage(result) { 
        const cxt = canvasClick.getContext("2d");
        const { width, height } = result.categoryMask;
        let imageData = cxt.getImageData(0, 0, width, height).data;
        canvasClick.width = width;
        canvasClick.height = height;
        const mask = result.categoryMask.getAsUint8Array();
        for (let i in mask) {  
            if(labels[mask[i]] === "hair"){
            const legendColor = legendColors[0];
            imageData[i * 4 + 0] = (legendColor[0] + imageData[i * 4 + 0]) / 2;
            imageData[i * 4 + 1] = (legendColor[1] + imageData[i * 4 + 1]) / 2;
            imageData[i * 4 + 2] = (legendColor[2] + imageData[i * 4 + 2]) / 2;
            imageData[i * 4 + 3] = (legendColor[3] + imageData[i * 4 + 3]) / 2;
            }
        }
        const uint8Array = new Uint8ClampedArray(imageData.buffer);
        const dataNew = new ImageData(uint8Array, width, height);
        canvasClick.imageSmoothingEnabled = true;
        cxt.putImageData(dataNew, 0, 0);
    }


    // Run makeup VTO on live webcam feed 
    // Currently only Lip coloring is given as example to keep the code simple
    function makeupForVideo(landmarks) {
        canvasElement.style.display = 'block';
        try{
            const landmarkLips1= [
                landmarks[0][61],landmarks[0][40],landmarks[0][39],landmarks[0][37],landmarks[0][0],landmarks[0][267],landmarks[0][269],landmarks[0][270],landmarks[0][409],
                landmarks[0][306], landmarks[0][415], landmarks[0][310],landmarks[0][311],landmarks[0][312],landmarks[0][13],landmarks[0][82],landmarks[0][81],landmarks[0][42],
                landmarks[0][183],landmarks[0][61]
                ];
            const landmarkLips2= [
                landmarks[0][61], landmarks[0][146],landmarks[0][91], landmarks[0][181],
                landmarks[0][84], landmarks[0][17], landmarks[0][314],landmarks[0][405], landmarks[0][321], landmarks[0][375], landmarks[0][306], 
                landmarks[0][409], landmarks[0][324], landmarks[0][318], landmarks[0][402],landmarks[0][317], 
                landmarks[0][14], landmarks[0][87], landmarks[0][178], landmarks[0][88],landmarks[0][95], landmarks[0][61]
                ]; 
            ctx.imageSmoothingEnabled = true;
            const baseColor = { r: legendColors[1][0], g: legendColors[1][1], b: legendColors[1][2], a: 0.6 } ; //change alpha change for opacity
            ctx.fillStyle = `rgba(${baseColor.r}, ${baseColor.g}, ${baseColor.b}, ${baseColor.a})`;
            ctx.strokeStyle = `rgba(${baseColor.r}, ${baseColor.g}, ${baseColor.b}, ${baseColor.a})`;
        
            ctx.beginPath();
            ctx.lineWidth= 0.9;
            landmarkLips1.forEach(point => { ctx.arc(point.x * canvasElement.width, point.y * canvasElement.height, 0, 0, Math.PI * 2) }); 
            //   landmarkLips2.forEach(point => { ctx.arc(point.x * canvasElement.width, point.y * canvasElement.height, 0, 0, Math.PI * 2) }); 
            ctx.fill();
            ctx.stroke(); 
            ctx.closePath();

            ctx.beginPath();
            ctx.lineWidth= 0.9;
            landmarkLips2.forEach(point => { ctx.arc(point.x * canvasElement.width, point.y * canvasElement.height, 0, 0, Math.PI * 2) }); 
            ctx.fill();
            ctx.stroke(); 
            ctx.closePath();
            canvasElement.style.filter = "blur(0px) brightness(110%) contrast(120%)"; 

            if (webcamRunning === true) {
                window.requestAnimationFrame(predictWebcam);
            }   

        } catch {
            document.getElementById("makeupMsg").textContent = " Face out of frame, Reload or Enable/Disable Button";
        }
    }


    // Run makeup VTO for image input
    function makeupForImage(landmarks, canvas) {  
        const ctx = canvas.getContext('2d');
        ctx.imageSmoothingEnabled = true;

        if (enableFaceMakeup) {  
            // Drawing Eye Liner Effect on Face
            const landmarkEyeliner1= [
                landmarks[0][362], landmarks[0][398], landmarks[0][384], landmarks[0][385], landmarks[0][386],
                landmarks[0][387], landmarks[0][388], landmarks[0][466], landmarks[0][263], landmarks[0][359]
            ];
            const landmarkEyeliner2= [
                landmarks[0][133], landmarks[0][173], landmarks[0][157], landmarks[0][158], landmarks[0][159], 
                landmarks[0][160], landmarks[0][161], landmarks[0][246], landmarks[0][33], landmarks[0][130]
            ];
            const linerBaseColor = { r: legendColors[4][0], g: legendColors[4][1], b: legendColors[4][2], a: 0.8}
            ctx.strokeStyle = `rgba(${linerBaseColor.r}, ${linerBaseColor.g}, ${linerBaseColor.b}, ${linerBaseColor.a})`;
            ctx.fillStyle = `rgba(${linerBaseColor.r}, ${linerBaseColor.g}, ${linerBaseColor.b}, ${linerBaseColor.a})`;
        
            ctx.beginPath();
            ctx.lineWidth = 3;  
            landmarkEyeliner1.forEach(point => {ctx.arc(point.x * canvas.width, point.y * canvas.height, 0, 0, Math.PI * 2)});    
            ctx.stroke();
            ctx.closePath();
        
            ctx.beginPath();
            ctx.lineWidth = 3;
            landmarkEyeliner2.forEach(point => { ctx.arc(point.x * canvas.width, point.y * canvas.height, 0, 0, Math.PI * 2) });
            ctx.stroke();
            ctx.closePath();   


            // Drawing Lip Stick Effect on Face
            const landmarkLips1= [
                landmarks[0][61], landmarks[0][40], landmarks[0][39], landmarks[0][37], landmarks[0][0], landmarks[0][267], landmarks[0][269], 
                landmarks[0][270], landmarks[0][409], landmarks[0][306], landmarks[0][415], landmarks[0][310], landmarks[0][311], landmarks[0][312],
                landmarks[0][13], landmarks[0][82], landmarks[0][81], landmarks[0][42], landmarks[0][183], landmarks[0][61]
            ];
            const landmarkLips2= [
                landmarks[0][61], landmarks[0][146], landmarks[0][91], landmarks[0][181], landmarks[0][84], landmarks[0][17], landmarks[0][314],
                landmarks[0][405], landmarks[0][321], landmarks[0][375], landmarks[0][306], landmarks[0][409], landmarks[0][324], landmarks[0][318], 
                landmarks[0][402], landmarks[0][317], landmarks[0][14], landmarks[0][87], landmarks[0][178], landmarks[0][88], landmarks[0][95], landmarks[0][61], 
            ];
            const lipBaseColor = { r: legendColors[1][0], g: legendColors[1][1], b: legendColors[1][2], a: 0.6} ; //{ r: 90, g: 30, b: 31, a: 0.6 };
            ctx.fillStyle = `rgba(${lipBaseColor.r}, ${lipBaseColor.g}, ${lipBaseColor.b}, ${lipBaseColor.a})`;
            ctx.strokeStyle = `rgba(${lipBaseColor.r}, ${lipBaseColor.g}, ${lipBaseColor.b}, ${lipBaseColor.a})`;
        
            ctx.beginPath();
            ctx.lineWidth= 0.9;
            landmarkLips1.forEach(point => { ctx.arc(point.x * canvas.width, point.y * canvas.height, 0, 0, Math.PI * 2) }); //upper lip
            ctx.fill(); 
            ctx.stroke();  
            ctx.closePath();

            ctx.beginPath();
            ctx.lineWidth= 0.9;
            landmarkLips2.forEach(point => { ctx.arc(point.x * canvas.width, point.y * canvas.height, 0, 0, Math.PI * 2) }); //upper lip
            ctx.fill(); 
            ctx.stroke();  
            ctx.closePath();
          

            // Drawing Blush Effect on Face
            // else if(enableFaceBlush){
            const landmarkCheeksLeft= [
                landmarks[0][119], landmarks[0][116], landmarks[0][123], landmarks[0][147],
                landmarks[0][187], landmarks[0][205], landmarks[0][36], landmarks[0][119] 
            ];
            const landmarkCheeksRight= [
                landmarks[0][348], landmarks[0][345], landmarks[0][352], landmarks[0][376],
                landmarks[0][411], landmarks[0][425], landmarks[0][266], landmarks[0][348] 
            ];
            ctx.globalAlpha = 0.55;
            ctx.filter = 'blur(6px)';
            const blushBaseColor = { r: legendColors[2][0], g: legendColors[2][1], b: legendColors[2][2], a: 0.3} ;
            const blushEnhancedColor = { ...blushBaseColor, a: 0.7};
            blushEnhancedColor.r += 36;
            blushEnhancedColor.g += 76;
            blushEnhancedColor.b -= 16;
            ctx.strokeStyle = `rgba(${blushBaseColor.r}, ${blushBaseColor.g}, ${blushBaseColor.b}, ${blushBaseColor.a})`; // ctx.strokeStyle = 'rgba(201, 15, 40, 0.3)';
            ctx.shadowBlur = 0;

            // Gradient filter for Left Cheek
            ctx.beginPath();
            const gradientLeftCheek = ctx.createRadialGradient(landmarks[0][50].x * canvas.width, landmarks[0][50].y * canvas.height, 0, landmarks[0][50].x * canvas.width, landmarks[0][50].y * canvas.height, 50);
            gradientLeftCheek.addColorStop(0, `rgba(${blushEnhancedColor.r}, ${blushEnhancedColor.g}, ${blushEnhancedColor.b}, ${blushEnhancedColor.a})`);
            gradientLeftCheek.addColorStop(1, `rgba(${blushBaseColor.r}, ${blushBaseColor.g}, ${blushBaseColor.b}, 0)`);
            ctx.fillStyle = gradientLeftCheek;
            landmarkCheeksLeft.forEach(point => { ctx.arc(point.x * canvas.width, point.y * canvas.height, 0, 0, Math.PI * 2) });
            ctx.fill();

            // Gradient filter for Right Cheek
            const gradientRightCheek = ctx.createRadialGradient( landmarks[0][280].x * canvas.width, landmarks[0][280].y * canvas.height, 0, landmarks[0][280].x * canvas.width, landmarks[0][280].y * canvas.height, 50);
            gradientRightCheek.addColorStop(0, `rgba(${blushEnhancedColor.r}, ${blushEnhancedColor.g}, ${blushEnhancedColor.b}, ${blushEnhancedColor.a})`);
            gradientRightCheek.addColorStop(1, `rgba(${blushBaseColor.r}, ${blushBaseColor.g}, ${blushBaseColor.b}, 0)`);
            ctx.fillStyle = gradientRightCheek;
            landmarkCheeksRight.forEach(point => {ctx.arc(point.x * canvas.width, point.y * canvas.height, 0, 0, Math.PI * 2)});
            ctx.fill();
            ctx.closePath()  
            // }


            // Drawing Eye Shadow Effect on Face
            // else if(enableEyeShadow){
            const landmarkEyeshadow1= [ 
                landmarks[0][113], landmarks[0][225], landmarks[0][224], landmarks[0][223], landmarks[0][222], landmarks[0][221], 
                landmarks[0][189], landmarks[0][190], landmarks[0][173], landmarks[0][157], landmarks[0][158], landmarks[0][159], 
                landmarks[0][160], landmarks[0][161], landmarks[0][246], landmarks[0][33], landmarks[0][130], landmarks[0][113] 
            ];
            const landmarkEyeshadow2= [ 
                landmarks[0][413], landmarks[0][441], landmarks[0][442], landmarks[0][443], landmarks[0][444], landmarks[0][445], 
                landmarks[0][342], landmarks[0][263], landmarks[0][466], landmarks[0][388], landmarks[0][387], landmarks[0][386], 
                landmarks[0][385], landmarks[0][384], landmarks[0][398], landmarks[0][362], landmarks[0][413] 
            ];
            ctx.globalAlpha= 0.5
            const shadowBaseColor = { r: legendColors[3][0], g: legendColors[3][1], b: legendColors[3][2], a: 0.9} ; //{ r: 228, g: 93, b: 125, a: 0.5 };
            const shadowEnhancedColor = { ...shadowBaseColor, a: 0.8}; // a=0.2
            shadowEnhancedColor.r -= 10;
            shadowEnhancedColor.g -= 20;
            shadowEnhancedColor.b -= 20;

            // ctx.fillStyle = gradient;
            ctx.fillStyle = `rgba(${shadowEnhancedColor.r}, ${shadowEnhancedColor.g}, ${shadowEnhancedColor.b}, ${shadowEnhancedColor.a})`;
            const glossyGradient = ctx.createRadialGradient( canvas.width / 2, canvas.height / 2, 0, canvas.width / 2, canvas.height / 2, canvas.width / 4);
            glossyGradient.addColorStop(0, `rgba(${shadowEnhancedColor.r}, ${shadowEnhancedColor.g}, ${shadowEnhancedColor.b}, ${shadowEnhancedColor.a})`);
            glossyGradient.addColorStop(1, `rgba(${shadowEnhancedColor.r}, ${shadowEnhancedColor.g}, ${shadowEnhancedColor.b}, 0.5)`);
   
            ctx.beginPath();
            landmarkEyeshadow1.forEach(point => {ctx.arc(point.x * canvas.width, point.y * canvas.height, 0, 0, Math.PI * 2)});
            landmarkEyeshadow2.forEach(point => {ctx.arc(point.x * canvas.width, point.y * canvas.height, 0, 0, Math.PI * 2)}); 
            ctx.fill();
            ctx.strokeStyle = `rgba(${shadowEnhancedColor.r - 20}, ${shadowEnhancedColor.g - 20}, ${shadowEnhancedColor.b - 20}, ${shadowEnhancedColor.a})`;
            ctx.shadowColor = `rgba(${shadowEnhancedColor.r}, ${shadowEnhancedColor.g}, ${shadowEnhancedColor.b}, ${shadowEnhancedColor.a})`;
            ctx.shadowBlur = 8.0;
            ctx.lineJoin = 'round';
            ctx.fillStyle = glossyGradient;
            ctx.closePath(); 
            //}   

        }
           
    }
  

    // UTILITY FUNCTIONS TO HANDLE IMAGE/VIDEO DATA, MAKEUP/HAIR TRYON ETC.

    // Get Image DOM Elements
    const imageContainers = document.getElementsByClassName("segmentOnClick");
    for (let i = 0; i < imageContainers.length; i++) {
        imageContainers[i]
            .getElementsByTagName("img")[0]
            .addEventListener("click", handleClick);
    }


    // Handle Image click event from User
    let canvasClick;
    let startTime;
    async function handleClick(event) {
        startTime = performance.now();
        if (imageSegmenter === undefined) {
            return;
        }
        if (faceLandmarker === undefined) {
            return;
        }
        canvasClick = event.target.parentElement.getElementsByTagName("canvas")[0];
        canvasClick.classList.remove("removed");
        canvasClick.width = event.target.naturalWidth;
        canvasClick.height = event.target.naturalHeight;
        const cxt = canvasClick.getContext("2d");
        cxt.clearRect(0, 0, canvasClick.width, canvasClick.height);
        cxt.drawImage(event.target, 0, 0, canvasClick.width, canvasClick.height);
        event.target.style.opacity = 0;
        // if VIDEO mode is initialized, set runningMode to IMAGE
        if (runningMode === "VIDEO" || runningMode === "LIVE_STREAM") {
            runningMode = "IMAGE";
            await imageSegmenter.setOptions({
                runningMode: runningMode
            });
            await faceLandmarker.setOptions({
                runningMode: runningMode
            });
        }

        if (enableFaceMakeup){
            const faceLandmarkerResult = faceLandmarker.detect(event.target);
            makeupForImage(faceLandmarkerResult.faceLandmarks, canvasClick);
        }

        else if (enableHairColor){
            imageSegmenter.segment(event.target, hairColorForImage);
        }

    }


    // Check if webcam access is supported.
    function hasGetUserMedia() {
        return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
    }

    // Get segmentation from the webcam
    let lastWebcamTime = -1;
    async function predictWebcam() {
        if (video.currentTime === lastWebcamTime) {
            if (webcamRunning === true) {
                window.requestAnimationFrame(predictWebcam);
            }
            return;
        }
        lastWebcamTime = video.currentTime;
        canvasCtx.drawImage(video, 0, 0, video.videoWidth, video.videoHeight);
        canvasCtx2.drawImage(video, 0, 0, video.videoWidth, video.videoHeight);
        if (imageSegmenter === undefined) {
            return;
        }
        if (runningMode === "IMAGE") {
            runningMode = "VIDEO";
            await imageSegmenter.setOptions({
                runningMode: runningMode
            });
            await faceLandmarker.setOptions({
                runningMode: "IMAGE"
            });
        }
  
        if (enableFaceMakeup) {
            const faceLandmarkerResult = faceLandmarker.detect(video);
            makeupForVideo(faceLandmarkerResult.faceLandmarks);
        }
        
        else if (enableHairColor && gpuStatus) {
            let startTimeMs = performance.now();
            imageSegmenter.segmentForVideo(video, startTimeMs, hairColorForVideo);
        }
 
    }


    // Enable the live webcam view and start imageSegmentation.
    async function enableCam(event) {
        if (imageSegmenter === undefined) {
            return;
        }
        if (webcamRunning === true) {
            webcamRunning = false;
            enableWebcamButton.innerText = "ENABLE LIVE TRY-ON";
        }
        else {
            webcamRunning = true;
            enableWebcamButton.innerText = "DISABLE LIVE TRY-ON";
        }
        const constraints = { video: true, audio: false};
        video = document.getElementById("webcam");
        video.srcObject = await navigator.mediaDevices.getUserMedia(constraints);
        video.addEventListener("loadeddata", predictWebcam);
        video.play();
        video.style.display = 'none';
    }


    // If webcam supported, add event listener to button.
    if (hasGetUserMedia()) {
        enableWebcamButton = document.getElementById("webcamButton");
        enableWebcamButton.addEventListener("click", enableCam);
    }
    else {
        console.warn("getUserMedia() is not supported by your browser");
    }



    function toggleFaceMakeup() {
        if(enableHairColor == true) {
            enableHairColor = false;
            const button = document.getElementById("hairColorButton");
            button.innerText = enableHairColor ? "Disable Hair Color" : "Enable Hair Color";
        }
        enableFaceMakeup = !enableFaceMakeup;
        const button = document.getElementById("faceMakeupButton");
        button.innerText = enableFaceMakeup ? "Disable Face Makeup" : "Enable Face Makeup";
    }

    function toggleHairColor() {
        if(enableFaceMakeup == true) {
            enableFaceMakeup = false;
            const button = document.getElementById("faceMakeupButton");
            button.innerText = enableFaceMakeup ? "Disable Face Makeup" : "Enable Face Makeup";
        }
        enableHairColor = !enableHairColor;
        const button = document.getElementById("hairColorButton");
        button.innerText = enableHairColor ? "Disable Hair Color" : "Enable Hair Color";
    }



    // Add a click event listener to the Enable/Disable Makuep Options
    document.getElementById("faceMakeupButton").addEventListener("click", toggleFaceMakeup);
    document.getElementById("hairColorButton").addEventListener("click", toggleHairColor);



    //ADD CONTROLS FOR COLOR, OPACITY AND BLUR
    const hexToRgb = hex =>
        hex.replace(/^#?([a-f\d])([a-f\d])([a-f\d])$/i, (m, r, g, b) => '#' + r + r + g + g + b + b)
            .substring(1).match(/.{2}/g)
            .map(x => parseInt(x, 16))

    function colVal() {
            let d = document.getElementById("color").value;
            let hex = hexToRgb(d);
            hex[3] = 200; // adjust alpha channel -> also called opacity/transparency channel
            legendColors[0] = hex;
            legendColors[1] = hex;
            legendColors[2] = hex;
            legendColors[3] = hex;
        }

    function blurVal() {
        let x = document.getElementById("blur").value;
        // console.log(x);
        document.getElementById('canvas1').style.filter = 'blur('+x+'px)';
        }

    function opVal() {
        let z = document.getElementById("opacity").value;
        // console.log(z);
        document.getElementById('canvas1').style.opacity = z;
        }

    document.getElementById("color").addEventListener("input", colVal);
    document.getElementById("blur").addEventListener("input", blurVal);
    document.getElementById("opacity").addEventListener("input", opVal);

}


gpuInit(); 
createImageSegmenter();
createFaceLandmarker();
virtualTryon();