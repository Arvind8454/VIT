(() => {
    const video = document.getElementById("webcam-video");
    const canvas = document.getElementById("capture-canvas");
    const startBtn = document.getElementById("start-webcam");
    const stopBtn = document.getElementById("stop-webcam");
    const explainBtn = document.getElementById("explain-btn");
    const modeSelect = document.getElementById("model-mode");
    const labelEl = document.getElementById("prediction-label");
    const confEl = document.getElementById("prediction-confidence");
    const runtimeEl = document.getElementById("prediction-runtime");
    const topkEl = document.getElementById("prediction-topk");
    const statusEl = document.getElementById("prediction-status");
    const explainGradcam = document.getElementById("explain-image-gradcam");
    const explainAttnMap = document.getElementById("explain-image-attnmap");
    const explainRollout = document.getElementById("explain-image-rollout");
    const explainIg = document.getElementById("explain-image-ig");
    const explainPlaceholder = document.getElementById("explain-placeholder");
    const explainTextEl = document.getElementById("explain-text");
    const technicalTextEl = document.getElementById("technical-text");
    const contrastiveTextEl = document.getElementById("contrastive-text");

    if (!video || !canvas || !startBtn || !stopBtn || !explainBtn) {
        return;
    }

    let mediaStream = null;
    let timerId = null;
    let requestInFlight = false;

    const PREDICT_INTERVAL_MS = 2500;

    function setStatus(text) {
        statusEl.textContent = text;
    }

    function renderTopPredictions(topPredictions) {
        if (!topkEl) {
            return;
        }
        const rows = Array.isArray(topPredictions) ? topPredictions : [];
        if (rows.length === 0) {
            topkEl.innerHTML = "<li>No predictions yet.</li>";
            return;
        }
        topkEl.innerHTML = rows
            .map((item) => `<li>${item.label}: ${Number(item.confidence).toFixed(2)}%</li>`)
            .join("");
    }

    function stopStream() {
        if (timerId) {
            clearInterval(timerId);
            timerId = null;
        }
        if (mediaStream) {
            mediaStream.getTracks().forEach((track) => track.stop());
            mediaStream = null;
        }
        video.srcObject = null;
        labelEl.textContent = "--";
        confEl.textContent = "--";
        setStatus("Stopped");
        if (runtimeEl) {
            runtimeEl.textContent = "Device: -- | FP16: --";
        }
        renderTopPredictions([]);
    }

    function captureFrameDataUrl() {
        const width = video.videoWidth || 640;
        const height = video.videoHeight || 480;
        canvas.width = width;
        canvas.height = height;
        const ctx = canvas.getContext("2d");
        ctx.drawImage(video, 0, 0, width, height);
        return canvas.toDataURL("image/jpeg", 0.8);
    }

    async function postFrame(url) {
        const frame = captureFrameDataUrl();
        const response = await fetch(url, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                frame,
                model_mode: modeSelect.value,
            }),
        });
        return response.json();
    }

    async function runPrediction() {
        if (!mediaStream || requestInFlight || video.readyState < 2) {
            return;
        }
        requestInFlight = true;
        setStatus("Predicting...");
        try {
            const data = await postFrame("/webcam-predict");
            if (data.error) {
                setStatus(`Error: ${data.error}`);
            } else {
                labelEl.textContent = data.label;
                confEl.textContent = `${data.confidence}%`;
                setStatus(`Running (${data.model_mode})`);
                if (runtimeEl) {
                    runtimeEl.textContent = `Device: ${data.device || "--"} | FP16: ${Boolean(data.fp16_enabled)}`;
                }
                renderTopPredictions(data.top_predictions || []);
            }
        } catch (err) {
            setStatus("Prediction failed");
        } finally {
            requestInFlight = false;
        }
    }

    async function startWebcam() {
        if (mediaStream) {
            return;
        }
        const host = window.location.hostname;
        const isLocalhost = host === "localhost" || host === "127.0.0.1";
        if (!window.isSecureContext && !isLocalhost) {
            setStatus("Camera requires HTTPS on mobile/browser security policy");
            return;
        }
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            setStatus("Camera API not supported in this browser");
            return;
        }
        try {
            const constraints = {
                video: {
                    width: { ideal: 640 },
                    height: { ideal: 480 },
                    facingMode: "user",
                },
                audio: false,
            };
            mediaStream = await navigator.mediaDevices.getUserMedia(constraints);
            video.srcObject = mediaStream;
            await video.play();
            setStatus("Running");
            await runPrediction();
            timerId = setInterval(runPrediction, PREDICT_INTERVAL_MS);
        } catch (err) {
            try {
                mediaStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
                video.srcObject = mediaStream;
                await video.play();
                setStatus("Running");
                await runPrediction();
                timerId = setInterval(runPrediction, PREDICT_INTERVAL_MS);
            } catch (_fallbackErr) {
                setStatus("Camera start failed. Check browser permissions.");
            }
        }
    }

    async function explainCurrentFrame() {
        if (!mediaStream) {
            setStatus("Start webcam first");
            return;
        }
        setStatus("Generating explanation...");
        try {
            const data = await postFrame("/webcam-explain");
            if (data.error) {
                setStatus(`Error: ${data.error}`);
                return;
            }
            if (data.gradcam) {
                explainGradcam.src = data.gradcam;
                explainGradcam.classList.remove("hidden");
            } else {
                explainGradcam.classList.add("hidden");
            }
            if (data.attention_map) {
                explainAttnMap.src = data.attention_map;
                explainAttnMap.classList.remove("hidden");
            } else {
                explainAttnMap.classList.add("hidden");
            }
            if (data.attention_rollout) {
                explainRollout.src = data.attention_rollout;
                explainRollout.classList.remove("hidden");
            } else {
                explainRollout.classList.add("hidden");
            }
            if (data.integrated_gradients && explainIg) {
                explainIg.src = data.integrated_gradients;
                explainIg.classList.remove("hidden");
            } else if (explainIg) {
                explainIg.classList.add("hidden");
            }
            if (explainTextEl) {
                explainTextEl.textContent = data.explanation && data.explanation.trim()
                    ? data.explanation
                    : "Explanation unavailable for this frame.";
            }
            if (technicalTextEl) {
                technicalTextEl.textContent = data.technical_explanation && data.technical_explanation.trim()
                    ? data.technical_explanation
                    : "Technical explanation unavailable for this frame.";
            }
            if (contrastiveTextEl) {
                contrastiveTextEl.textContent = data.contrastive_explanation && data.contrastive_explanation.trim()
                    ? data.contrastive_explanation
                    : "Contrastive explanation unavailable for this frame.";
            }
            explainPlaceholder.classList.add("hidden");
            labelEl.textContent = data.label;
            confEl.textContent = `${data.confidence}%`;
            if (runtimeEl) {
                runtimeEl.textContent = `Device: ${data.device || "--"} | FP16: ${Boolean(data.fp16_enabled)}`;
            }
            renderTopPredictions(data.top_predictions || []);
            setStatus("Explanation ready");
        } catch (err) {
            setStatus("Explanation failed");
        }
    }

    startBtn.addEventListener("click", startWebcam);
    stopBtn.addEventListener("click", stopStream);
    explainBtn.addEventListener("click", explainCurrentFrame);
    window.addEventListener("beforeunload", stopStream);
})();
