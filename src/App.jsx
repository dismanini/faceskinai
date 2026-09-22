import { useCallback, useEffect, useRef, useState } from "react";
import {
  FaceLandmarker,
  FilesetResolver,
} from "@mediapipe/tasks-vision";
import "./App.css";

const MEDIAPIPE_WASM =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm";

const FACE_MODEL =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";

const PARAMETERS = [
  {
    key: "acne",
    label: "Acne",
    icon: "🔴",
  },
  {
    key: "pigmentation",
    label: "Pigmentation",
    icon: "🟤",
  },
  {
    key: "wrinkles",
    label: "Fine Lines",
    icon: "〰️",
  },
  {
    key: "pores",
    label: "Pores",
    icon: "⚫",
  },
  {
    key: "redness",
    label: "Redness",
    icon: "🌡️",
  },
  {
    key: "underEye",
    label: "Under Eye",
    icon: "👁️",
  },
  {
    key: "hydration",
    label: "Hydration",
    icon: "💧",
  },
  {
    key: "texture",
    label: "Texture",
    icon: "✨",
  },
];

function clamp(value, min = 0, max = 100) {
  return Math.max(min, Math.min(max, value));
}

function average(values) {
  if (!values.length) return 0;

  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function distance(a, b) {
  return Math.sqrt(
    Math.pow(a.x - b.x, 2) +
      Math.pow(a.y - b.y, 2)
  );
}

function getPoint(landmarks, index) {
  return landmarks[index] || { x: 0.5, y: 0.5 };
}

function createCanvasFromVideo(video) {
  const canvas = document.createElement("canvas");

  canvas.width = video.videoWidth || 640;
  canvas.height = video.videoHeight || 480;

  const ctx = canvas.getContext("2d");

  ctx.drawImage(
    video,
    0,
    0,
    canvas.width,
    canvas.height
  );

  return canvas;
}

function sampleRegion(canvas, landmarks, indexes) {
  const ctx = canvas.getContext("2d");

  const width = canvas.width;
  const height = canvas.height;

  const points = indexes.map((index) =>
    getPoint(landmarks, index)
  );

  const xs = points.map((p) => p.x * width);
  const ys = points.map((p) => p.y * height);

  const minX = Math.max(0, Math.floor(Math.min(...xs)));
  const maxX = Math.min(width - 1, Math.ceil(Math.max(...xs)));

  const minY = Math.max(0, Math.floor(Math.min(...ys)));
  const maxY = Math.min(height - 1, Math.ceil(Math.max(...ys)));

  const regionWidth = Math.max(1, maxX - minX);
  const regionHeight = Math.max(1, maxY - minY);

  const imageData = ctx.getImageData(
    minX,
    minY,
    regionWidth,
    regionHeight
  );

  const pixels = imageData.data;

  const brightness = [];
  const redness = [];
  const saturation = [];

  for (let i = 0; i < pixels.length; i += 4) {
    const r = pixels[i];
    const g = pixels[i + 1];
    const b = pixels[i + 2];

    const max = Math.max(r, g, b);
    const min = Math.min(r, g, b);

    const bright = (r + g + b) / 3;

    brightness.push(bright);

    redness.push(
      clamp(
        ((r - (g + b) / 2) / 255) * 100 + 50
      )
    );

    saturation.push(
      max === 0
        ? 0
        : ((max - min) / max) * 100
    );
  }

  return {
    brightness: average(brightness),
    redness: average(redness),
    saturation: average(saturation),
  };
}

function calculateSkinAnalysis(canvas, landmarks) {
  /*
    These landmark indexes are used as approximate
    facial regions.

    MediaPipe provides 478 landmarks in supported
    configurations. We use selected points to estimate
    areas for the prototype.
  */

  const forehead = sampleRegion(
    canvas,
    landmarks,
    [10, 67, 109, 103, 104, 108]
  );

  const leftCheek = sampleRegion(
    canvas,
    landmarks,
    [116, 117, 118, 119, 120, 121]
  );

  const rightCheek = sampleRegion(
    canvas,
    landmarks,
    [345, 346, 347, 348, 349, 350]
  );

  const nose = sampleRegion(
    canvas,
    landmarks,
    [1, 2, 4, 5, 6, 195]
  );

  const chin = sampleRegion(
    canvas,
    landmarks,
    [152, 148, 176, 377, 400]
  );

  const leftUnderEye = sampleRegion(
    canvas,
    landmarks,
    [33, 133, 159, 145]
  );

  const rightUnderEye = sampleRegion(
    canvas,
    landmarks,
    [362, 263, 386, 374]
  );

  const skinRegions = [
    forehead,
    leftCheek,
    rightCheek,
    nose,
    chin,
  ];

  const brightness = average(
    skinRegions.map((r) => r.brightness)
  );

  const redness = average(
    skinRegions.map((r) => r.redness)
  );

  const saturation = average(
    skinRegions.map((r) => r.saturation)
  );

  const underEyeBrightness = average([
    leftUnderEye.brightness,
    rightUnderEye.brightness,
  ]);

  /*
   * Prototype scoring.
   *
   * These are NOT trained AI models.
   * They are placeholders so the application
   * can produce dynamic results.
   */

  const acneScore = clamp(
    100 -
      Math.abs(brightness - 125) * 0.45 -
      Math.abs(redness - 50) * 0.3
  );

  const pigmentationScore = clamp(
    100 -
      Math.abs(brightness - 135) * 0.6 -
      saturation * 0.25
  );

  const wrinkleScore = clamp(
    100 -
      Math.abs(brightness - 130) * 0.35
  );

  const poreScore = clamp(
    100 - saturation * 0.55
  );

  const rednessScore = clamp(
    100 - Math.abs(redness - 50) * 1.4
  );

  const underEyeScore = clamp(
    100 -
      Math.abs(
        underEyeBrightness - brightness
      ) *
        1.5
  );

  const hydrationScore = clamp(
    brightness * 0.62 +
      (100 - saturation) * 0.38
  );

  const textureScore = clamp(
    100 -
      Math.abs(brightness - 135) * 0.4 -
      saturation * 0.15
  );

  const results = {
    acne: Math.round(acneScore),
    pigmentation: Math.round(pigmentationScore),
    wrinkles: Math.round(wrinkleScore),
    pores: Math.round(poreScore),
    redness: Math.round(rednessScore),
    underEye: Math.round(underEyeScore),
    hydration: Math.round(hydrationScore),
    texture: Math.round(textureScore),
  };

  const overallScore = Math.round(
    average(Object.values(results))
  );

  const skinType = determineSkinType(
    brightness,
    saturation
  );

  return {
    ...results,
    overallScore,
    skinType,
  };
}

function determineSkinType(brightness, saturation) {
  if (brightness < 85) {
    return "Dry / Dehydrated";
  }

  if (saturation > 45) {
    return "Oily";
  }

  if (brightness > 155) {
    return "Normal";
  }

  return "Combination";
}

function getSeverity(score) {
  if (score >= 80) return "Good";
  if (score >= 60) return "Moderate";
  if (score >= 40) return "Needs Attention";

  return "High Concern";
}

function getRecommendations(results) {
  const recommendations = [];

  if (results.acne < 65) {
    recommendations.push(
      "Use a gentle cleanser and avoid harsh scrubbing."
    );
  }

  if (results.pigmentation < 65) {
    recommendations.push(
      "Consider daily broad-spectrum sunscreen and gentle brightening skincare."
    );
  }

  if (results.wrinkles < 65) {
    recommendations.push(
      "Focus on sun protection and consistent moisturizing."
    );
  }

  if (results.hydration < 65) {
    recommendations.push(
      "Increase hydration-focused skincare such as moisturizer and hydrating serum."
    );
  }

  if (results.redness < 65) {
    recommendations.push(
      "Prefer gentle, fragrance-free skincare and avoid over-exfoliation."
    );
  }

  if (results.texture < 65) {
    recommendations.push(
      "Use a gentle cleansing and moisturizing routine consistently."
    );
  }

  if (!recommendations.length) {
    recommendations.push(
      "Maintain your current skincare routine and daily sun protection."
    );
  }

  return recommendations;
}

function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);

  const streamRef = useRef(null);
  const faceLandmarkerRef = useRef(null);
  const animationRef = useRef(null);

  const [cameraStarted, setCameraStarted] =
    useState(false);

  const [loading, setLoading] = useState(false);

  const [modelReady, setModelReady] =
    useState(false);

  const [faceDetected, setFaceDetected] =
    useState(false);

  const [capturedImage, setCapturedImage] =
    useState(null);

  const [landmarks, setLandmarks] =
    useState(null);

  const [analysis, setAnalysis] =
    useState(null);

  const [error, setError] = useState("");

  const stopCamera = useCallback(() => {
    if (animationRef.current) {
      cancelAnimationFrame(animationRef.current);
      animationRef.current = null;
    }

    if (streamRef.current) {
      streamRef.current
        .getTracks()
        .forEach((track) => track.stop());

      streamRef.current = null;
    }

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    setCameraStarted(false);
  }, []);

  useEffect(() => {
    return () => {
      stopCamera();
    };
  }, [stopCamera]);

  const loadModel = async () => {
    try {
      setLoading(true);
      setError("");

      const vision =
        await FilesetResolver.forVisionTasks(
          MEDIAPIPE_WASM
        );

      const landmarker =
        await FaceLandmarker.createFromOptions(
          vision,
          {
            baseOptions: {
              modelAssetPath: FACE_MODEL,
              delegate: "GPU",
            },

            runningMode: "VIDEO",

            numFaces: 1,

            minFaceDetectionConfidence: 0.5,

            minFacePresenceConfidence: 0.5,

            minTrackingConfidence: 0.5,

            outputFaceBlendshapes: false,

            outputFacialTransformationMatrixes: false,
          }
        );

      faceLandmarkerRef.current = landmarker;

      setModelReady(true);
    } catch (err) {
      console.error(err);

      setError(
        "Unable to load the MediaPipe face model. Please check your internet connection and reload the page."
      );
    } finally {
      setLoading(false);
    }
  };

  const startCamera = async () => {
    try {
      setError("");

      if (!modelReady) {
        await loadModel();
      }

      const stream =
        await navigator.mediaDevices.getUserMedia({
          video: {
            facingMode: "user",
            width: {
              ideal: 1280,
            },
            height: {
              ideal: 720,
            },
          },

          audio: false,
        });

      streamRef.current = stream;

      if (!videoRef.current) return;

      videoRef.current.srcObject = stream;

      await videoRef.current.play();

      setCameraStarted(true);

      detectFace();
    } catch (err) {
      console.error(err);

      setError(
        "Camera permission was denied or the camera is unavailable."
      );
    }
  };

  const detectFace = () => {
    if (
      !videoRef.current ||
      !faceLandmarkerRef.current
    ) {
      return;
    }

    const video = videoRef.current;

    if (video.readyState >= 2) {
      const timestamp =
        performance.now();

      try {
        const result =
          faceLandmarkerRef.current.detectForVideo(
            video,
            timestamp
          );

        if (
          result &&
          result.faceLandmarks &&
          result.faceLandmarks.length > 0
        ) {
          setFaceDetected(true);

          setLandmarks(
            result.faceLandmarks[0]
          );
        } else {
          setFaceDetected(false);
        }
      } catch (err) {
        console.error(
          "Face detection error:",
          err
        );
      }
    }

    animationRef.current =
      requestAnimationFrame(detectFace);
  };

  const captureFace = () => {
    if (!videoRef.current) return;

    if (!faceDetected) {
      setError(
        "Please position your face inside the camera frame."
      );

      return;
    }

    try {
      const canvas =
        createCanvasFromVideo(
          videoRef.current
        );

      const image =
        canvas.toDataURL(
          "image/jpeg",
          0.92
        );

      setCapturedImage(image);

      if (landmarks) {
        const results =
          calculateSkinAnalysis(
            canvas,
            landmarks
          );

        setAnalysis(results);
      }

      stopCamera();
    } catch (err) {
      console.error(err);

      setError(
        "Unable to capture the image."
      );
    }
  };

  const retake = async () => {
    setCapturedImage(null);
    setAnalysis(null);
    setLandmarks(null);
    setFaceDetected(false);
    setError("");

    await startCamera();
  };

  const drawLandmarks = () => {
    const canvas = canvasRef.current;

    if (!canvas || !landmarks) return;

    const ctx = canvas.getContext("2d");

    ctx.clearRect(
      0,
      0,
      canvas.width,
      canvas.height
    );

    canvas.width = 640;
    canvas.height = 480;

    ctx.fillStyle =
      "rgba(255,255,255,0.8)";

    landmarks.forEach((point) => {
      const x = point.x * canvas.width;
      const y = point.y * canvas.height;

      ctx.beginPath();

      ctx.arc(
        x,
        y,
        1.5,
        0,
        Math.PI * 2
      );

      ctx.fill();
    });
  };

  useEffect(() => {
    if (capturedImage && landmarks) {
      drawLandmarks();
    }
  }, [capturedImage, landmarks]);

  const renderScore = (score) => {
    return `${score}/100`;
  };

  const recommendations = analysis
    ? getRecommendations(analysis)
    : [];

  return (
    <div className="app">
      {/* HEADER */}

      <header className="header">
        <div className="brand">
          <div className="brand-icon">
            ✦
          </div>

          <div>
            <h1>FaceSkin AI</h1>

            <span>
              AI Face & Skin Analysis
            </span>
          </div>
        </div>

        <div className="model-status">
          <span
            className={
              modelReady
                ? "status-dot ready"
                : "status-dot"
            }
          />

          {modelReady
            ? "AI Model Ready"
            : "AI Model"}
        </div>
      </header>

      <main className="main">
        {/* INTRO */}

        {!capturedImage && (
          <section className="hero">
            <div className="hero-text">
              <span className="eyebrow">
                SMART SKIN ANALYSIS
              </span>

              <h2>
                Understand your skin
                <br />
                with <span>AI.</span>
              </h2>

              <p>
                Capture your face and receive
                an interactive skin analysis
                across acne, pigmentation,
                texture, redness, pores and
                more.
              </p>
            </div>

            <div className="analysis-card">
              <div className="card-title">
                <span>
                  {cameraStarted
                    ? "Live Face Scan"
                    : "Ready to Scan"}
                </span>

                <span
                  className={
                    faceDetected
                      ? "face-status detected"
                      : "face-status"
                  }
                >
                  {faceDetected
                    ? "Face Detected"
                    : "No Face"}
                </span>
              </div>

              <div className="camera-wrapper">
                <video
                  ref={videoRef}
                  className="camera"
                  muted
                  playsInline
                />

                {!cameraStarted && (
                  <div className="camera-placeholder">
                    <div className="face-placeholder">
                      🙂
                    </div>

                    <p>
                      Start the camera
                      <br />
                      to begin your analysis
                    </p>
                  </div>
                )}

                {cameraStarted && (
                  <div className="face-guide">
                    <div className="corner top-left" />
                    <div className="corner top-right" />
                    <div className="corner bottom-left" />
                    <div className="corner bottom-right" />
                  </div>
                )}

                {cameraStarted &&
                  faceDetected && (
                    <div className="scan-line" />
                  )}
              </div>

              <div className="camera-actions">
                {!cameraStarted ? (
                  <button
                    className="primary-button"
                    onClick={startCamera}
                    disabled={loading}
                  >
                    {loading
                      ? "Loading AI..."
                      : "Start Skin Analysis"}
                  </button>
                ) : (
                  <button
                    className="capture-button"
                    onClick={captureFace}
                  >
                    <span className="capture-circle" />
                    Capture Face
                  </button>
                )}
              </div>

              {cameraStarted && (
                <p className="camera-tip">
                  Keep your face straight and
                  make sure your face is well lit.
                </p>
              )}
            </div>
          </section>
        )}

        {/* CAPTURED IMAGE */}

        {capturedImage && (
          <section className="results-section">
            <div className="results-header">
              <div>
                <span className="eyebrow">
                  AI ANALYSIS COMPLETE
                </span>

                <h2>
                  Your Skin Analysis
                </h2>

                <p>
                  Results are generated from
                  the captured facial image.
                </p>
              </div>

              <button
                className="secondary-button"
                onClick={retake}
              >
                ↻ Retake
              </button>
            </div>

            {/* SCORE */}

            <div className="score-layout">
              <div className="face-result-card">
                <div className="result-image-wrapper">
                  <img
                    src={capturedImage}
                    alt="Captured face"
                    className="result-image"
                  />

                  <canvas
                    ref={canvasRef}
                    className="landmark-canvas"
                  />
                </div>

                <div className="skin-profile">
                  <span>
                    Estimated Skin Type
                  </span>

                  <strong>
                    {analysis?.skinType}
                  </strong>
                </div>
              </div>

              <div className="overall-card">
                <span className="card-label">
                  OVERALL SKIN SCORE
                </span>

                <div className="score-circle">
                  <div>
                    <strong>
                      {analysis?.overallScore}
                    </strong>

                    <span>/100</span>
                  </div>
                </div>

                <h3>
                  {analysis?.overallScore >= 80
                    ? "Healthy Appearance"
                    : analysis?.overallScore >= 60
                    ? "Good Foundation"
                    : "Needs Attention"}
                </h3>

                <p>
                  This score summarizes the
                  prototype's image-based
                  skin indicators.
                </p>
              </div>
            </div>

            {/* PARAMETERS */}

            <section className="parameters-section">
              <div className="section-heading">
                <div>
                  <span className="eyebrow">
                    DETAILED ANALYSIS
                  </span>

                  <h3>
                    Skin Parameters
                  </h3>
                </div>
              </div>

              <div className="parameter-grid">
                {PARAMETERS.map(
                  (parameter) => {
                    const score =
                      analysis?.[
                        parameter.key
                      ] ?? 0;

                    return (
                      <div
                        className="parameter-card"
                        key={parameter.key}
                      >
                        <div className="parameter-top">
                          <div className="parameter-name">
                            <span className="parameter-icon">
                              {parameter.icon}
                            </span>

                            <div>
                              <strong>
                                {parameter.label}
                              </strong>

                              <small>
                                {getSeverity(
                                  score
                                )}
                              </small>
                            </div>
                          </div>

                          <strong className="parameter-score">
                            {renderScore(
                              score
                            )}
                          </strong>
                        </div>

                        <div className="progress-track">
                          <div
                            className="progress-bar"
                            style={{
                              width: `${score}%`,
                            }}
                          />
                        </div>
                      </div>
                    );
                  }
                )}
              </div>
            </section>

            {/* RECOMMENDATIONS */}

            <section className="recommendation-section">
              <div className="recommendation-card">
                <div className="recommendation-heading">
                  <span className="recommendation-icon">
                    ✨
                  </span>

                  <div>
                    <span className="eyebrow">
                      PERSONALIZED INSIGHTS
                    </span>

                    <h3>
                      Suggested Skin Routine
                    </h3>
                  </div>
                </div>

                <div className="recommendations">
                  {recommendations.map(
                    (recommendation, index) => (
                      <div
                        className="recommendation-item"
                        key={index}
                      >
                        <span>✓</span>

                        <p>
                          {recommendation}
                        </p>
                      </div>
                    )
                  )}
                </div>
              </div>
            </section>

            {/* DISCLAIMER */}

            <div className="disclaimer">
              <strong>
                Prototype information
              </strong>

              <p>
                This application demonstrates
                image-based skin analysis and
                should not be considered a
                medical diagnosis. For production
                use, parameter-specific models
                should be trained and clinically
                validated where appropriate.
              </p>
            </div>
          </section>
        )}

        {error && (
          <div className="error-message">
            <span>⚠</span>
            {error}
          </div>
        )}
      </main>

      <footer className="footer">
        <span>
          FaceSkin AI
        </span>

        <span>
          Powered by React + MediaPipe
        </span>
      </footer>
    </div>
  );
}

export default App;