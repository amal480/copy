import { useRef, useEffect, useState, useCallback } from "react";
import './App.css';

function App() {
  const videoRef = useRef(null);
  const websocketRef = useRef(null); // For video
  const gazeWebsocketRef = useRef(null); // <<-- New: For gaze
  const audioWebSocketRef = useRef(null); // For audio
  const streamingIntervalRef = useRef(null);
  const canvasRef = useRef(null);
  const sessionIdRef = useRef(null);

  const [isStreaming, setIsStreaming] = useState(false);
  const [frameRate, setFrameRate] = useState(10); // Adjustable frame rate
  const [detectionData, setDetectionData] = useState({
    cell_phone_boxes: [],
    head_direction: "Unknown",
    people_count: 0,
  });
  const [gazeDirection, setGazeDirection] = useState("Unknown"); // <<-- New: Gaze direction state
  const [anomalies, setAnomalies] = useState([]);
  const [showAnomalies, setShowAnomalies] = useState(false);

  // Audio streaming state and refs
  const [isRecording, setIsRecording] = useState(false);
  const [speechDetected, setSpeechDetected] = useState(false);
  const audioContextRef = useRef(null);
  const processorRef = useRef(null);
  const streamRef = useRef(null);

  useEffect(() => {
    const startCamera = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { width: 1280, height: 720 },
          audio: false,
        });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      } catch (error) {
        console.error("Error accessing the camera:", error);
      }
    };

    startCamera();

    // Video WebSocket connection
    websocketRef.current = new WebSocket("ws://127.0.0.1:8000/video");

    websocketRef.current.onopen = () => console.log("Video WebSocket connected");
    websocketRef.current.onclose = (event) => console.warn("Video WebSocket closed", event);
    websocketRef.current.onerror = (error) => console.error("Video WebSocket error", error);

    websocketRef.current.onmessage = async (event) => {
      const data = JSON.parse(event.data);
      setDetectionData(data);
      
      // Store anomalies
      if (data.cell_phone_boxes.length > 0) {
        await storeAnomaly({
          type: 'cell_phone',
          count: data.cell_phone_boxes.length
        });
      }
      
      if (data.people_count > 1) {
        await storeAnomaly({
          type: 'multiple_people',
          count: data.people_count
        });
      }
      
      if (data.head_direction !== "Looking Straight") {
        await storeAnomaly({
          type: 'head_direction',
          direction: data.head_direction
        });
      }
    };

    // <<-- New: Gaze WebSocket connection
    gazeWebsocketRef.current = new WebSocket("ws://127.0.0.1:8000/gaze");

    gazeWebsocketRef.current.onopen = () => console.log("Gaze WebSocket connected");
    gazeWebsocketRef.current.onclose = (event) => console.warn("Gaze WebSocket closed", event);
    gazeWebsocketRef.current.onerror = (error) => console.error("Gaze WebSocket error", error);

    gazeWebsocketRef.current.onmessage = async (event) => {
      const data = JSON.parse(event.data);
      setGazeDirection(data.gaze_direction);
      
      // Store gaze direction anomalies
      if (data.gaze_direction !== "center") {
        await storeAnomaly({
          type: 'gaze_direction',
          direction: data.gaze_direction
        });
      }
    };

    // Audio WebSocket connection
    audioWebSocketRef.current = new WebSocket("ws://127.0.0.1:8001/audio");

    audioWebSocketRef.current.onopen = () => console.log("Audio WebSocket connected");
    audioWebSocketRef.current.onclose = (event) => console.warn("Audio WebSocket closed", event);
    audioWebSocketRef.current.onerror = (error) => console.error("Audio WebSocket error", error);

    audioWebSocketRef.current.onmessage = async (event) => {
      const response = JSON.parse(event.data);
      const hasSpeech = response.speech_timestamps.length > 0;
      setSpeechDetected(hasSpeech);
      
      // Store speech detection anomalies
      if (hasSpeech) {
        await storeAnomaly({
          type: 'speech'
        });
      }
    };

    return () => {
      if (websocketRef.current) websocketRef.current.close();
      if (gazeWebsocketRef.current) gazeWebsocketRef.current.close(); // <<-- New
      if (audioWebSocketRef.current) audioWebSocketRef.current.close();
      if (videoRef.current && videoRef.current.srcObject) {
        const tracks = videoRef.current.srcObject.getTracks();
        tracks.forEach((track) => track.stop());
      }
    };
  }, []);

  useEffect(() => {
    // Generate a unique session ID when component mounts
    sessionIdRef.current = `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }, []);

  const startStreaming = useCallback(() => {
    if (
      !videoRef.current ||
      !websocketRef.current ||
      websocketRef.current.readyState !== WebSocket.OPEN
    )
      return;

    if (isStreaming) {
      console.log("Streaming is already running");
      return;
    }

    setIsStreaming(true);
    streamingIntervalRef.current = setInterval(() => {
      const canvas = document.createElement("canvas");
      canvas.width = videoRef.current.videoWidth;
      canvas.height = videoRef.current.videoHeight;

      const context = canvas.getContext("2d");
      context.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);

      canvas.toBlob((blob) => {
        if (blob && websocketRef.current.readyState === WebSocket.OPEN) {
          websocketRef.current.send(blob);
        }
        // <<-- New: Send frame to gaze WebSocket as well
        if (blob && gazeWebsocketRef.current && gazeWebsocketRef.current.readyState === WebSocket.OPEN) {
          gazeWebsocketRef.current.send(blob);
        }
      }, "image/jpeg");
    }, 1000 / frameRate);
  }, [isStreaming, frameRate]);

  const stopStreaming = useCallback(() => {
    if (streamingIntervalRef.current) {
      clearInterval(streamingIntervalRef.current);
      streamingIntervalRef.current = null;
    }
    setIsStreaming(false);
    console.log("Streaming stopped");
  }, []);

  useEffect(() => {
    const drawDetections = () => {
      const canvas = canvasRef.current;
      const video = videoRef.current;

      if (canvas && video && detectionData) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const context = canvas.getContext("2d");
        context.clearRect(0, 0, canvas.width, canvas.height);

        // Draw video frame
        context.drawImage(video, 0, 0, canvas.width, canvas.height);

        // Draw bounding boxes for cell phones
        detectionData.cell_phone_boxes.forEach((box) => {
          context.strokeStyle = "#FF00FF"; // Pink color
          context.lineWidth = 3;
          context.strokeRect(box.x1, box.y1, box.x2 - box.x1, box.y2 - box.y1);
          context.font = "18px Arial";
          context.fillStyle = "#00FF00"; // Green color
          context.fillText("Cell Phone", box.x1, box.y1 - 10);
        });

        // Draw head direction
        context.font = "24px Arial";
        context.fillStyle = "#FFFFFF"; // White color
        context.fillText(
          `Head Direction: ${detectionData.head_direction}`,
          20,
          canvas.height - 50
        );

        // Draw people count
        context.fillText(
          `People Count: ${detectionData.people_count}`,
          20,
          canvas.height - 20
        );

        // <<-- New: Draw gaze direction
        context.fillText(
          `Gaze Direction: ${gazeDirection}`,
          20,
          canvas.height - 80
        );
      }
    };

    const interval = setInterval(drawDetections, 1000 / frameRate);
    return () => clearInterval(interval);
  }, [detectionData, frameRate, gazeDirection]);

  // Audio streaming functions
  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        audio: {
          channelCount: 1,
          sampleRate: 16000,
          sampleSize: 16,
        } 
      });

      streamRef.current = stream;
      
      // Create AudioContext for format conversion
      audioContextRef.current = new AudioContext({
        sampleRate: 16000,
      });
      
      const source = audioContextRef.current.createMediaStreamSource(stream);
      processorRef.current = audioContextRef.current.createScriptProcessor(4096, 1, 1);

      source.connect(processorRef.current);
      processorRef.current.connect(audioContextRef.current.destination);

      processorRef.current.onaudioprocess = (e) => {
        if (audioWebSocketRef.current && audioWebSocketRef.current.readyState === WebSocket.OPEN) {
          const inputData = e.inputBuffer.getChannelData(0);
          
          // Convert Float32Array to Int16Array
          const intData = new Int16Array(inputData.length);
          for (let i = 0; i < inputData.length; i++) {
            const s = Math.max(-1, Math.min(1, inputData[i]));
            intData[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
          }
          
          // Send audio data to the audio WebSocket server
          audioWebSocketRef.current.send(intData.buffer);
          console.log("Sending audio chunk, size:", intData.length);
        }
      };

      setIsRecording(true);
      console.log("Recording started");
    } catch (err) {
      console.error("Error accessing microphone:", err);
    }
  };

  const stopRecording = () => {
    console.log("Stopping recording...");
    
    if (processorRef.current) {
      processorRef.current.disconnect();
      processorRef.current = null;
    }

    if (audioContextRef.current) {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }

    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }

    setIsRecording(false);
    console.log("Recording stopped");
  };

  const storeAnomaly = async (anomaly) => {
    try {
      const response = await fetch('http://127.0.0.1:8000/store-anomaly', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          ...anomaly,
          session_id: sessionIdRef.current,
          timestamp: new Date().toISOString()
        }),
      });
      const data = await response.json();
      return data.success;
    } catch (error) {
      console.error('Error storing anomaly:', error);
      return false;
    }
  };

  const fetchAnomalies = async () => {
    try {
      const response = await fetch(`http://127.0.0.1:8000/get-anomalies/${sessionIdRef.current}`);
      const data = await response.json();
      setAnomalies(data.anomalies);
      setShowAnomalies(true);
    } catch (error) {
      console.error('Error fetching anomalies:', error);
    }
  };

  const handleStart = () => {
    startStreaming();
    startRecording();
  };

  const handleStop = async () => {
    stopStreaming();
    stopRecording();
    await fetchAnomalies();
  };

  return (
    <div className="app-container">
      <header className="app-header">
        <h1>Online Exam Proctoring System</h1>
      </header>
      <main className="app-main">
        <div className="video-container">
          <video
            ref={videoRef}
            autoPlay
            playsInline
            className="hidden-video"
          ></video>
          <canvas ref={canvasRef} className="video-canvas"></canvas>
        </div>
        <div className="controls">
          <button
            onClick={handleStart}
            disabled={isStreaming}
            className="control-button start-button"
          >
            Start Proctoring
          </button>
          <button
            onClick={handleStop}
            disabled={!isStreaming}
            className="control-button stop-button"
          >
            Stop Proctoring
          </button>
          <div className="frame-rate-control">
            <label htmlFor="frameRate">Frame Rate:</label>
            <input
              id="frameRate"
              type="number"
              value={frameRate}
              onChange={(e) => setFrameRate(Number(e.target.value))}
              min="1"
              max="60"
              className="frame-rate-input"
            />
          </div>
          {/* <button 
            onClick={startRecording} 
            disabled={isRecording}
            style={{ 
              padding: '10px 20px',
              margin: '5px',
              backgroundColor: isRecording ? '#ccc' : '#4CAF50'
            }}
          >
            Start Recording
          </button>
          <button 
            onClick={stopRecording} 
            disabled={!isRecording}
            style={{ 
              padding: '10px 20px',
              margin: '5px',
              backgroundColor: !isRecording ? '#ccc' : '#f44336'
            }}
          >
            Stop Recording
          </button> */}
        </div>
        <div className="detection-info">
          <h2>Proctoring Details</h2>
          <div className="info-item">
            <strong>Head Direction:</strong> {detectionData.head_direction}
          </div>
          <div className="info-item">
            <strong>Number of People Detected:</strong> {detectionData.people_count}
          </div>
          <div className="info-item">
            <strong>Detected Devices:</strong> {detectionData.cell_phone_boxes.length}
          </div>
          <div className="info-item">
            <strong>Speech Detected:</strong> {speechDetected ? '🗣️ Yes' : '❌ No'}
          </div>
          {/* <<-- New: Display gaze direction */}
          <div className="info-item">
            <strong>Gaze Direction:</strong> {gazeDirection}
          </div>
        </div>
        <div className="exam-status">
          <div className={`status-indicator ${isStreaming ? 'active' : 'inactive'}`}>
            {isStreaming ? "Proctoring Active" : "Proctoring Inactive"}
          </div>
        </div>
        
        {showAnomalies && (
          <div className="anomalies-container">
            <h2>Detected Anomalies</h2>
            {anomalies.length > 0 ? (
              <div className="anomalies-list">
                {anomalies.map((anomaly, index) => (
                  <div key={index} className="anomaly-item">
                    <div className="anomaly-timestamp">
                      {new Date(anomaly.timestamp).toLocaleString()}
                    </div>
                    <div className="anomaly-details">
                      {anomaly.type === 'cell_phone' && (
                        <div>📱 Cell Phone Detected</div>
                      )}
                      {anomaly.type === 'multiple_people' && (
                        <div>👥 Multiple People Detected: {anomaly.count}</div>
                      )}
                      {anomaly.type === 'head_direction' && (
                        <div>👤 Head Direction: {anomaly.direction}</div>
                      )}
                      {anomaly.type === 'gaze_direction' && (
                        <div>👁️ Gaze Direction: {anomaly.direction}</div>
                      )}
                      {anomaly.type === 'speech' && (
                        <div>🗣️ Speech Detected</div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="no-anomalies">No anomalies detected during this session.</div>
            )}
          </div>
        )}
      </main>
      <footer className="app-footer">
        <p>&copy; 2025 Your Company Name. All rights reserved.</p>
      </footer>
    </div>
  );
}

export default App;