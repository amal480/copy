import io
import torch
import numpy as np
from fastapi import FastAPI, WebSocket
import logging

app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load Silero VAD model once at startup
logger.info("Loading Silero VAD model...")
vad_model, utils = torch.hub.load("snakers4/silero-vad", "silero_vad")
(get_speech_timestamps, _, _, _, _) = utils
logger.info("Silero VAD model loaded.")

@app.websocket("/audio")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("Client connected")
    
    chunk_count = 0
    try:
        while True:
            # Receive raw audio data
            audio_data = await websocket.receive_bytes()
            chunk_count += 1
            
            #logger.info(f"Received audio chunk {chunk_count}, size: {len(audio_data)} bytes")
            
            # Convert bytes to numpy array of int16
            audio_np = np.frombuffer(audio_data, dtype=np.int16)
            
            # Convert to float32 and normalize to [-1, 1]
            audio_float = audio_np.astype(np.float32) / 32768.0
            
            # Convert to PyTorch tensor and add batch dimension
            audio_tensor = torch.from_numpy(audio_float).unsqueeze(0)  # Shape: [1, num_samples]
            
            # Run Silero VAD (disable gradients for better performance)
            with torch.no_grad():
                speech_timestamps = get_speech_timestamps(audio_tensor, vad_model, sampling_rate=16000)
            
            # Determine if speech was detected
            speech_detected = len(speech_timestamps) > 0
            logger.info(len(speech_timestamps))
            
            # Log speech detection
            if speech_detected:
                logger.info(f"Speech detected in chunk {chunk_count}!")
            
            # Send results back to client
            #logger.info(speech_detected)
            await websocket.send_json({
                "speech_timestamps": speech_timestamps,
                "speech_detected": speech_detected,  # Add a boolean flag for easier frontend handling
            })

    except Exception as e:
        logger.error(f"Error in websocket connection: {e}")
        await websocket.close()
    finally:
        logger.info("Client disconnected")