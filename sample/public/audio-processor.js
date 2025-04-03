// audio-processor.js
class AudioProcessor extends AudioWorkletProcessor {
    process(inputs, outputs) {
      const input = inputs[0];
      if (!input || input.length === 0) return true; // No input, exit early
  
      // Send audio data to the main thread
      this.port.postMessage({
        audioData: input[0] // Send the first channel of audio data
      });
  
      return true; // Keep the processor alive
    }
  }
  
  registerProcessor('audio-processor', AudioProcessor);