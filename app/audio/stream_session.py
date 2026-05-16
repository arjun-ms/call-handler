import collections
import logging
import numpy as np

from app.audio.vad import VadTrigger
from app.inference.diarizer import Diarizer
from app.inference.pipeline import predict

logger = logging.getLogger(__name__)

class StreamSession:
    """
    Manages the state for a single ongoing call.
    Receives continuous audio chunks, chunks them into bursts using VAD,
    diarizes the bursts, and performs voice attribute inference.
    
    Applies the 'Greeting Heuristic' where the first detected speaker is
    assumed to be the Agent, and all subsequent speakers are Customers.
    Maintains a rolling window consensus for predictions to reduce noise.
    """
    def __init__(self, sample_rate: int = 16000, consensus_window: int = 5):
        self.sample_rate = sample_rate
        self.consensus_window = consensus_window
        
        self.vad = VadTrigger(sample_rate=sample_rate)
        self.diarizer = Diarizer()
        
        # State tracking
        self.speaker_role_map = {}  # speaker_id -> "Agent" | "Customer"
        
        # History for rolling consensus
        self.speaker_history = collections.defaultdict(lambda: {
            "gender": collections.deque(maxlen=self.consensus_window),
            "age_bracket": collections.deque(maxlen=self.consensus_window)
        })

    def process_chunk(self, audio_chunk: np.ndarray) -> list[dict]:
        """
        Process a new chunk of audio. If a burst is detected, 
        it runs diarization and inference, returning the results.
        
        Returns a list of event dictionaries (empty if no burst triggered).
        """
        bursts = self.vad.process_audio(audio_chunk)
        events = []
        
        for burst in bursts:
            event = self._process_burst(burst)
            if event:
                events.append(event)
                
        return events

    def _process_burst(self, burst: np.ndarray) -> dict:
        """Run diarization and inference on a single burst."""
        try:
            burst_duration = len(burst) / self.sample_rate
            logger.info(f"Processing burst: {burst_duration:.2f}s ({len(burst)} samples)")
            
            # Safety: truncate to max 7 seconds to prevent OOM in wav2vec2
            max_samples = int(self.sample_rate * 7.0)
            if len(burst) > max_samples:
                logger.warning(f"Burst too long ({burst_duration:.1f}s), truncating to 7.0s")
                burst = burst[:max_samples]
            
            # 1. Diarization
            speaker_id = self.diarizer.process_burst(burst)
            
            # 2. Greeting Heuristic
            if speaker_id not in self.speaker_role_map:
                # First speaker is Agent, else Customer
                if not self.speaker_role_map:
                    self.speaker_role_map[speaker_id] = "Agent"
                    logger.info(f"Mapped {speaker_id} to Agent")
                else:
                    self.speaker_role_map[speaker_id] = "Customer"
                    logger.info(f"Mapped {speaker_id} to Customer")
                    
            role = self.speaker_role_map[speaker_id]
            
            # 3. Skip inference for the Agent (our own voice AI)
            if role == "Agent":
                logger.info(f"Skipping inference for Agent ({speaker_id})")
                return {
                    "speaker_id": speaker_id,
                    "role": role,
                    "status": "speaking"
                }
            
            # 4. Inference for Customer (requires 2D array [channels, samples])
            burst_2d = burst.reshape(1, -1)
            raw_preds = predict(burst_2d, self.sample_rate)
            
            # 4. Consensus Smoothing
            hist = self.speaker_history[speaker_id]
            hist["gender"].append(raw_preds["gender"]["prediction"])
            hist["age_bracket"].append(raw_preds["age_bracket"]["prediction"])
            
            consensus_gender = collections.Counter(hist["gender"]).most_common(1)[0][0]
            consensus_age = collections.Counter(hist["age_bracket"]).most_common(1)[0][0]
            
            return {
                "speaker_id": speaker_id,
                "role": role,
                "gender": consensus_gender,
                "age_bracket": consensus_age,
                # Include the raw confidence of the current burst
                "confidence_gender": raw_preds["gender"]["confidence"],
                "confidence_age": raw_preds["age_bracket"]["confidence"]
            }
            
        except Exception as e:
            logger.error(f"Error processing burst: {e}", exc_info=True)
            return None
