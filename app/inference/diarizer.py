import logging
import numpy as np
import torch

logger = logging.getLogger(__name__)

class Diarizer:
    """
    Lightweight speaker clustering module using SpeechBrain ECAPA-TDNN embeddings.
    Assigns each burst a speaker_id based on cosine similarity to known speaker centroids.
    """
    def __init__(self, similarity_threshold: float = 0.5):
        self.similarity_threshold = similarity_threshold
        self.speaker_centroids = {}  # speaker_id -> numpy array embedding
        self.next_speaker_idx = 0
        
        try:
            from speechbrain.inference.speaker import EncoderClassifier
            # Load pre-trained ECAPA-TDNN model for speaker recognition
            self.classifier = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb", 
                run_opts={"device": "cpu"}
            )
            self._use_mock = False
            logger.info("Loaded SpeechBrain ECAPA-TDNN model successfully.")
        except Exception as e:
            logger.warning(f"Failed to load speechbrain model: {e}. Using mock clustering.")
            self.classifier = None
            self._use_mock = True

    def process_burst(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """
        Process a speech burst and return the assigned speaker ID.
        
        Args:
            audio: numpy array of float32 containing the speech burst.
            sample_rate: sample rate of the audio (default 16000).
            
        Returns:
            str: The speaker ID assigned to this burst (e.g., "speaker_0", "speaker_1").
        """
        if len(audio) == 0:
            return "unknown"
            
        if self._use_mock:
            # Simple mock: return speaker_0 for odd calls, speaker_1 for even calls 
            # to simulate two speakers if we don't have the model
            if not self.speaker_centroids:
                self.speaker_centroids["speaker_0"] = np.array([1.0])
                return "speaker_0"
            speaker_id = f"speaker_{len(self.speaker_centroids) % 2}"
            if speaker_id not in self.speaker_centroids:
                self.speaker_centroids[speaker_id] = np.array([1.0])
            return speaker_id
            
        # speechbrain expects shape (batch, time)
        signal = torch.from_numpy(audio).float().unsqueeze(0)
        
        # Get embeddings
        embeddings = self.classifier.encode_batch(signal)
        # Shape is (batch, 1, embedding_size), squeeze to 1D
        emb = embeddings.squeeze().detach().numpy()
        
        # In case the burst is too short, embedding might not be 1D 
        if emb.ndim == 0:
            return "unknown"
            
        # Normalize embedding for cosine similarity
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        
        if not self.speaker_centroids:
            speaker_id = f"speaker_{self.next_speaker_idx}"
            self.speaker_centroids[speaker_id] = emb
            self.next_speaker_idx += 1
            return speaker_id
            
        # Find best match
        best_speaker = None
        best_score = -1.0
        
        for spk, centroid in self.speaker_centroids.items():
            score = np.dot(emb, centroid)
            if score > best_score:
                best_score = score
                best_speaker = spk
                
        if best_score >= self.similarity_threshold:
            # Update centroid (moving average)
            # This allows the speaker's representation to drift slightly over time
            alpha = 0.1
            new_centroid = (1 - alpha) * self.speaker_centroids[best_speaker] + alpha * emb
            new_norm = np.linalg.norm(new_centroid)
            if new_norm > 0:
                self.speaker_centroids[best_speaker] = new_centroid / new_norm
            return best_speaker
        else:
            # New speaker
            speaker_id = f"speaker_{self.next_speaker_idx}"
            self.speaker_centroids[speaker_id] = emb
            self.next_speaker_idx += 1
            return speaker_id
