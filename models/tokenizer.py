"""
Symptom Tokenizer: Converts raw measurements to token sequences
Each token: [symptom_id, intensity, time_delta_norm, modality]
"""
import numpy as np
from pathlib import Path
import yaml

def load_config():
    with open(Path(__file__).parent.parent / "config.yaml") as f:
        return yaml.safe_load(f)

class SymptomTokenizer:
    """Converts patient symptom data to token sequences for ETHOS encoder."""
    
    def __init__(self, seq_length=25, mask_token_id=0, vocab_size=23):
        self.seq_length = seq_length
        self.mask_token_id = mask_token_id
        self.vocab_size = vocab_size
    
    def fit(self, X, y=None):
        """No fitting needed; tokenizer is deterministic."""
        return self
    
    def transform(self, X):
        """
        X: (n_samples, n_features) or (n_samples, seq_len, 4) if already tokenized
        Returns: (n_samples, seq_len, 4) token tensor
        """
        if X.ndim == 3 and X.shape[2] == 4:
            return X
        # X is (n, n_feat) - convert to tokens
        n = X.shape[0]
        tokens = np.zeros((n, self.seq_length, 4), dtype=np.float32)
        n_feat = min(X.shape[1], self.seq_length)
        for i in range(n):
            for j in range(n_feat):
                if j >= self.seq_length:
                    break
                val = X[i, j]
                sid = j + 1 if j < self.vocab_size - 1 else self.mask_token_id
                tokens[i, j, 0] = sid
                tokens[i, j, 1] = float(val) if not np.isnan(val) else 0.0
                tokens[i, j, 2] = 0.0
                tokens[i, j, 3] = 0.0 if j < 15 else 1.0
            for j in range(n_feat, self.seq_length):
                tokens[i, j, 0] = self.mask_token_id
                tokens[i, j, 1] = 0.0
                tokens[i, j, 2] = 0.0
                tokens[i, j, 3] = 0.0
        return tokens
    
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
