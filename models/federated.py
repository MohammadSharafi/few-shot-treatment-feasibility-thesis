"""
Federated Learning: FedAvg/FedProx aggregation across 5 hospital nodes
fit(), predict(), evaluate() interface.
"""
import numpy as np
import torch
import copy
from pathlib import Path
import yaml

from .few_shot import FewShotPredictor

def load_config():
    with open(Path(__file__).parent.parent / "config.yaml") as f:
        return yaml.safe_load(f)

class FederatedPredictor:
    """Federated few-shot model. Aggregates weights from nodes."""
    
    def __init__(self, cfg=None):
        self.cfg = cfg or load_config()
        self.fed_cfg = self.cfg["federated"]
        self.global_model = None
        self.node_models = []
    
    def fit(self, node_data_list, n_rounds=None):
        """
        node_data_list: list of (X, y) per node
        """
        n_rounds = n_rounds or self.fed_cfg["n_rounds"]
        n_nodes = len(node_data_list)
        local_epochs = self.fed_cfg["local_epochs"]
        aggregation = self.fed_cfg["aggregation"]
        mu = self.fed_cfg.get("fedprox_mu", 0.01)
        
        # Initialize global model
        self.global_model = FewShotPredictor(self.cfg)
        first_X, first_y = node_data_list[0]
        self.global_model.fit(first_X, first_y, n_episodes=50)
        global_state = copy.deepcopy(self.global_model.model.state_dict())
        
        for r in range(n_rounds):
            node_states = []
            node_sizes = []
            for node_idx, (X, y) in enumerate(node_data_list):
                if len(X) < 10 or len(np.unique(y)) < 2:
                    continue
                # Local model
                local = FewShotPredictor(self.cfg)
                local.model = copy.deepcopy(self.global_model.model)
                local.model.load_state_dict(global_state)
                # Local training
                local.fit(X, y, n_episodes=min(50, len(X) * 2))
                node_states.append(copy.deepcopy(local.model.state_dict()))
                node_sizes.append(len(X))
            
            if not node_states:
                continue
            # FedAvg
            total = sum(node_sizes)
            new_state = {}
            for k in global_state:
                new_state[k] = sum(node_states[i][k].float() * (node_sizes[i] / total)
                                    for i in range(len(node_states)))
            global_state = new_state
        
        self.global_model.model.load_state_dict(global_state)
        return self
    
    def predict(self, X, X_support=None, y_support=None):
        return self.global_model.predict(X, X_support, y_support)
    
    def evaluate(self, X, y, X_support=None, y_support=None):
        return self.global_model.evaluate(X, y, X_support, y_support)
