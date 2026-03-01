from typing import Dict, Any


class ExperimentConfig:
    def __init__(self, 
                 name: str, 
                 model_params: Dict[str, Any], 
                 train_params: Dict[str, Any], 
                 mi_params: Dict[str, Any]):
        """
        Configuration container for a single experiment.
        
        Args:
            name: Unique ID for the experiment.
            model_params: Dict passed to VariationalAutoEncoder (hidden_dims, etc.)
            train_params: Dict with 'lr', 'epochs'.
            mi_params: Dict with 'method' (kde/kraskov), 'sigma', 'n_neig'.
        """
        self.name = name
        self.model_params = model_params
        self.train_params = train_params
        self.mi_params = mi_params