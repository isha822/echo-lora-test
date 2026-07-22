from dataclasses import dataclass , field
from typing import List, Optional


@dataclass
class EchoLoraConfig:
    #LoRA parameters
    r: int = 16
    lora_alpha: float = 32
    lora_dropout: float = 0.0
    target_modules: Optional[List[str]] = None

    #echo specific parameters
    source_layers: List[int] = field(default_factory=lambda: [-8, -7, -6, -5])
    target_layers: List[int] = field(default_factory=lambda: [4, 5, 6, 7])
    bottleneck_dim: int = 64

    #Routing parameters
    p_start: float = 1.0
    p_end: float = 0.2

    #Distillation parameters
    lambda_kd: float = 1.0
    tau: float = 2.0