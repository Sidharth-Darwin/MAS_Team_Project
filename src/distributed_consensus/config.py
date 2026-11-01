"""Lightweight, explicit configuration defaults (no global mutable state)."""
from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    seed: int=16
    eta: float=.95
    lr: float=.002
    horizon: int=80
    epochs: int=200
    batch: int=128
    eval_samples: int=24
    device: str='cpu'
