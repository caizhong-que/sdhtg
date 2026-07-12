from __future__ import annotations
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class CurriculumState:
    boundary_temperature: float
    film_strength: float
    boundary_loss_scale: float


class Curriculum:
    def __init__(self, config: dict): self.config=config
    @staticmethod
    def cosine(start: float, end: float, progress: float) -> float:
        progress=min(max(progress,0.0),1.0)
        return end+(start-end)*0.5*(1+math.cos(math.pi*progress))
    def at(self, epoch: int, maximum_epochs: int) -> CurriculumState:
        c=self.config; warm=int(c["warmup_epochs"])
        progress=max(epoch-warm,0)/max(maximum_epochs-warm-1,1)
        temperature=self.cosine(c["boundary_initial_temperature"],c["boundary_final_temperature"],progress)
        film_progress=min((epoch+1)/max(warm,1),1.0)
        film=c["film_initial_strength"]+(c["film_final_strength"]-c["film_initial_strength"])*film_progress
        boundary_scale=min((epoch+1)/max(int(c["boundary_loss_warmup_epochs"]),1),1.0)
        return CurriculumState(temperature,film,boundary_scale)
