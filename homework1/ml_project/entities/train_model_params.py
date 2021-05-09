from typing import Optional
from dataclasses import dataclass, field


@dataclass()
class TrainingParams:
    penalty: Optional[str]
    C: Optional[float]
    fit_intercept: Optional[bool]
    solver: Optional[str]
    max_iter: Optional[int]
    random_state: Optional[int]
    n_neighbors: Optional[int]
    algorithm: Optional[str]
    metric: Optional[str]
    model_type: str = field(default="LogisticRegression")

