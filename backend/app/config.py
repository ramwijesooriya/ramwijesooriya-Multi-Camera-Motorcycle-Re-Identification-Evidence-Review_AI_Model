"""Validated, exported per-job matching policy. Scores are not probabilities."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator


class MatchConfig(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    samples_per_track: int = Field(default=8, ge=2, le=16)
    top_k: int = Field(default=5, ge=1, le=50)
    ambiguity_margin: float = Field(default=.04, ge=0, le=1)
    visual_accept: float = Field(default=.90, ge=-1, le=1)
    visual_reject: float = Field(default=.55, ge=-1, le=1)
    min_crop_side: int = Field(default=24, ge=8)
    min_blur_variance: float = Field(default=20, ge=0)
    min_track_seconds: float = Field(default=.4, ge=0)
    min_detection_confidence: float = Field(default=.35, ge=0, le=1)
    min_within_track_similarity: float = Field(default=.75, ge=-1, le=1)
    max_tracking_gap_seconds: float = Field(default=1, gt=0)
    temporal_enabled: bool = True
    route: Literal['a_to_b', 'b_to_a'] = 'a_to_b'
    # Add this signed offset to B's video-relative timestamps to align with A.
    # Includes recording-start difference and camera-clock correction.
    camera_b_offset_seconds: float | None = None
    clock_uncertainty_seconds: float = Field(default=0, ge=0)
    boundary_uncertainty_seconds: float = Field(default=1, ge=0)
    min_travel_seconds: float = Field(default=0, ge=0)
    max_travel_seconds: float = Field(default=120, ge=0)
    camera_a_direction: Literal['unknown', 'left', 'right', 'up', 'down'] = 'unknown'
    camera_b_direction: Literal['unknown', 'left', 'right', 'up', 'down'] = 'unknown'

    @model_validator(mode='after')
    def ordered_bounds(self):
        if self.max_travel_seconds < self.min_travel_seconds:
            raise ValueError('Maximum travel time must be at least the minimum')
        if self.visual_reject >= self.visual_accept:
            raise ValueError('Visual rejection threshold must be below acceptance threshold')
        return self
