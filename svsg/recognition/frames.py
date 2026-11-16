"""Bind a MotionSpec evidence-frame row to the actual bytes before recognition."""

from __future__ import annotations

from hashlib import sha256

from pydantic import Field, StrictInt

from svsg.contracts.recognition import FrameReference, StrictModel


class EvidenceFrame(StrictModel):
    frame: StrictInt = Field(ge=0)
    t_ms: float = Field(ge=0)
    image: str
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    roles: list[str]


def frame_reference(sequence_id: str, row: dict, image: bytes) -> FrameReference:
    reference = EvidenceFrame.model_validate(row)
    if reference.image != f"frame-{reference.frame:08d}-{reference.sha256}.png":
        raise ValueError("invalid MotionSpec evidence filename")
    if sha256(image).hexdigest() != reference.sha256:
        raise ValueError("frame bytes do not match MotionSpec evidence hash")
    return FrameReference(
        sequence_id=sequence_id, frame_index=reference.frame, timestamp_ms=reference.t_ms
    )
