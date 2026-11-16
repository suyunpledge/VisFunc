from hashlib import sha256

import pytest

from svsg.recognition.frames import frame_reference


def test_motionspec_bridge_verifies_bytes_and_preserves_nonuniform_time():
    data = b"an already-read frame file"
    digest = sha256(data).hexdigest()
    row = {
        "frame": 17,
        "t_ms": 413.725,
        "image": f"frame-00000017-{digest}.png",
        "sha256": digest,
        "roles": ["start", "end"],
    }
    reference = frame_reference("video-a", row, data)
    assert reference.timestamp_ms == 413.725
    assert reference.frame_index == 17
    with pytest.raises(ValueError, match="hash"):
        frame_reference("video-a", row, b"another frame")
    with pytest.raises(ValueError, match="filename"):
        frame_reference("video-a", {**row, "image": "../a.png"}, data)
    with pytest.raises(ValueError):
        frame_reference("video-a", {**row, "t_ms": float("inf")}, data)
