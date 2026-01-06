
import torch
import numpy as np
from data_utils.features.subject import SubjectEncoder
from data_utils.events import Event

class MockEvent(Event):
    def __init__(self, start, duration, label):
        self.start = start
        self.duration = duration
        self.label = label
        self.extra = {}

    def to_dict(self):
        return {"label": self.label}

def test_subject_encoder_fix():
    # Create events
    events = [
        MockEvent(start=0.0, duration=1.0, label="sub-01"),
        MockEvent(start=1.0, duration=1.0, label="sub-02")
    ]
    
    encoder = SubjectEncoder()
    encoder.prepare(events)
    
    # Test normal case
    print("Testing normal case...")
    tensor = encoder(events[0], start=0.0, duration=1.0, trigger=events[0].to_dict())
    print(f"Normal tensor shape: {tensor.shape}")
    
    # Test missing case (empty events list)
    print("Testing missing case...")
    missing_tensor = encoder([], start=2.0, duration=1.0, trigger={})
    print(f"Missing tensor shape: {missing_tensor.shape}")
    
    if tensor.ndim == missing_tensor.ndim:
        print("SUCCESS: Dimensions match!")
    else:
        print(f"FAILURE: Dimensions mismatch! {tensor.ndim} vs {missing_tensor.ndim}")

if __name__ == "__main__":
    test_subject_encoder_fix()
