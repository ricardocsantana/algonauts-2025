#!/usr/bin/env python3
"""
Simple feature extraction script for Algonauts 2025 dataset.

This script extracts video, audio, and text features from the movie stimuli
and saves them to disk for later use in model training.

Environment Variables:
    DATAPATH: Path to the algonauts dataset
    SAVEPATH: Path to save extracted features cache

Usage:
    export DATAPATH=/path/to/dataset
    export SAVEPATH=/path/to/cache
    python extract_features.py
    
    Or with command line arguments:
    python extract_features.py --data_dir /path/to/dataset --cache_dir /path/to/cache
"""

import argparse
import logging
import os
from pathlib import Path

from data_utils.data import StudyLoader
from data_utils.features.audio import Wav2VecBert
from data_utils.features.text import LLAMA3p2
from data_utils.features.video import VJEPA2
from data_utils.helpers import prepare_features

os.environ.pop('MallocStackLogging', None)
os.environ.pop('MallocStackLoggingNoCompact', None)

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s %(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def extract_features(
    data_dir: str | Path,
    cache_dir: str | Path,
    modalities: list[str] = ["video", "audio", "text"],
    device: str = "auto",
    query: str | None = None,
):
    """
    Extract features from the Algonauts 2025 dataset.

    Parameters
    ----------
    data_dir : str or Path
        Path to the algonauts_2025.competitors directory containing stimuli and fmri folders
    cache_dir : str or Path
        Path to store extracted features cache
    modalities : list of str
        List of modalities to extract. Options: ['video', 'audio', 'text']
    device : str
        Device to use for feature extraction ('auto', 'cpu', 'cuda', or 'mps')
    query : str, optional
        Optional query to filter the dataset (e.g., 'subject=="sub-01"')
    """
    
    data_dir = Path(data_dir)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Cache directory: {cache_dir}")
    logger.info(f"Modalities to extract: {modalities}")
    
    # Initialize study loader
    logger.info("Initializing study loader...")
    study_loader = StudyLoader(
        path=data_dir,
        query=query,
        infra={
            "folder": str(cache_dir / "study"),
            "mode": "cached",
        },
        enhancers={
            "addtext": {"name": "AddText"},
            "addsentence": {
                "name": "AddSentenceToWords",
                "max_unmatched_ratio": 0.05,
            },
            "addcontext": {
                "name": "AddContextToWords",
                "sentence_only": False,
                "max_context_len": 1024,
            },
            "removemissing": {"name": "RemoveMissing"},
            "extractaudio": {"name": "ExtractAudioFromVideo"},
            "chunkevents": {
                "name": "ChunkEvents",
                "event_type_to_chunk": "Sound",
                "max_duration": 60,
                "min_duration": 30,
            },
        },
    )
    
    # Build events dataframe
    logger.info("Building events dataframe...")
    events = study_loader.build()
    logger.info(f"Total events: {len(events)}")
    logger.info(f"Event types: {events.type.unique()}")
    logger.info(f"Subjects: {events.subject.unique()}")
    
    # Initialize feature extractors based on selected modalities
    features = {}
    
    feature_config = {
        "layers": [0.5, 0.75, 1.0],
        "layer_aggregation": "group_mean",
        "device": device,
        "infra": {
            "folder": str(cache_dir / "features"),
            "keep_in_ram": False,
            "mode": "cached",
        },
    }
    
    if "video" in modalities:
        logger.info("Initializing video feature extractor (VJEPA2)...")
        features["video"] = VJEPA2(**feature_config)
    
    if "audio" in modalities:
        logger.info("Initializing audio feature extractor (Wav2VecBert)...")
        features["audio"] = Wav2VecBert(**feature_config)
    
    if "text" in modalities:
        logger.info("Initializing text feature extractor (LLAMA3p2)...")
        features["text"] = LLAMA3p2(**feature_config)
    
    # Extract features
    logger.info("Starting feature extraction...")
    logger.info("This may take a while depending on the dataset size...")
    
    prepare_features(features, events)
    
    logger.info("Feature extraction complete!")
    logger.info(f"Features saved to: {cache_dir / 'features'}")
    
    # Print summary
    logger.info("\n=== Extraction Summary ===")
    for modality, feature in features.items():
        logger.info(f"{modality.capitalize()} features: Extracted")


def main():
    parser = argparse.ArgumentParser(
        description="Extract features from Algonauts 2025 dataset"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.getenv('DATAPATH', None),
        help="Path to the data directory (or set DATAPATH env var)",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=os.getenv('SAVEPATH', "./feature_cache"),
        help="Path to store extracted features cache (or set SAVEPATH env var, default: ./feature_cache)",
    )
    parser.add_argument(
        "--modalities",
        type=str,
        nargs="+",
        default=["video", "audio", "text"],
        choices=["video", "audio", "text"],
        help="Modalities to extract (default: all)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device to use for feature extraction (default: auto)",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help='Optional query to filter dataset (e.g., "subject==\'sub-01\'")',
    )
    
    args = parser.parse_args()
    
    if args.data_dir is None:
        parser.error("--data_dir is required (or set DATAPATH environment variable)")
    
    extract_features(
        data_dir=args.data_dir,
        cache_dir=args.cache_dir,
        modalities=args.modalities,
        device=args.device,
        query=args.query,
    )


if __name__ == "__main__":
    main()
