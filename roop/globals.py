from settings import Settings
from typing import List, Optional, Any

# --- Paths ---
source_path: Optional[str] = None
target_path: Optional[str] = None
output_path: Optional[str] = None
target_folder_path: Optional[str] = None

# --- Hardware & Execution ---
cuda_device_id: int = 0
execution_providers: List[str] = []
execution_threads: Optional[int] = None
max_memory: Optional[int] = None
headless: Optional[bool] = None
log_level: str = 'error'
fp16_enabled: bool = False

# --- Processing Configuration ---
# Face Swapping
face_swap_mode: Optional[str] = None
source_face_index: int = 0
target_face_index: int = 0
many_faces: Optional[bool] = None
distance_threshold: float = 0.65
blend_ratio: float = 0.5
selected_enhancer: Optional[str] = None
no_face_action: int = 0

# Extraction / Frame Ops
frame_processors: List[str] = []
keep_fps: Optional[bool] = None
keep_frames: Optional[bool] = None
skip_audio: Optional[bool] = None
wait_after_extraction: Optional[bool] = None
use_batch: Optional[bool] = None
video_encoder: Optional[str] = None
video_quality: Optional[int] = None
subsample_size: int = 128

# UI / Runtime Config
autorotate_faces: Optional[bool] = None
vr_mode: Optional[bool] = None
face_position: Optional[Any] = None 
default_det_size: bool = True
startup_args: Any = None

# --- Runtime State ---
processing: bool = False

# Face Analysis Objects
g_current_face_analysis: Any = None
g_desired_face_analysis: Any = None
FACE_ENHANCER: Any = None

# Inputs/Targets (Runtime Lists)
INPUT_FACESETS: List[Any] = []
TARGET_FACES: List[Any] = []

# Processor Chains
IMAGE_CHAIN_PROCESSOR: Any = None
VIDEO_CHAIN_PROCESSOR: Any = None
BATCH_IMAGE_CHAIN_PROCESSOR: Any = None

# Settings Object
CFG: Optional[Settings] = None
