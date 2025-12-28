import numpy as np
import cv2
import onnxruntime
import roop.globals

from roop.typing import Frame
from roop.utilities import resolve_relative_path


class Mask_BisNet():
    plugin_options:dict = None
    model_bisenet = None
    processorname = 'mask_bisenet'
    type = 'mask'
    
    # BiSeNet Face Parsing Classes
    # 0: 'background'
    # 1: 'skin', 2: 'nose', 3: 'eye_g', 4: 'l_eye', 5: 'r_eye'
    # 6: 'l_brow', 7: 'r_brow', 8: 'l_ear', 9: 'r_ear', 10: 'mouth'
    # 11: 'u_lip', 12: 'l_lip', 13: 'hair', 14: 'hat', 15: 'ear_r'
    # 16: 'neck_l', 17: 'neck', 18: 'cloth'

    # Mapping keywords to classes
    # If a keyword is present, we remove it from the "face_parts" (so it becomes 1 in the mask -> kept original)
    
    # Default Face Parts (Things we want to SWAP - MASK 0)
    DEFAULT_FACE_PARTS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15] 

    KEYWORD_MAP = {
        'skin': [1],
        'nose': [2],
        'glasses': [3],
        'eyes': [4, 5],
        'brows': [6, 7],
        'ears': [8, 9, 15],
        'mouth': [10, 11, 12],
        'lips': [11, 12],
        'hair': [13],
        'hat': [14],
        'neck': [16, 17],
        'cloth': [18],
        'background': [0]
    }

    def Initialize(self, plugin_options:dict):
        if self.plugin_options is not None:
            if self.plugin_options["devicename"] != plugin_options["devicename"]:
                self.Release()

        self.plugin_options = plugin_options
        if self.model_bisenet is None:
            model_path = resolve_relative_path('../models/resnet18.onnx')
            onnxruntime.set_default_logger_severity(3)
            self.model_bisenet = onnxruntime.InferenceSession(model_path, None, providers=roop.globals.execution_providers)
            self.model_inputs = self.model_bisenet.get_inputs()
            self.model_outputs = self.model_bisenet.get_outputs()

            # replace Mac mps with cpu for the moment
            self.devicename = self.plugin_options["devicename"].replace('mps', 'cpu')

    def Run(self, img1, keywords:str) -> Frame:
        # Preprocessing similar to what BiSeNet expects
        # 512x512, standardized
        input_size = (512, 512)
        temp_frame = cv2.resize(img1, input_size, interpolation=cv2.INTER_LINEAR)
        
        # Normalize: (image - mean) / std
        # Mean: [0.485, 0.456, 0.406], Std: [0.229, 0.224, 0.225]
        temp_frame = temp_frame.astype(np.float32) / 255.0
        temp_frame = temp_frame - np.array([0.485, 0.456, 0.406], dtype=np.float32)
        temp_frame = temp_frame / np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        # NCHW format
        temp_frame = temp_frame.transpose(2, 0, 1)
        temp_frame = np.expand_dims(temp_frame, axis=0)

        io_binding = self.model_bisenet.io_binding()
        io_binding.bind_cpu_input(self.model_inputs[0].name, temp_frame)
        io_binding.bind_output(self.model_outputs[0].name, self.devicename)
        self.model_bisenet.run_with_iobinding(io_binding)
        ort_outs = io_binding.copy_outputs_to_cpu()
        
        parsing = ort_outs[0][0] # [19, 512, 512]
        parsing = parsing.argmax(0).astype(np.uint8) # [512, 512]

        # Determine which classes constitute the "Face" (to be swapped)
        # Default: Everything that belongs to the head/face area.
        # We start with a base set of parts.
        
        # Logic: 
        # Mask 0 = Modified (Swapped)
        # Mask 1 = Original (Kept)
        
        # If no keywords, we likely want to swap the "Standard Face" (Skin, Nose, Eyes, Mouth, Brows).
        # And keep Hair, Background, Clothes, Neck.
        
        # Standard Swap Set:
        swap_classes = {1, 2, 3, 4, 5, 6, 7, 10, 11, 12} 
        
        # If keywords provided, we assume the user wants to EXCLUDE these from swapping (Keep Original).
        # Or maybe INCLUDE them? 
        # "List of objects to mask and restore back on fake face" -> "Restore back" means KEEP ORIGINAL (Mask=1).
        
        if keywords:
            kw_list = [k.strip().lower() for k in keywords.split(',')]
            for kw in kw_list:
                if kw in self.KEYWORD_MAP:
                    # Remove these classes from swap_classes (so they become 'keep')
                    for cls in self.KEYWORD_MAP[kw]:
                        if cls in swap_classes:
                            swap_classes.remove(cls)
                        # Optionally, if user types "hair", and hair isn't in swap_classes default, nothing happens.
                        # But wait, if they types "hair" in Clip2Seg, it masks the hair (keeps it original).
                        # So existing behavior is "Keep Original".
                        # My swap_classes are "What to swap" (Mask 0).
                        # So removing from swap_classes = Make it Mask 1 (Keep Original).
                        
                        # Special Case: If user wants to FORCE swap something that isn't usually swapped?
                        # Probably rare. Usually they want to prevent swapping eyes or glasses.
        
        # Construct the mask
        # Initialize with 1s (Keep Original / Background)
        mask = np.ones_like(parsing, dtype=np.float32)
        
        # Set 0s where we want to SWAP
        for cls in swap_classes:
            mask[parsing == cls] = 0.0
            
        return mask

    def Release(self):
        del self.model_bisenet
        self.model_bisenet = None
