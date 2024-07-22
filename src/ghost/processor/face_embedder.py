import os.path as op
from typing import Optional, Sequence

import numpy as np
import torch
from insightface.model_zoo import ArcFaceONNX
from onnxruntime import InferenceSession
import torch.nn.functional as F

FILE_DIR = op.dirname(__file__)




class FaceEmbedder:
    def __init__(
        self,
        arcface_weight_path: Optional[str] = None,
        providers: Optional[Sequence[str]] = None,
    ) -> None:
        """Initializes the FaceProcessor class."""
        if arcface_weight_path is None:
            arcface_weight_path = op.join(FILE_DIR, "../../../weights/arcface.onnx")
        session = InferenceSession(arcface_weight_path, providers=providers)
        self.embedder = ArcFaceONNX(arcface_weight_path, session=session)

    def embed(self, face: np.ndarray) -> torch.Tensor:
        """Embeds the faces using the ArcFace model."""
        # Run face through ArcFace model
        embedding: torch.Tensor = self.embedder.get_feat(face)
        # Normalize the embedding
        embedding = F.normalize(embedding)
        return embedding