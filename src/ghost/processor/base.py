import os.path as op
from typing import Literal, Optional, Sequence
from insightface.app import FaceAnalysis

FILE_DIR = op.dirname(__file__)


class BaseProcessor:
    def __init__(
        self,
        task: Literal["detection", "landmark", "attribute"],
        providers: Optional[Sequence[str]] = None,
    ) -> None:
        """Initializes the BaseProcessor class."""
        # Check if the task is valid
        if task not in {"detection", "landmark", "attribute"}:
            raise ValueError(
                f"task must be one of 'detection', 'landmark', 'attribute'. Got: {task}"
            )
        # Change the task to the appropriate format
        if task == "landmark":
            task = "landmark_2d_106"
        elif task == "attribute":
            task = "genderage"
        # Initialize the model for the task
        self.model = FaceAnalysis(
            allowed_modules=[task],
            providers=providers,
        )
        self.model.prepare(ctx_id=0, det_size=(640, 640))
