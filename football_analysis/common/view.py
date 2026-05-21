from typing import Tuple, Optional
from collections import deque
import cv2
import numpy as np
import numpy.typing as npt


class ViewTransformer:
    def __init__(
        self,
        source: Optional[npt.NDArray[np.float32]] = None,
        target: Optional[npt.NDArray[np.float32]] = None,
        matrix: Optional[npt.NDArray[np.float32]] = None
    ) -> None:
        if matrix is not None:
            matrix = np.asarray(matrix, dtype=np.float32)
            if matrix.shape != (3, 3):
                raise ValueError("Homography matrix must be 3x3.")
            self.m = self._normalize_matrix(matrix)
            return

        if source is None or target is None:
            raise ValueError("Source and target must be provided when matrix is not given.")
        if source.shape != target.shape:
            raise ValueError("Source and target must have the same shape.")
        if source.shape[1] != 2:
            raise ValueError("Source and target points must be 2D coordinates.")

        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m, _ = cv2.findHomography(source, target, cv2.RANSAC, 5.0)
        if self.m is None:
            raise ValueError("Homography matrix could not be calculated.")
        self.m = self._normalize_matrix(self.m)

    @staticmethod
    def _normalize_matrix(matrix: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        matrix = np.asarray(matrix, dtype=np.float32)
        if matrix.shape != (3, 3):
            raise ValueError("Homography matrix must be 3x3.")
        if np.isclose(matrix[2, 2], 0.0):
            raise ValueError("Homography matrix cannot be normalized.")
        return (matrix / matrix[2, 2]).astype(np.float32)

    @classmethod
    def from_matrix(cls, matrix: npt.NDArray[np.float32]) -> "ViewTransformer":
        return cls(matrix=matrix)

    def transform_points(
        self,
        points: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:
        if points.size == 0:
            return points.astype(np.float32)

        if points.shape[1] != 2:
            raise ValueError("Points must be 2D coordinates.")

        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2).astype(np.float32)

    def transform_image(
        self,
        image: npt.NDArray[np.uint8],
        resolution_wh: Tuple[int, int]
    ) -> npt.NDArray[np.uint8]:
        if len(image.shape) not in {2, 3}:
            raise ValueError("Image must be either grayscale or color.")
        return cv2.warpPerspective(image, self.m, resolution_wh)
