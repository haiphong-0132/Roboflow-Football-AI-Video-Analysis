from typing import Tuple
import cv2
import numpy as np
import numpy.typing as npt

class ViewTransformer:
    def __init__(self, source: npt.NDArray[np.float32], target: npt.NDArray[np.float32]) -> None:
        if source.shape != target.shape:
            raise ValueError('Source and target must has the same shape.')
        if source.shape[1] != 2:
            raise ValueError('Source and target must be 2D.')
        
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m, _ = cv2.findHomography(source, target)

        if self.m is None:
            raise ValueError('Cannot find homography matrix.')
        
    def transform_points(self, points: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        if points.size == 0:
            return points
        if points.shape[1] != 2:
            raise ValueError('Points must be 2D.')
        
        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        
        return transformed_points.reshape(-1, 2).astype(np.float32)
    
    def transform_image(self, image: npt.NDArray[np.uint8], resolution_wh: Tuple[int, int]) -> npt.NDArray[np.uint8]:
        if len(image.shape) not in {2, 3}:
            raise ValueError('Image must be either grayscale or color.')
        
        return cv2.warpPerspective(image, self.m, resolution_wh)