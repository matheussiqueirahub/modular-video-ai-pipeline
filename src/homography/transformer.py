
import logging
import cv2
import numpy as np

class PerspectiveTransformer:
    """
    Handles Homography transformations to map 2D video coordinates to a 2D top-down map/court.
    """
    def __init__(self, src_points: np.ndarray = None, dst_points: np.ndarray = None):
        self.logger = logging.getLogger(__name__)
        self.homography_matrix = None
        
        if src_points is not None and dst_points is not None:
            self.compute_homography(src_points, dst_points)

    def compute_homography(self, src_points: np.ndarray, dst_points: np.ndarray):
        """
        Computes the homography matrix from source (video) points to destination (map) points.
        Both arrays should be shape (4, 2) or (N, 2).
        """
        self.homography_matrix, status = cv2.findHomography(src_points, dst_points)
        if self.homography_matrix is None:
            self.logger.warning("Could not compute homography matrix.")
        else:
            self.logger.info("Homography matrix computed successfully.")

    def transform_point(self, point: tuple) -> tuple:
        """
        Transforms a single (x, y) point from video space to map space.
        """
        if self.homography_matrix is None:
            return point # Return original if no matrix (or logic to return None)

        # Convert to homogenous coordinates
        pt = np.array([point[0], point[1], 1.0])
        # Project
        dst_pt = self.homography_matrix @ pt
        # Normalize
        dst_pt = dst_pt / (dst_pt[2] + 1e-6)
        
        return (int(dst_pt[0]), int(dst_pt[1]))

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """
        Transforms an array of points (N, 2).
        """
        if self.homography_matrix is None:
            return points
            
        if points.ndim == 1:
            points = points.reshape(1, -1)
            
        transformed_points = cv2.perspectiveTransform(np.array([points], dtype=np.float32), self.homography_matrix)
        return transformed_points[0]
