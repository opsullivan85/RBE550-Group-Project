import heapq
import math

import numpy as np


class PriorityQueue:
    """A simple priority queue.

    Non thread safe, unlike queue.PriorityQueue
    """

    def __init__(self):
        """Creates a new PriorityQueue"""
        self._elements = []

    def is_empty(self) -> bool:
        """Checks if the queue is empty

        Returns:
            bool: If the queue is empty
        """
        return len(self._elements) == 0

    def put(self, item: any, priority: float):
        """Insert an item with its priority (smaller priorities pop first)

        Args:
            item (any): Item to store
            priority (float): Priority to insert with
        """
        heapq.heappush(self._elements, (priority, item))

    def get(self) -> any:
        """
        Returns:
            any: The item with the smallest priority
        """
        return heapq.heappop(self._elements)[1]


def transform_2d(
    verts: list[list[float, float]], x: float, y: float, theta: float
) -> np.ndarray:
    """Transforms a list of verticies

    Args:
        verts (list[list[float, float]]): verticies to transform. verts must have a trailing 1
        x (float): x shift
        y (float): y shift
        theta (float): rotation

    Returns:
        np.ndarray: transformed verticies
    """
    # Define vertices as a 2D array
    verts = np.array(verts).T  # Transpose to make them column vectors

    # Precompute transformation matrix
    transform = np.array(
        [
            [math.cos(theta), -math.sin(theta), x],
            [math.sin(theta), math.cos(theta), y],
            [0, 0, 1],
        ]
    )

    # Apply the transformation in one go
    transformed_verts = (transform @ verts)[:2].T  # Apply and drop last row

    return transformed_verts


def drop_invalid_coordinates(coordinates: np.ndarray, array: np.ndarray) -> np.ndarray:
    """drops coordinates out of bounds of the array

    coordinates might look like:
        [[46 46 46 46 47 47 47 47 48 48 48 48 49 49 49 49]
        [46 47 48 49 46 47 48 49 46 47 48 49 46 47 48 49]]

    Args:
        coordinates (np.ndarray): the coorinates to examine. (2, n)
        array (np.ndarray): the array (x, y) bounding the coorinates

    Returns:
        np.ndarray: coordinates without values that cause bounds issues on array.
    """
    xmax_check = coordinates[0] < array.shape[0]
    ymax_check = coordinates[1] < array.shape[1]
    xmin_check = coordinates[0] >= 0
    ymin_check = coordinates[1] >= 0
    valid_coordinate_mask = np.logical_and(
        np.logical_and(xmax_check, ymax_check),
        np.logical_and(xmin_check, ymin_check),
    )
    coordinates = coordinates[:, valid_coordinate_mask]
    return coordinates


def unit_vectors_2d(n: int, theta: float = 0) -> np.ndarray:
    """Generate equally spaced unit vectors, clocked by a theta value

    Args:
        n (int): number of vectors to create
        theta (float, optional): Clocking value. Defaults to 0.

    Returns:
        np.ndarray: Unit vectors. shape=(n,2)
    """
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    unit_vectors = np.column_stack((np.cos(angles), np.sin(angles)))

    # Create rotation matrix for angle `theta`
    rotation_matrix = np.array(
        ((np.cos(theta), -np.sin(theta)), (np.sin(theta), np.cos(theta)))
    )

    # Rotate each unit vector
    rotated_vectors = unit_vectors @ rotation_matrix.T
    return rotated_vectors

def angular_difference(theta1, theta2):
    """Calculate the smallest angular difference between two angles in radians."""
    diff = (theta1 - theta2) % (2 * math.pi)
    return min(diff, 2 * math.pi - diff)