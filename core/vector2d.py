"""High-precision 2D Vector implementation for planar aerospace mechanics.

All operations enforce 64-bit floating point precision (np.float64 / IEEE 754 double).
SI Units:
    - Coordinates / Distances: meters (m)
    - Velocities: meters per second (m/s)
    - Accelerations / Forces: m/s^2, Newtons (N)
    - Angles: radians (rad)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence, Union

import numpy as np

Scalar = Union[int, float, np.floating, np.integer]


@dataclass(frozen=True, slots=True)
class Vector2D:
    """Immutable 2D Vector with 64-bit floating point precision.

    Parameters
    ----------
    x : Scalar
        Component along the primary axis (e.g., downrange position in meters).
    y : Scalar
        Component along the secondary axis (e.g., altitude in meters).
    """

    x: float = 0.0
    y: float = 0.0

    def __post_init__(self) -> None:
        """Enforce 64-bit floating point precision."""
        object.__setattr__(self, "x", float(np.float64(self.x)))
        object.__setattr__(self, "y", float(np.float64(self.y)))

    # --- Arithmetic Operators ---

    def __add__(self, other: Vector2D) -> Vector2D:
        """Vector addition: self + other."""
        if not isinstance(other, Vector2D):
            return NotImplemented
        return Vector2D(self.x + other.x, self.y + other.y)

    def __sub__(self, other: Vector2D) -> Vector2D:
        """Vector subtraction: self - other."""
        if not isinstance(other, Vector2D):
            return NotImplemented
        return Vector2D(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: Scalar) -> Vector2D:
        """Scalar multiplication: self * scalar."""
        if isinstance(scalar, (int, float, np.floating, np.integer)):
            s = float(np.float64(scalar))
            return Vector2D(self.x * s, self.y * s)
        return NotImplemented

    def __rmul__(self, scalar: Scalar) -> Vector2D:
        """Scalar multiplication: scalar * self."""
        return self.__mul__(scalar)

    def __truediv__(self, scalar: Scalar) -> Vector2D:
        """Scalar division: self / scalar."""
        if isinstance(scalar, (int, float, np.floating, np.integer)):
            s = float(np.float64(scalar))
            if s == 0.0:
                raise ZeroDivisionError("Cannot divide Vector2D by zero.")
            return Vector2D(self.x / s, self.y / s)
        return NotImplemented

    def __neg__(self) -> Vector2D:
        """Vector negation: -self."""
        return Vector2D(-self.x, -self.y)

    def __pos__(self) -> Vector2D:
        """Vector unary positive: +self."""
        return self

    # --- Geometric & Linear Algebra Operations ---

    def dot(self, other: Vector2D) -> float:
        """Compute the Euclidean dot product: self . other.

        Parameters
        ----------
        other : Vector2D
            Vector to dot with.

        Returns
        -------
        float
            Dot product scalar (64-bit float).
        """
        if not isinstance(other, Vector2D):
            raise TypeError(f"Dot product requires a Vector2D, got {type(other).__name__}")
        return float(np.float64(self.x * other.x + self.y * other.y))

    def __matmul__(self, other: Vector2D) -> float:
        """Matrix multiplication operator syntax for dot product: self @ other."""
        return self.dot(other)

    def cross(self, other: Vector2D) -> float:
        """Compute the 2D cross product (perpendicular dot product / determinant).

        Equivalent to the z-component of (x1, y1, 0) x (x2, y2, 0):
            self.x * other.y - self.y * other.x

        Parameters
        ----------
        other : Vector2D
            Vector to cross with.

        Returns
        -------
        float
            Signed scalar determinant (64-bit float).
            Positive if `other` is counter-clockwise relative to `self`.
        """
        if not isinstance(other, Vector2D):
            raise TypeError(f"Cross product requires a Vector2D, got {type(other).__name__}")
        return float(np.float64(self.x * other.y - self.y * other.x))

    def magnitude_squared(self) -> float:
        """Compute the squared Euclidean length: |self|^2."""
        return float(np.float64(self.x * self.x + self.y * self.y))

    def magnitude(self) -> float:
        """Compute the Euclidean length: |self|."""
        return float(np.float64(math.hypot(self.x, self.y)))

    def normalized(self) -> Vector2D:
        """Compute the unit vector pointing in the same direction.

        Returns
        -------
        Vector2D
            Unit vector with magnitude 1.0.

        Raises
        ------
        ZeroDivisionError
            If the vector has zero magnitude (within 1e-15 numerical tolerance).
        """
        mag = self.magnitude()
        if math.isclose(mag, 0.0, abs_tol=1e-15):
            raise ZeroDivisionError("Cannot normalize a zero-magnitude Vector2D.")
        return Vector2D(self.x / mag, self.y / mag)

    def angle(self) -> float:
        """Calculate the counter-clockwise angle relative to the +X axis.

        Returns
        -------
        float
            Attitude / direction angle in radians within [-pi, +pi].
        """
        return float(np.float64(math.atan2(self.y, self.x)))

    def rotate(self, angle_rad: float) -> Vector2D:
        """Rotate the vector counter-clockwise by a given angle.

        Parameters
        ----------
        angle_rad : float
            Rotation angle in radians (rad).

        Returns
        -------
        Vector2D
            Rotated vector.
        """
        angle_64 = float(np.float64(angle_rad))
        cos_theta = math.cos(angle_64)
        sin_theta = math.sin(angle_64)
        rot_x = self.x * cos_theta - self.y * sin_theta
        rot_y = self.x * sin_theta + self.y * cos_theta
        return Vector2D(rot_x, rot_y)

    def perpendicular(self) -> Vector2D:
        """Return a vector rotated 90 degrees counter-clockwise: (-y, x)."""
        return Vector2D(-self.y, self.x)

    def distance_to(self, other: Vector2D) -> float:
        """Compute Euclidean distance to another vector."""
        if not isinstance(other, Vector2D):
            raise TypeError(f"distance_to requires a Vector2D, got {type(other).__name__}")
        return (self - other).magnitude()

    def distance_squared_to(self, other: Vector2D) -> float:
        """Compute squared Euclidean distance to another vector."""
        if not isinstance(other, Vector2D):
            raise TypeError(f"distance_squared_to requires a Vector2D, got {type(other).__name__}")
        return (self - other).magnitude_squared()

    def is_close(self, other: Vector2D, rtol: float = 1e-9, atol: float = 1e-12) -> bool:
        """Determine whether two vectors are element-wise equal within a tolerance."""
        if not isinstance(other, Vector2D):
            return False
        return math.isclose(self.x, other.x, rel_tol=rtol, abs_tol=atol) and math.isclose(
            self.y, other.y, rel_tol=rtol, abs_tol=atol
        )

    # --- Conversions & Container Protocols ---

    def to_numpy(self) -> np.ndarray:
        """Convert vector to a 1D numpy array of shape (2,) with dtype np.float64."""
        return np.array([self.x, self.y], dtype=np.float64)

    @classmethod
    def from_numpy(cls, arr: np.ndarray) -> Vector2D:
        """Construct a Vector2D from a numpy array of shape (2,)."""
        arr_64 = np.asarray(arr, dtype=np.float64)
        if arr_64.shape != (2,):
            raise ValueError(f"Expected numpy array of shape (2,), got {arr_64.shape}")
        return cls(float(arr_64[0]), float(arr_64[1]))

    def to_tuple(self) -> tuple[float, float]:
        """Convert vector to a tuple (x, y)."""
        return (self.x, self.y)

    @classmethod
    def from_tuple(cls, tup: Sequence[Scalar]) -> Vector2D:
        """Construct a Vector2D from a 2-element sequence."""
        if len(tup) != 2:
            raise ValueError(f"Expected sequence of length 2, got {len(tup)}")
        return cls(float(tup[0]), float(tup[1]))

    def __iter__(self):
        """Allow sequence unpacking: x, y = vec."""
        yield self.x
        yield self.y

    def __getitem__(self, index: int) -> float:
        """Enable indexed access vec[0] -> x, vec[1] -> y."""
        if index in (0, -2):
            return self.x
        if index in (1, -1):
            return self.y
        raise IndexError(f"Vector2D index out of range: {index}")

    def __len__(self) -> int:
        """Length of the 2D vector is always 2."""
        return 2

    def __repr__(self) -> str:
        return f"Vector2D(x={self.x:.12g}, y={self.y:.12g})"