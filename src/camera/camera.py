import numpy as np
from typing import Optional

class CameraFeed:
  """Manages video capture from camera or file source."""

  def __init__(self, source: int = 0):
    pass

  def start(self) -> None:
    """Initialize and open the video capture."""
    pass

  def read(self) -> Optional[np.ndarray]:
    """Read next frame from the camera."""
      pass

  def stop(self) -> None:
    """Release the video capture."""
    pass

  def __enter__(self):
    self.start()
    return self

  def __exit__(self, *args):
    self.stop()
    pass
