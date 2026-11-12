import numpy as np

from src.eye_state.geometry import Face, crop_eye, eye_boxes


def face(right, left):
    return Face(box=np.array([0, 0, 10, 10], np.float32), right_eye=np.array(right, np.float32),
                left_eye=np.array(left, np.float32), score=0.9)


def test_boxes_square_centered():
    r, l = eye_boxes(face((100, 100), (160, 100)), (480, 640), scale=0.6, min_dist=12)
    assert r == (82, 82, 118, 118)
    assert l == (142, 82, 178, 118)


def test_box_out_of_frame_is_none():
    r, l = eye_boxes(face((5, 100), (65, 100)), (480, 640), scale=0.6, min_dist=12)
    assert r is None
    assert l is not None


def test_tiny_face_both_none():
    assert eye_boxes(face((100, 100), (108, 100)), (480, 640), scale=0.6, min_dist=12) == (None, None)


def test_crop_eye_shape_range():
    gray = (np.arange(480 * 640) % 256).astype(np.uint8).reshape(480, 640)
    c = crop_eye(gray, (82, 82, 118, 118))
    assert c.shape == (64, 64)
    assert c.dtype == np.float32
    assert 0.0 <= c.min() and c.max() <= 1.0
