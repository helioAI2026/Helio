import cv2

class Alert:
    COLOR_OK    = (60, 200, 60)
    COLOR_ALERT = (60, 60, 220)

    def draw(self, img, drowsy, ear):
        h, w   = img.shape[:2]

        color  = self.COLOR_ALERT if drowsy else self.COLOR_OK

        label  = "SONOLENCIA" if drowsy else "NORMAL"

        cv2.rectangle(img, (0, 0), (w, 50), (20, 20, 20), -1)

        cv2.putText(img, label,          (12,      36), cv2.FONT_HERSHEY_SIMPLEX, 1.0,  color,        2, cv2.LINE_AA)

        cv2.putText(img, f"EAR {ear:.2f}", (w - 115, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1, cv2.LINE_AA)

        cv2.putText(img, "Q: sair",      (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1, cv2.LINE_AA)