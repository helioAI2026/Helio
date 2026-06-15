import cv2


class Alert:
    COLORS = {'NORMAL': (60, 200, 60), 'AVISO': (0, 165, 255), 'ALERTA': (60, 60, 220), 'SEM_FACE': (100, 100, 100)}
    HUD_HEIGHT = 100

    def draw(self, img, result: dict, alert_duration: float = 0.0, overlay_threshold: float = 3.0):
        h, w = img.shape[:2]
        
        if alert_duration > overlay_threshold:
            overlay = img.copy()
            overlay[:] = (0, 0, 255)
            cv2.addWeighted(overlay, 0.2, img, 0.8, 0, img)
        
        cv2.rectangle(img, (0, 0), (w, self.HUD_HEIGHT), (20, 20, 20), -1)

        color = self.COLORS.get(result['status'], (100, 100, 100))
        cv2.putText(img, result['status'], (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        cv2.putText(img, f"{result['score']*100:.0f}%", (w - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        
        ear_text = f"EAR: {result['ear']:.2f}"
        perclos_text = f"PERCLOS: {result['perclos']*100:.0f}%"
        cv2.putText(img, ear_text, (12, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
        cv2.putText(img, perclos_text, (w - 180, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)

        bar_w = int((w - 20) * result['score'])
        cv2.rectangle(img, (10, self.HUD_HEIGHT - 15), (10 + bar_w, self.HUD_HEIGHT - 5), color, -1)
        cv2.putText(img, "Q: sair", (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1)