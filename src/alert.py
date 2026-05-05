import cv2


class Alert:
    # EAR — sonolência instantânea
    COLOR_EAR_OK    = (60, 200, 60)
    COLOR_EAR_ALERT = (60, 60, 220)

    # PERCLOS — tendência acumulada
    COLOR_PERCLOS_OK    = (160, 160, 160)
    COLOR_PERCLOS_ALERT = (0, 165, 255)

    # Bocejo
    COLOR_YAWN_OK    = (160, 160, 160)
    COLOR_YAWN_ALERT = (0, 200, 220)

    HUD_HEIGHT = 120  

    
    _Y1 = 28   # EAR
    _Y2 = 68   # PERCLOS
    _Y3 = 105  # Bocejo

    
    _SEP1 = 42
    _SEP2 = 82

    
    _BAR_Y1 = 78
    _BAR_Y2 = 81

    def draw(
        self,
        img,
        drowsy:        bool,
        ear:           float,
        perclos:       float = 0.0,
        perclos_alert: bool  = False,
        yawning:       bool  = False,
        yawn_alert:    bool  = False,
        mar:           float = 0.0,
        yawn_count:    int   = 0,
    ):
        h, w = img.shape[:2]

        # ── fundo ─────────────────────────────────────────────────────
        cv2.rectangle(img, (0, 0), (w, self.HUD_HEIGHT), (20, 20, 20), -1)

        # ── separadores ───────────────────────────────────────────────
        cv2.line(img, (0, self._SEP1), (w, self._SEP1), (50, 50, 50), 1)
        cv2.line(img, (0, self._SEP2), (w, self._SEP2), (50, 50, 50), 1)

        # ── linha 1: EAR ──────────────────────────────────────────────
        ear_color = self.COLOR_EAR_ALERT if drowsy else self.COLOR_EAR_OK
        ear_label = "SONOLENCIA" if drowsy else "NORMAL"

        cv2.putText(img, ear_label, (12, self._Y1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, ear_color, 2, cv2.LINE_AA)
        cv2.putText(img, f"EAR {ear:.2f}", (w - 110, self._Y1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1, cv2.LINE_AA)

        # ── linha 2: PERCLOS ──────────────────────────────────────────
        pct_color = self.COLOR_PERCLOS_ALERT if perclos_alert else self.COLOR_PERCLOS_OK
        pct_label = "PERCLOS ALTO" if perclos_alert else "PERCLOS OK"

        cv2.putText(img, pct_label, (12, self._Y2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, pct_color, 1, cv2.LINE_AA)
        cv2.putText(img, f"{perclos * 100:.1f}%", (w - 110, self._Y2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, pct_color, 1, cv2.LINE_AA)

        
        bar_w = int((w - 20) * min(perclos / 0.30, 1.0))
        bar_color = self.COLOR_PERCLOS_ALERT if perclos_alert else (60, 100, 60)
        cv2.rectangle(img, (10, self._BAR_Y1), (10 + bar_w, self._BAR_Y2), bar_color, -1)

        # ── linha 3: bocejo ───────────────────────────────────────────
        
        active_yawn = yawning or yawn_alert
        yawn_color  = self.COLOR_YAWN_ALERT if active_yawn else self.COLOR_YAWN_OK
        yawn_label  = "BOCEJO" if active_yawn else "BOCEJO OK"
        count_text  = f"MAR {mar:.2f}  |  {yawn_count}x/10s"

        cv2.putText(img, yawn_label, (12, self._Y3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, yawn_color, 1, cv2.LINE_AA)
        cv2.putText(img, count_text, (w - 210, self._Y3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, yawn_color, 1, cv2.LINE_AA)

        
        cv2.putText(img, "Q: sair", (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1, cv2.LINE_AA)