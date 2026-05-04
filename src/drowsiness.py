from scipy.spatial import distance

class Drowsiness:
    def __init__(self, threshold, frames, cooldown):
        self.threshold = threshold
        self.frames    = frames
        self.cooldown  = cooldown
        self.counter   = 0
        self.since     = cooldown

    def _ear(self, eye):
        A = distance.euclidean(eye[1], eye[5]) # distancia de p2 a p6 (vertical esquerda do olho)
        B = distance.euclidean(eye[2], eye[4]) # distancia de p3 a p5 (vertical direita do olho)
        C = distance.euclidean(eye[0], eye[3]) # distancia de p1 a p4 (horizontal do olho)

        if C == 0:
            return 0.0
        
        return (A + B) / (2.0 * C)

    def update(self, left, right):
        if left is None or right is None:
            self.counter = 0
            return False, False, 0.0
        
        ear = (self._ear(left) + self._ear(right)) / 2.0

        self.counter = self.counter + 1 if ear < self.threshold else 0

        self.since  += 1

        drowsy = self.counter >= self.frames

        alert  = drowsy and self.since >= self.cooldown

        if alert:
            self.since = 0

        return drowsy, alert, ear