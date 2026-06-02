import hmac, hashlib, os, time

class Ride:
    __device_id: str = None
    __device_secret: bytes = None

    def __init__(self, device_id: str, device_secret: bytes):
        self.__device_id = device_id
        self.__device_secret = device_secret

    def generate_ride_id(self) -> dict:
        timestamp = int(time.time())
        nonce = os.urandom(4).hex()  # 8 hex chars

        message = f"{self.__device_id}:{timestamp}:{nonce}".encode()
        ride_id = hmac.new(self.__device_secret, message, hashlib.sha256).hexdigest()

        return {
            "ride_id": ride_id,
            "timestamp": timestamp,
            "nonce": nonce
        }
