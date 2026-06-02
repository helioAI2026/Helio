from typing import List


class Event:
    """
    Modelo de evento a ser enviado ao MQTT
    """

    __device_id: str = None
    __ride_id: str = None
    __timestamp: str = None
    __nonce: str = None
    __data: List[dict] = []

    def __init__(self, device_id: str, ride_id: str, timestamp: str, nonce: str, data: List[dict] = []):
        self.__device_id = device_id
        self.__ride_id = ride_id
        self.__timestamp = timestamp
        self.__nonce = nonce
        self.__data = data

    def to_dict(self) -> dict:
        return {
            "device_id" : self.__device_id,
            "ride_id" : self.__ride_id,
            "timestamp" : self.__timestamp,
            "nonce" : self.__nonce,
            "data" : self.__data
        }
