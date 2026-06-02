import json

import boto3

from models.event import Event


class MQTT:
    """
    Classe responsável por lidar com a conexão e interação com o AWS MQTT
    """

    def __init__(self, region: str, topic: str) -> None:
        self.__region = region
        self.__client = self.__create_session()
        self.__topic = topic

    def __create_session(self):
        """
        Método responsável por instanciar uma sessão no AWS MQTT
        :return:
        """
        return boto3.client("iot-data", region_name=self.__region)

    def send_payload(self, payload: Event) -> None:
        """
        Método responsável por enviar o payload de um evento ao MQTT
        :param payload: Evento do Hélio
        :return:
        """

        payload_bytes = json.dumps(payload.to_dict()).encode("utf-8")

        res = self.__client.publish(
            topic=self.__topic,
            qos=1,
            retain=False,
            payload=payload_bytes
        )