import socket
import struct
import traceback
import logging
import time
import numpy as np
import cmath

from enum import Enum
import os
class msg_type(Enum):
    VIDEOPLAYER = 0
    FEEDFORWARD = 1
    RNN         = 2
    MOGLOW      = 3
    UNKNOWN = 99

    @staticmethod
    def type_as_string(move_type):
        return str(move_type).split('.')[1]


def identify_data(floatarray):
    in_msg_type = msg_type.UNKNOWN

    if cmath.isclose(floatarray[0],0.0):
        in_msg_type =  msg_type.VIDEOPLAYER
    if cmath.isclose(floatarray[0],1.0):
        in_msg_type =  msg_type.FEEDFORWARD

    return in_msg_type, floatarray[1:]

def sending_and_reciveing():
    s = socket.socket()
    socket.setdefaulttimeout(None)
    print('socket created ')
    port = 60000
    s.bind(('127.0.0.1', port)) #local host
    s.listen(30) #listening for connection for 30 sec?
    print('socket listensing ... ')
    while True:
        try:
            c, addr = s.accept() #when port connected
            bytes_received = c.recv(8192) #received bytes
            array_received = np.frombuffer(bytes_received, dtype=np.float32) #converting into float array

            nn_output = (array_received) #NN prediction (e.g. model.predict())
            print(nn_output)
            bytes_to_send = struct.pack('%sf' % len(nn_output), *nn_output) #converting float to byte
            c.sendall(bytes_to_send) #sending back
            c.close()
        except Exception as e:
            logging.error(traceback.format_exc())
            print("error")
            c.sendall(bytearray([]))
            c.close()
            break

sending_and_reciveing()