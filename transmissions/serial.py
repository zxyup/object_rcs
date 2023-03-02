import logging
import time
import struct
import serial
import base64
from transmissions.transmission import Transmission



class Serial(Transmission):
    
    def __init__(self, name: str):
        super().__init__(name)
        self.buf_size = 1024
        self.baudrate: int = 9600
        self.bytesize: int = 8
        self.parity: str = serial.PARITY_NONE
        self.stopbits: int = serial.STOPBITS_ONE
        self.timeout: float = .1
        self.prefix = struct.pack(">B", 0xFF)
        self.suffix = struct.pack(">B", 0xEF)
        self.data = -1
        self.port = '/dev/ttyTHS0'
    
    def start(self):
        self.port = serial.Serial(
                    self.port, 
                    self.baudrate, 
                    self.bytesize, 
                    self.parity, 
                    self.stopbits, 
                    self.timeout
                    )
    
    def read(self):
        # self.data=-1
        try:
            if self.port.inWaiting() > 0:
                # print('hhe')

                # self.data=self.port.read(self.buf_size)
                tmp = self.port.read(self.buf_size)
                # print(1111)
                # tmp = base64.b64decode(tmp)
                tmp=tmp.decode('utf8')
                # print(2222)
                self.data = int(tmp)
                
                    # .decode(encoding=encoding) 视情况而定
        except:
            print('no open')
        
        return self.data
    
    
    def restart(self):
        self.port.close()
        time.sleep(0.1)
        self.port.open()
    
    def close(self):
        self.port.close()
    
    def write(self, val1: int, val2:int, bits: int = 16):
        if bits not in [8, 16]:
            raise ValueError('bits only support 8 or 16.')
        ubound = {8:255, 16:65535}
        fmt = {8:'>B', 16:'>H'}
        if val1 > ubound[bits]:  
            val1 = ubound[bits]
            val2 = ubound[bits]
        elif val1 < 0: 
            val1 = 0
            val2 = 0

        
	    
        data1 = struct.pack(fmt[bits], val1)
        data2 = struct.pack(fmt[bits], val2)
        data = self.prefix  +data1 +data2+ self.suffix
        # print(data)
        # self.port.write(b'\xef\x00')
        return self.port.write(data)
 

        
