"""
@author: 
@date:
@version:
"""

from collections import namedtuple
from abc import ABC, abstractmethod



class Sensor(ABC):
    
    def __init__(self, name: str):
        super().__init__()
        self.__name = name
        
    @property
    def name(self):
        return self.__name
    
    @abstractmethod
    def readRaw(self, *args, **kwargs): ...
    
    @abstractmethod
    def read(self, *args, **kwargs): ...
    
    @abstractmethod
    def close(self, *args, **kwargs): ...
    
    @abstractmethod
    def start(self, *args, **kwargs): ...
    
    @abstractmethod
    def update(self, *args, **kwargs): ...

    def restart(self): 
        self.close()
        self.start()
        
    def __repr__(self):
        return "Sensor({})".format(self.__name)