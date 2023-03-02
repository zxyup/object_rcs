from abc import ABC, abstractmethod
from typing import Mapping
import logging

class Transmission(ABC):
    
    def __init__(self, name: str):
        super().__init__()
        self.__name = name
        
    @property
    def name(self):
        return self.__name
    
    @abstractmethod
    def read(self): ...
    
    # @abstractmethod
    # def readRaw(self): ...
    
    @abstractmethod
    def write(self, *args, **kwargs): ...

    @abstractmethod
    def start(self): ...

    @abstractmethod
    def restart(self): ...
    
    @abstractmethod
    def close(self): ...
    
    # @abstractmethod
    # def attach(self, *sensors): ...
        
    def setProperties(self, properties: Mapping):
        for attr in properties:
            if not hasattr(self, attr):
                logging.warning("{} has no attribute '{}'.".format(self.__name, attr))   
            else:
                setattr(self, attr, properties.get(attr))
    