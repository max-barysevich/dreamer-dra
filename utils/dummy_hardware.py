import numpy as np

class Core:

    def __init__(self):
        self.sequence_is_running = False
        self.getRemainingImageCount = lambda: 1
        pass

    def isSequenceRunning(self):
        return self.sequence_is_running
    
    def initializeCircularBuffer(self):
        pass

    def setExposure(self, exposure):
        pass

    def startContinuousSequenceAcquisition(self,delay):
        self.sequence_is_running = True

    def popNextTaggedImage(self):
        return taggedImage()
    
    def load_system_configuration(self,config):
        pass

    def getProperty(self,*args):
        pass

    def setProperty(self,*args):
        pass

class taggedImage:

    def __init__(self):
        self.pix = np.random.randint(0,2**16,(512*512,)).astype(np.uint16)
        self.tags = {'Height':512,'Width':512}
    
class Task:

    def __init__(self):
        self.do_channels = DAQ_channel()
        self.ao_channels = DAQ_channel()
    
    def __enter__(self):
        return self

    def __exit__(self,type,value,traceback):
        self.close()

    def write(self, data):
        pass

    def close(self):
        pass

class DAQ_channel:
    
    def __init__(self):
        pass
    
    def add_do_chan(self,chan):
        pass

    def add_ao_voltage_chan(self,chan):
        pass

class DAQ:
    Task = Task