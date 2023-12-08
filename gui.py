import napari
from PyQt5.QtWidgets import QPushButton
from PyQt5.QtCore import QTimer

import time

import multiprocessing as mp
import threading

from live import DRA

class GUI:

    # TODO: Add run without saving
    # TODO: Add load MM config
    # TODO: Add load DRA config
    # TODO: Load and trace agent outside run()

    def __init__(self):

        self.viewer = napari.Viewer()

        self.vis_queue = mp.Queue()

        self.stop_acq = mp.Value('b',False)
        self.stop_agent = mp.Value('b',False)
        self.stop_save = mp.Value('b',False)
        self.stop_vis = mp.Value('b',False)

        self.backend = DRA(self.stop_acq,
                           self.stop_agent,
                           self.stop_save,
                           self.stop_vis,
                           self.vis_queue)

        self.start_button = QPushButton('Start')
        self.viewer.window.add_dock_widget(self.start_button)

        self.stop_button = QPushButton('Stop')
        self.viewer.window.add_dock_widget(self.stop_button)

        self.start_button.clicked.connect(self._start_backend)
        self.stop_button.clicked.connect(self._stop_backend)

        self.timer = QTimer()
        self.timer.setInterval(20)
        self.timer.timeout.connect(self.update_gui)
        self.timer.start()

        self.layer = None

    def update_gui(self):
        if not self.vis_queue.empty():
            img = self.vis_queue.get()
            if self.layer is None or len(self.viewer.layers) == 0:
                self.layer = self.viewer.add_image(img,name='DRA')
            else:
                self.layer.data = img
            
            if not self.vis_queue.empty():
                while not self.vis_queue.empty():
                    _ = self.vis_queue.get()
    
    def _start_backend(self):
        self.stop_acq.value = False
        self.stop_agent.value = False
        self.stop_save.value = False
        self.stop_vis.value = False

        self.backend_thread = threading.Thread(target=self.backend.run)
        self.backend_thread.start()
    
    def _stop_backend(self):
        self.stop_acq.value = True
        print('Attempting to join backend thread in 20 seconds.')
        time.sleep(20)
        self.backend_thread.join()
        print('Backend thread joined.')

if __name__ == '__main__':
    mp.set_start_method('spawn')
    gui = GUI()
    napari.run()