import nidaqmx
import serial

import jax
import jax.numpy as jnp

import numpy as np
import einops
import tifffile

import time
import datetime

import os
import platform

import multiprocessing as mp

from pycromanager import Core # move to pymmcoreplus?

import dreamerv3
from dreamerv3 import embodied

class DRA:
    def __init__(self):
        # start_headless
        self.tracing_done = mp.Value('b',False)

        self.stop_acq = mp.Value('b',False) # probably define outside, make a required arg for init

        self.stop_agent = mp.Value('b',False)
        self.stop_save = mp.Value('b',False)
        self.stop_vis = mp.Value('b',False)

        self.trigger_channel = 'Dev1/port0/line2' # remove after rewriting Task()s for arbitrary hardware
        self.illumination_channel = 'Dev1/ao1'
        pass

    def setup(self):
        # make a separate class that is passed to init of DRA, with fns like set_galvo(), set_aotf(), etc. for configurability?
        # load config file
        # override with individual functions, e.g. def set_exposure_time(self,exposure_time)
        pass

    def acquire(self):
        # consider pre-setting acquisition via MMCore for speed, adjust illumination through python in effectively a separate thread
        core = self._prepare_camera()

        # find ROI for agent
        agent_roi = self._find_roi(core)

        # wait until all GPU processing fns have been traced
        while not self.tracing_done.value:
            time.sleep(0.001)

        with (nidaqmx.Task() as trigger,
              nidaqmx.Task() as illum_task): # TODO: rewrite for arbitrary hardware - make separate classes for clarity
            
            trigger.do_channels.add_do_chan(self.trigger_channel)
            illum_task.ao_channels.add_ao_voltage_chan(self.illumination_channel)

            # start acquisition loop
            start = True
            while not self.stop_acq.value:
                
                # always start by acquiring a reference image
                if start:
                    action = 1
                    start = False
                else:
                    # wait for action
                    while self.action_queue.empty():
                        time.sleep(0.001)
                
                    action = self.action_queue.get()
                
                # set illumination
                self._set_illumination(illum_task,action)
                
                # snap
                self.snap(trigger)
                '''
                # It would be nice to process the previous image here,
                # while the next one is being acquired.
                #
                # This removes a potential delay between frame acquisition,
                # but introduces a delay between action and response.
                #
                # This can be implemented in the training environment.
                #
                # The delay between frame acquisition may not be an issue
                # if there are gaps between frames.
                '''
                img = self._get_image(core)
                # if gaps between frames, turn off illumination here

                self._send_img(img,agent_roi,action)
            
            # stop acquisition
            self._stop(core)

        pass

    def save(self):
        
        with tifffile.TiffWriter(self.file_name,bigtiff=True) as tif:

            while not self.stop_save.value: # maybe also check if queue is empty
                
                while self.save_queue.empty():
                    time.sleep(0.001)
                
                save_dict = self.save_queue.get()

                img = save_dict['img']
                metadata = {'IsReference': save_dict['is_ref']} # create full metadata later
                # IsReference is retrievable with json.loads(tif.pages[i].tags['ImageDescription'].value)['IsReference']

                tif.write(img,metadata=metadata)

        pass

    def vis(self):
        # remember to normalize
        pass

    def process(self):
        # set up the agent
        agent = self._prepare_agent()

        # create normalisation pre-processing function
        self._create_normalise_fn()

        # create dummy reference - this way it should always be in the same place in memory
        self.ref = jnp.zeros((1,128,128,1))

        # notify the acquisition thread that tracing is done
        self.tracing_done.value = True

        # start processing loop
        state = None
        while not self.stop_agent.value:
            
            # put most recent frame on GPU and wrap it for the agent
            obs = self._get_obs()

            outs, state = agent.policy(obs,state=state,mode='eval')

            action = np.argmax(outs['action'][0])

            self.action_queue.put(action)

        pass

    ### Backend ###

    def _prepare_agent(self):

        # load and trace agent

        ckpt = self.ckpt

        config = embodied.Config(dreamerv3.configs['defaults'])
        config = config.update(dreamerv3.configs['small'])
        config = config.update({
            'logdir': '/dev/null' if platform.system()=='Linux' else 'NUL',
            'batch_size': 1,
            # turn off jax preallocation to avoid OOM by napari? or allocate less than the default
            # check precision - switch to bfloat16 if possible
            'encoder.cnn_keys': ['obs','ref'],
            'decoder.cnn_keys': 'obs',
        })
        config = embodied.Flags(config).parse()

        act_space = {'action':embodied.Space(np.float32,shape=(2,),low=0.,high=1.),
                     'reset':embodied.Space(np.bool,shape=(),low=False,high=True)}
        
        obs_space = {'obs':embodied.Space(np.float32,shape=(128,128,1),low=0.,high=1.), # TODO: fix dims
                     'ref':embodied.Space(np.float32,shape=(128,128,1),low=0.,high=1.), # TODO: fix dims
                     'reward':embodied.Space(np.float32,shape=(),low=-np.inf,high=np.inf),
                     'is_first':embodied.Space(np.bool,shape=(),low=False,high=True),
                     'is_last':embodied.Space(np.bool,shape=(),low=False,high=True),
                     'is_terminal':embodied.Space(np.bool,shape=(),low=False,high=True)}
        
        step = embodied.Counter()

        agent = dreamerv3.Agent(obs_space, act_space, step, config)

        checkpoint = embodied.Checkpoint()
        checkpoint.agent = agent
        checkpoint.load(ckpt, keys=['agent'])

        dummy_input = jnp.ones((1,128,128,1)) # TODO: fix dims
        dummy_obs = {
            'obs': dummy_input,
            'ref': dummy_input,
            'reward': np.array([0.]),
            'is_first': np.array([True]),
            'is_last': np.array([False]),
            'is_terminal': np.array([False])
        }

        _ = agent.policy(dummy_obs,state=None,mode='eval')

        return agent

    def _prepare_camera(self):

        core = Core()

        if core.is_sequence_running():
            core.stop_sequence_acquisition()
        
        core.initialize_circular_buffer()

        if self.trigger_mode != 'external exposure control':
            core.set_exposure(self.exposure_time)
        
        core.start_continuous_sequence_acquisition(0)

        return core
    
    def _find_roi(self,core):
        '''
        Finds the brightest region of the image 
        to use as the ROI for the agent.
        '''

        # take one image
        with (nidaqmx.Task() as trigger,
              nidaqmx.Task() as illum_task):
            
            trigger.do_channels.add_do_chan(self.trigger_channel)
            illum_task.ao_channels.add_ao_voltage_chan(self.illumination_channel)

            self._set_illumination(illum_task,1)
            self.snap(trigger)
            self._set_illumination(illum_task,-1)
        
        img = self._get_image(core)

        # clear buffer
        core.initialize_circular_buffer()

        # split into 128x128 regions
        _, H, W, _ = img.shape
        hi = 128
        wi = 128
        tiles = einops.rearrange(img,'b (nh hi) (nw wi) c -> (b nh nw) hi wi c',hi=hi,wi=wi)

        # calculate mean of each
        means = np.mean(tiles,axis=(1,2,3))

        # store coords of brightest region
        ind = np.argmax(means)

        rx = W // wi
        ry = H // hi

        x0 = (ind % rx) * wi
        y0 = (ind // ry) * hi

        x1 = x0 + wi
        y1 = y0 + hi

        return lambda img: img[:,x0:x1,y0:y1,:]

    def _set_illumination(self,illum_task,action):
        if action == 0:
            illum_task.write(self.illum_signal_low)
        elif action == 1:
            illum_task.write(self.illum_signal_high)
        elif action == -1:
            illum_task.write(self.illum_signal_off)

    def _create_snap_fn(self):

        if self.trigger_mode == 'external':
            def snap(trigger):
                trigger.write(True)
                trigger.write(False)

        elif self.trigger_mode == 'external exposure control':
            def snap(trigger):
                trigger.write(True)
                time.sleep(self.exposure_time)
                trigger.write(False)

        else:
            raise NotImplementedError('Trigger mode must be \'external\' or \'external exposure control\'.')
        
        self.snap = snap
    
    def _get_image(self,core):

        while core.get_remaining_image_count() == 0:
            time.sleep(0.001)

        img = core.pop_next_tagged_image()
        img = np.reshape(img.pix,newshape=[1,img.tags['Height'], img.tags['Width'],1])

        return img
    
    def _send_img(self,img,agent_roi,action):
        # send to save, vis, process

        # send dicts of {'obs':img,'is_ref':is_ref} to ensure syncronicity

        # send to process
        self.obs_queue.put({'obs':agent_roi(img),'is_ref':action==1})

        img = img[0,:,:,0]

        self.save_queue.put({'img':img,'is_ref':action==1})

        if self.vis_queue.empty():
            self.vis_queue.put(img)
    
    def _get_obs(self):

        while self.obs_queue.empty():
            time.sleep(0.001)
        
        obs_raw = self.obs_queue.get()

        obs = jnp.array(obs_raw['obs'])
        obs = self.normalise(obs)

        is_ref = obs_raw['is_ref']
        if is_ref:
            self.ref.at[:].set(obs) # self.ref = obs may be faster
        
        obs_wrapped = {
            'obs':obs,
            'ref':self.ref,
            'reward':np.array([0.]),
            'is_first':np.array([False]),
            'is_last':np.array([False]),
            'is_terminal':np.array([False])
        }

        return obs_wrapped
    
    def _create_normalise_fn(self):
        self.normalise = jax.jit(lambda x: x / x.max())
        dummy_input = jnp.ones((1,128,128,1))
        _ = self.normalise(dummy_input)
    
    def _stop(self,core):
        '''
        Stop acquisition and all processes
        '''
        # stop acquisition
        core.stop_sequence_acquisition()

        # send stop signal to processing, save and vis
        self.stop_agent.value = True
        self.stop_save.value = True
        self.stop_vis.value = True