import numpy as np
import numpy.random as nr
from screeninfo import get_monitors
import sys
import struct
from datoviz import canvas, run, colormap
import time
import math
import madmom
from utilities import *
import audioio as aio

from pathlib import Path

extensions = ["mp3", "wav", "ogg", "flac", "m4a"]  # we will look for all those file types.


screen = get_monitors()[0]
aspect_ratio = screen.width / screen.height

c = canvas(show_fps=True, width = screen.width*1.18, height = screen.height*1.18)
ctx = c.gpu().context()
panel = c.scene().panel(controller='camera', depth_test = True)

global pos, pos_a



                               
global draw_mode
draw_mode = 1

global read_fps, draw_fps
read_fps = 256
draw_fps = 4*read_fps
wait_time = 0.0

#timing
global start_time, delay, dt, span, t0, t1, meas_span_a, meas_span_e, read_elapsed, draw_elapsed, measurements
start_time = time.perf_counter()
t0 = start_time
t1 = start_time
delay = dt = span = meas_span_a = meas_span_e = read_elapsed = draw_elapsed = measurements = 0.0

global offset
offset = 0.15

#_history
global history__drums_type1, history__drums_type1_x, history__drums_type1_y, history__drums_type2, history__drums_type2_x, history__drums_type2_y,  history__drums_type3, history__drums_type3_x, history__drums_type3_y, history__drums_type4, history__drums_type4_x, history__drums_type4_y, history_et
global history_ax, history_ay, history_az, history_at



# Mode 0
n = 256

if(draw_mode == 0):
    
    visual_0 = panel.visual('path', transform=None, depth_test=True)
    visual_0.data('length', np.array([n, n, n, n, n, n, n, n, n, n, n, n,]))
else:
    visual_1 = panel.visual('marker')
        
        
x = np.linspace(-1.0, 1.0, n)
y = np.zeros(n)
z = x*y

pos = np.c_[x, y, z]  # an (N, 3) array with the coordinates of the path vertices.
pos_a = np.c_[np.ones(1), np.ones(1), np.ones(1)]

history_et = history_at = np.array(pos[:,0])
history_ax = history_ay = history_az = np.array(pos[:,1])



fade = np.arange(n)
zeros = np.zeros(n)


##Drums 
cmap_mycmap_drums_1_5 = np.c_[fade*0.99, fade*0.01, fade*0.02, fade]
ctx.colormap('mycmap_drums_1_5', cmap_mycmap_drums_1_5.astype(np.uint8))


cmap_mycmap_drums_2_6 = np.c_[fade*0.7, fade*0.01, fade*0.02, fade]
ctx.colormap('mycmap_drums_2_6', cmap_mycmap_drums_2_6.astype(np.uint8))


cmap_mycmap_drums_3_7 = np.c_[fade*0.6, fade*0.01, fade*0.02, fade]
ctx.colormap('mycmap_drums_3_7', cmap_mycmap_drums_3_7.astype(np.uint8))


cmap_mycmap_drums_4_8 = np.c_[fade*0.5, fade*0.01, fade*0.02, fade]
ctx.colormap('mycmap_drums_4_8', cmap_mycmap_drums_4_8.astype(np.uint8))


##Bass 
cmap_mycmap_bass_1_5 = np.c_[fade*0.4, fade*0.3, 0.3*fade, fade]
ctx.colormap('mycmap_bass_1_5', cmap_mycmap_bass_1_5.astype(np.uint8))


cmap_mycmap_bass_2_6 = np.c_[fade*0.35, fade*0.2, 0.2*fade, fade]
ctx.colormap('mycmap_bass_2_6', cmap_mycmap_bass_2_6.astype(np.uint8))


cmap_mycmap_bass_3_7 = np.c_[fade*0.3, fade*0.05, 0.05*fade, fade]
ctx.colormap('mycmap_bass_3_7', cmap_mycmap_bass_3_7.astype(np.uint8))


cmap_mycmap_bass_4_8 = np.c_[fade*0.25, fade*0.05, 0.05*fade, fade]
ctx.colormap('mycmap_bass_4_8', cmap_mycmap_bass_4_8.astype(np.uint8))


##Other 
cmap_mycmap_other_1_5 = np.c_[fade*0.01, fade*0.1, 0.99*fade, fade]
ctx.colormap('mycmap_other_1_5', cmap_mycmap_other_1_5.astype(np.uint8))


cmap_mycmap_other_2_6 = np.c_[fade*0.01, fade*0.1, 0.8*fade, fade]
ctx.colormap('mycmap_other_2_6', cmap_mycmap_other_2_6.astype(np.uint8))


cmap_mycmap_other_3_7 = np.c_[fade*0.01, fade*0.01, 0.6*fade, fade]
ctx.colormap('mycmap_other_3_7', cmap_mycmap_other_3_7.astype(np.uint8))


cmap_mycmap_other_4_8 = np.c_[fade*0.01, fade*0.01 , 0.5*fade, fade]
ctx.colormap('mycmap_other_4_8', cmap_mycmap_other_4_8.astype(np.uint8))


##Vocals 
cmap_mycmap_vocals_1_5 = np.c_[fade,fade, fade, fade]
ctx.colormap('mycmap_vocals_1_5', cmap_mycmap_vocals_1_5.astype(np.uint8))


cmap_mycmap_vocals_2_6 = np.c_[0.925*fade,0.925*fade, 0.925*fade, fade]
ctx.colormap('mycmap_vocals_2_6', cmap_mycmap_vocals_2_6.astype(np.uint8))


cmap_mycmap_vocals_3_7 = np.c_[0.7*fade,0.7*fade, 0.7*fade, fade]
ctx.colormap('mycmap_vocals_3_7', cmap_mycmap_vocals_3_7.astype(np.uint8))


cmap_mycmap_vocals_4_8 = np.c_[0.5*fade,0.5*fade, 0.5*fade, fade]
ctx.colormap('mycmap_vocals_4_8', cmap_mycmap_vocals_4_8.astype(np.uint8))


N = 4*64



###Drums
#x = np.linspace(-1.0, 1.0, N)
#y = np.cos(x)
#x = np.sin(x)
#z = x*y
#pos_2 = np.c_[x, y, z]
#ms = 15*np.ones(N)
#
## We use a built-in colormap
#color_values = np.ones(N)
##alpha = ms * np.ones(N)
## (N, 4) array of uint8
#color = colormap(color_values, vmin=0, vmax=1, cmap='mycmap_drums_1_5')


# Add a first column Drums
history__drums_type1 = history__drums_type1_x = history__drums_type1_y = history__drums_type1_z = np.array(pos[:,1])
history__drums_type2 = history__drums_type2_x = history__drums_type2_y = history__drums_type2_z = np.array(pos[:,1])
history__drums_type3 = history__drums_type3_x = history__drums_type3_y = history__drums_type3_z = np.array(pos[:,1])
history__drums_type4 = history__drums_type4_x = history__drums_type4_y = history__drums_type4_z = np.array(pos[:,1])

# Add second column Bass
history__bass_type1 = history__bass_type1_x = history__bass_type1_y = history__bass_type1_z = np.array(pos[:,1])
history__bass_type2 = history__bass_type2_x = history__bass_type2_y = history__bass_type2_z = np.array(pos[:,1])
history__bass_type3 = history__bass_type3_x = history__bass_type3_y = history__bass_type3_z = np.array(pos[:,1])
history__bass_type4 = history__bass_type4_x = history__bass_type4_y = history__bass_type4_z = np.array(pos[:,1])

# Add third column Other
history__other_type1 = history__other_type1_x = history__other_type1_y = history__other_type1_z = np.array(pos[:,1])
history__other_type2 = history__other_type2_x = history__other_type2_y = history__other_type2_z = np.array(pos[:,1])
history__other_type3 = history__other_type3_x = history__other_type3_y =  history__other_type3_z = np.array(pos[:,1])
history__other_type4 = history__other_type4_x = history__other_type4_y = history__other_type4_z = np.array(pos[:,1])

# Add forth column Vocals
history__vocals_type1 = history__vocals_type1_x = history__vocals_type1_y = history__vocals_type1_z = np.array(pos[:,1])
history__vocals_type2 = history__vocals_type2_x = history__vocals_type2_y = history__vocals_type2_z = np.array(pos[:,1])
history__vocals_type3 = history__vocals_type3_x = history__vocals_type3_y =  history__vocals_type3_z = np.array(pos[:,1])
history__vocals_type4 = history__vocals_type4_x = history__vocals_type4_y =  history__vocals_type4_z = np.array(pos[:,1])




si_vals = [0.0, 0.0] #euler, accel

from os import path

basepath = path.dirname(__file__)


stem = sys.argv[1:][0]

track_path = basepath + "/separated/mdx_extra/" + stem + "/"
audio_path = basepath + "/" + stem

def find_files(in_path):
        out_2 = []
        for file in Path(in_path).iterdir():
            if file.suffix.lower().lstrip(".") in extensions:
                out_2.append(file)

        return out_2
        
files = [str(f) for f in find_files(track_path)]

out_2 = [str(f) for f in find_files(basepath)]

for file in out_2:
    if str(file).find(stem) != -1:
        print ("Found!")
        print(file)
        audio_path = file
#signal_drums = madmom.audio.signal.Signal(files[1], sample_rate = 44100,dtype = np.float32,num_channels=1)
#signal_bass = madmom.audio.signal.Signal(files[0], sample_rate = 44100,dtype = np.float32,num_channels=1)
#signal_other = madmom.audio.signal.Signal(files[2], sample_rate = 44100,dtype = np.float32,num_channels=1)
#signal_vocals = madmom.audio.signal.Signal(files[3], sample_rate = 44100,dtype = np.float32,num_channels=1)

_track_intensity = np.loadtxt(track_path+'intensity.csv', delimiter=',')
intensity_max = np.max(_track_intensity)
intensity_min = np.min(_track_intensity)
#intensity_cutoff = np.mean(_track_intensity)/(intensity_max - intensity_min) - 0.2*((np.abs(intensity_max) - np.abs(intensity_min))/(intensity_max + intensity_min))*(intensity_max - intensity_min)
intensity_cutoff = (intensity_max + intensity_min)*0.5

_track1_rms = np.loadtxt(track_path+'drums_trend.csv', delimiter=',')
_track2_rms = np.loadtxt(track_path+'bass_trend.csv', delimiter=',')
_track3_rms = np.loadtxt(track_path+'other_trend.csv', delimiter=',')
_track4_rms = np.loadtxt(track_path+'vocals_trend.csv', delimiter=',')


_track1_types = np.loadtxt(track_path+'drums_beats_cd.csv', delimiter=',')
_track2_types = np.loadtxt(track_path+'bass_beats_cd.csv', delimiter=',')
_track3_types = np.loadtxt(track_path+'other_beats_cd.csv', delimiter=',')
_track4_types = np.loadtxt(track_path+'vocals_beats_cd.csv', delimiter=',')

print("0: " + files[0])
print("1: " + files[1])
print("2: " + files[2])
print("3: " + files[3])
#data, samplingrate = aio.load_audio(audio_path+'.mp3')
play_data = madmom.audio.signal.Signal(audio_path, dtype = np.float32,num_channels=2)
track_data = madmom.audio.signal.Signal(audio_path, sample_rate = 44100,dtype = np.float32,num_channels=1)
data_sample_rate = track_data.sample_rate
data__track1 = madmom.audio.signal.Signal(files[0], sample_rate = 44100,dtype = np.float32,num_channels=1)
data__track2 = madmom.audio.signal.Signal(files[1], sample_rate = 44100,dtype = np.float32,num_channels=1)
data__track3 = madmom.audio.signal.Signal(files[2], sample_rate = 44100,dtype = np.float32,num_channels=1)
data__track4 = madmom.audio.signal.Signal(files[3], sample_rate = 44100,dtype = np.float32,num_channels=1)

#_track_ratio= (_track_type2.shape[0] / data.shape[0])
_track1_rms_ratio= _track1_rms.shape[0]/(data__track1.shape[0]/data__track1.sample_rate) 
_track2_rms_ratio= _track2_rms.shape[0]/(data__track2.shape[0]/data__track2.sample_rate) 
_track3_rms_ratio= _track3_rms.shape[0]/(data__track3.shape[0]/data__track3.sample_rate) 
_track4_rms_ratio= _track4_rms.shape[0]/(data__track4.shape[0]/data__track4.sample_rate) 

_track1_fps= _track1_types.shape[0]/(data__track1.shape[0]/data__track1.sample_rate) 
_track2_fps= _track2_types.shape[0]/(data__track2.shape[0]/data__track2.sample_rate) 
_track3_fps= _track3_types.shape[0]/(data__track3.shape[0]/data__track3.sample_rate) 
_track4_fps= _track4_types.shape[0]/(data__track4.shape[0]/data__track4.sample_rate) 



delay_multiplier = 0.75

print(_track1_rms.shape, _track1_rms_ratio, (data__track1.shape[0]/data__track1.sample_rate))


global intensity, beat_offset,  current_frame, current_time,_track_beat_count, _track1_beat_count,_track_beat_type2, _track1_beat_rms, _track2_beat_count, _track2_beat_rms, _track3_beat_count, _track3_beat_rms
current_frame = current_time = measurements = _track_beat_count = _track1_beat_count = _track_beat_type2= _track2_beat_count = _track3_beat_count = _track4_beat_count =  _track1_beat_rms =  _track2_beat_rms = _track3_beat_rms = _track4_beat_rms = intensity = beat_offset = 0.0


delay = 0

data = np.hstack((track_data, data__track1))
data = np.hstack((data, data__track2))
data = np.hstack((data, data__track3))
data = np.hstack((data, data__track4))


import multiprocessing
from multiprocessing import Queue
global queue
#queue = multiprocessing.Manager().Queue()
#def main_read():
#    global s1, read_elapsed, read_fps, dt, current_frame, current_time, span, delay
#    
#    if(read_elapsed > 1/read_fps):
#        read_elapsed = 0
#        current_frame = int(data_sample_rate*(span - delay*delay_multiplier))
#    queue.put_nowait(data[int(current_frame)])

def draw_mode_0():
    ##Drums
    pos[:,0] = history__drums_type1_x*10.0
    pos[:,1] = -1.0+1.5*aspect_ratio*history__drums_type1_y
    pos[:,2] = aspect_ratio*history__drums_type1_z
    color = colormap(np.linspace(0, 1, n), vmin=0, vmax=1, cmap='mycmap_drums_1_5')
    draw_Path(visual_0, pos, 16+64*((_track1_beat_rms*_track2_beat_rms*_track2_beat_rms - 1*_track1_beat_rms*_track1_beat_rms*_track2_beat_rms)*(1-4.0*intensity)), color, 0)

    pos[:,0] = history__drums_type2_x*10.0
    pos[:,1] = -1.0+1.5*aspect_ratio*history__drums_type2_y
    pos[:,2] = aspect_ratio*history__drums_type2_z
    color = colormap(np.linspace(0, 1, n), vmin=0, vmax=1, cmap='mycmap_drums_2_6')
    draw_Path(visual_0, pos, 16+64*((_track1_beat_rms*_track2_beat_rms*_track2_beat_rms - 1*_track1_beat_rms*_track1_beat_rms*_track2_beat_rms)*(1-4.0*intensity)), color, 1)

    pos[:,0] = history__drums_type3_x*10.0
    pos[:,1] = -1.0+1.5*aspect_ratio*history__drums_type3_y
    pos[:,2] = aspect_ratio*history__drums_type3_z
    color = colormap(np.linspace(0, 1, n), vmin=0, vmax=1, cmap='mycmap_drums_3_7')
    draw_Path(visual_0, pos, 16+64*((_track1_beat_rms*_track2_beat_rms*_track2_beat_rms - 1*_track1_beat_rms*_track1_beat_rms*_track2_beat_rms)*(1-4.0*intensity)), color, 1)

    pos[:,0] = history__drums_type4_x*10.0
    pos[:,1] = -1.0+1.5*aspect_ratio*history__drums_type4_y
    pos[:,2] = aspect_ratio*history__drums_type4_z
    color = colormap(np.linspace(0, 1, n), vmin=0, vmax=1, cmap='mycmap_drums_4_8')
    draw_Path(visual_0, pos, 16+64*((_track1_beat_rms*_track2_beat_rms*_track2_beat_rms - 1*_track1_beat_rms*_track1_beat_rms*_track2_beat_rms)*(1-4.0*intensity)), color, 1)

    
    ##Other
    pos[:,0] = history__other_type1_x*10.0
    pos[:,1] = -1.0+1.5*aspect_ratio*history__other_type1_y
    pos[:,2] = aspect_ratio*history__other_type1_z
    color = colormap(np.linspace(0, 1, n), vmin=0, vmax=1, cmap='mycmap_other_1_5')
    draw_Path(visual_0, pos, 16+64*((_track3_beat_rms*_track4_beat_rms*_track4_beat_rms - 1*_track3_beat_rms*_track3_beat_rms*_track4_beat_rms)*(1-4.0*intensity)), color, 1)

    pos[:,0] = history__other_type2_x*10.0
    pos[:,1] = -1.0+1.5*aspect_ratio*history__other_type2_y
    pos[:,2] = aspect_ratio*history__other_type2_z
    color = colormap(np.linspace(0, 1, n), vmin=0, vmax=1, cmap='mycmap_other_2_6')
    draw_Path(visual_0, pos, 16+64*((_track3_beat_rms*_track4_beat_rms*_track4_beat_rms - 1*_track3_beat_rms*_track3_beat_rms*_track4_beat_rms)*(1-4.0*intensity)), color, 1)
     

    pos[:,0] = history__other_type3_x*10.0
    pos[:,1] = -1.0+1.5*aspect_ratio*history__other_type3_y
    pos[:,2] = aspect_ratio*history__other_type3_z
    color = colormap(np.linspace(0, 1, n), vmin=0, vmax=1, cmap='mycmap_other_3_7')
    draw_Path(visual_0, pos, 16+64*((_track3_beat_rms*_track4_beat_rms*_track4_beat_rms - 1*_track3_beat_rms*_track3_beat_rms*_track4_beat_rms)*(1-4.0*intensity)), color, 1)

    pos[:,0] = history__other_type4_x*10.0
    pos[:,1] = -1.0+1.5*aspect_ratio*history__other_type4_y
    pos[:,2] = aspect_ratio*history__other_type4_z
    color = colormap(np.linspace(0, 1, n), vmin=0, vmax=1, cmap='mycmap_other_4_8')
    draw_Path(visual_0, pos, 16+64*((_track3_beat_rms*_track4_beat_rms*_track4_beat_rms - 1*_track3_beat_rms*_track3_beat_rms*_track4_beat_rms)*(1-4.0*intensity)), color, 1)

    
    ##Vocals
    pos[:,0] = history__vocals_type1_x*10.0
    pos[:,1] = -1.0+1.5*aspect_ratio*history__vocals_type1_y
    pos[:,2] = aspect_ratio*history__vocals_type1_z
    color = colormap(np.linspace(0, 1, n), vmin=0, vmax=1, cmap='mycmap_vocals_1_5')
    draw_Path(visual_0, pos, 16+64*((_track4_beat_rms*_track3_beat_rms*_track3_beat_rms - 1*_track4_beat_rms*_track4_beat_rms*_track3_beat_rms)*(1-4.0*intensity)), color, 1)


    pos[:,0] = history__vocals_type2_x*10.0
    pos[:,1] = -1.0+1.5*aspect_ratio*history__vocals_type2_y
    pos[:,2] = aspect_ratio*history__vocals_type2_z
    color = colormap(np.linspace(0, 1, n), vmin=0, vmax=1, cmap='mycmap_vocals_2_6')
    draw_Path(visual_0, pos, 16+64*((_track4_beat_rms*_track3_beat_rms*_track3_beat_rms - 1*_track4_beat_rms*_track4_beat_rms*_track3_beat_rms)*(1-4.0*intensity)), color, 1)
     

    pos[:,0] = history__vocals_type3_x*10.0
    pos[:,1] = -1.0+1.5*aspect_ratio*history__vocals_type3_y
    pos[:,2] = aspect_ratio*history__vocals_type3_z
    color = colormap(np.linspace(0, 1, n), vmin=0, vmax=1, cmap='mycmap_vocals_3_7')
    draw_Path(visual_0, pos, 16+64*((_track4_beat_rms*_track3_beat_rms*_track3_beat_rms - 1*_track4_beat_rms*_track4_beat_rms*_track3_beat_rms)*(1-4.0*intensity)), color, 1)

    pos[:,0] = history__vocals_type4_x*10.0
    pos[:,1] = -1.0+1.5*aspect_ratio*history__vocals_type4_y
    pos[:,2] = aspect_ratio*history__vocals_type4_z
    color = colormap(np.linspace(0, 1, n), vmin=0, vmax=1, cmap='mycmap_vocals_4_8')
    draw_Path(visual_0, pos, 16+64*((_track4_beat_rms*_track3_beat_rms*_track3_beat_rms - 1*_track4_beat_rms*_track4_beat_rms*_track3_beat_rms)*(1-4.0*intensity)), color, 1)


def draw_mode_1():
    ###Drums
    x = np.hstack((history__drums_type1_y[-256:], history__drums_type2_y[-256:], history__drums_type3_y[-256:], history__drums_type4_y[-256:]))
    
    
    
    y = np.cos((intensity*x+57.3*x)) * (_track1_beat_rms /(1+ (_track2_beat_rms+_track3_beat_rms+_track4_beat_rms)))
    x = np.sin((intensity*x+57.3*x)) * (_track1_beat_rms /(1+ (_track2_beat_rms+_track3_beat_rms+_track4_beat_rms)))
    
    z = 0.10*(x*y)
    pos_2 = np.c_[x, y, z]

    if(draw_mode == 1):
        ms = np.array([32+ 32*(((_track1_beat_rms*_track1_beat_rms*_track2_beat_rms)*(1)))]) + 2
    else:
        ms = np.array([4 + 8*(((_track1_beat_rms*_track1_beat_rms*_track2_beat_rms)*(1)))]) + 2
    

    color_values = -z * (ms + 1)  + 1.0
    
    alpha = 20*color_values
    # (N, 4) array of uint8
    color = colormap(color_values, vmin=0, vmax=1, alpha=alpha, cmap='mycmap_drums_2_6')
    
    draw_Dots_at(visual_1, pos_2, alpha, color, 0)



    ###Bass

    ###Other
    x = np.hstack((history__other_type1_y[-256:], history__other_type2_y[-256:], history__other_type3_y[-256:], history__other_type4_y[-256:]))

        
    y = np.cos((intensity*x+57.3*x)) * (_track3_beat_rms /(1+ (_track1_beat_rms+_track2_beat_rms+_track4_beat_rms)))
    x = np.sin((intensity*x+57.3*x)) * (_track3_beat_rms /(1+ (_track1_beat_rms+_track2_beat_rms+_track4_beat_rms)))
    
    z = 0.10*(x*y)
    pos_2 = np.c_[x, y, z]
    
    
    if(draw_mode == 1):
        ms = np.array([32+ 32*((((_track3_beat_rms*_track3_beat_rms*_track4_beat_rms))*(1)))]) + 2
    else:
        ms = np.array([4 + 8*((((_track3_beat_rms*_track3_beat_rms*_track4_beat_rms))*(1)))]) + 2
        
    

    color_values = -z * (ms + 1)  + 1.0
    
    alpha = 20*color_values
    # (N, 4) array of uint8
    color = colormap(color_values, vmin=0, vmax=1, alpha=alpha, cmap='mycmap_other_2_6')

    draw_Dots_at(visual_1, pos_2, alpha, color, 1)


    ###Vocals
    x = np.hstack((history__vocals_type1_y[-256:], history__vocals_type2_y[-256:], history__vocals_type3_y[-256:], history__vocals_type4_y[-256:]))
    #print(x)
    
    y = (intensity+0.25)*np.cos((intensity*x+57.3*x)) * (_track4_beat_rms /(1+ (_track1_beat_rms+_track2_beat_rms+_track3_beat_rms))) + (intensity+5/8)*np.sin(2/3*(intensity*x+57.3*x)) * (_track4_beat_rms /(1+ (_track1_beat_rms+_track2_beat_rms+_track3_beat_rms)))
    x = (intensity+0.25)*np.sin((intensity*x+57.3*x)) * (_track4_beat_rms /(1+ (_track1_beat_rms+_track2_beat_rms+_track3_beat_rms))) + (intensity+5/8)*np.cos(2/3*(intensity*x+57.3*x)) * (_track4_beat_rms /(1+ (_track1_beat_rms+_track2_beat_rms+_track3_beat_rms)))
    
    z = 0.10*(x*y)
    pos_2 = np.c_[x, y, z]
    #ms = 15*np.ones(N)

    # We use a built-in colormap
    # We set the visual_1 props.
    
    if(draw_mode == 1):
        ms = np.array([32+ 32*((((_track4_beat_rms*_track4_beat_rms*_track3_beat_rms))*(1)))]) + 2
    else:
        ms = np.array([4 + 8*((((_track4_beat_rms*_track4_beat_rms*_track3_beat_rms))*(1)))]) + 2
        
    

    color_values = -z * (ms + 1)  + 1.0
    
    alpha = 20*color_values
    # (N, 4) array of uint8
    color = colormap(color_values, vmin=0, vmax=1, alpha=alpha, cmap='mycmap_vocals_3_7')

    draw_Dots_at(visual_1, pos_2, alpha, color, 1)

    
def draw():
    global draw_mode
    if(draw_mode == 0):
        draw_mode_0()
    elif(draw_mode == 1):
        draw_mode_1()
#    else:
    #draw_mode_0()
    #draw_mode_1()
        

def main_write():
        global statevars, s1, sv, si_vals
        global pos, pos_a, history_ax, history_ay, history_az, history_at, history__drums_type1, history__drums_type1_x, history__drums_type1_y, history__drums_type1_z
        global history__bass_type1, history__bass_type1_x, history__bass_type1_y, history__bass_type1_z
        global history__other_type1, history__other_type1_x, history__other_type1_y, history__other_type1_z
        global history__vocals_type1, history__vocals_type1_x, history__vocals_type1_y, history__vocals_type1_z
        global history__drums_type2, history__drums_type2_x, history__drums_type2_y, history__drums_type2_z, history__drums_type3, history__drums_type3_x, history__drums_type3_y, history__drums_type3_z
        global history__drums_type4, history__drums_type4_x, history__drums_type4_y, history__drums_type4_z, history__drums_type5, history__drums_type5_x, history__drums_type5_y, history__drums_type5_z
        global history__drums_type6, history__drums_type6_x, history__drums_type6_y, history__drums_type6_z, history__drums_type7, history__drums_type7_x, history__drums_type7_y, history__drums_type7_z
        global history__drums_type8, history__drums_type8_x, history__drums_type8_y, history__drums_type8_z
        global history__bass_type2, history__bass_type2_x, history__bass_type2_y, history__bass_type2_z, history__bass_type3, history__bass_type3_x, history__bass_type3_y, history__bass_type3_z
        global history__bass_type4, history__bass_type4_x, history__bass_type4_y, history__bass_type4_z, history__bass_type5, history__bass_type5_x, history__bass_type5_y, history__bass_type5_z
        global history__bass_type6, history__bass_type6_x, history__bass_type6_y, history__bass_type6_z, history__bass_type7, history__bass_type7_x, history__bass_type7_y, history__bass_type7_z
        global history__bass_type8, history__bass_type8_x, history__bass_type8_y, history__bass_type8_z
        global history__other_type2, history__other_type2_x, history__other_type2_y, history__other_type2_z, history__other_type3, history__other_type3_x, history__other_type3_y, history__other_type3_z
        global history__other_type4, history__other_type4_x, history__other_type4_y, history__other_type4_z, history__other_type5, history__other_type5_x, history__other_type5_y, history__other_type5_z
        global history__other_type6, history__other_type6_x, history__other_type6_y, history__other_type6_z, history__other_type7, history__other_type7_x, history__other_type7_y, history__other_type7_z
        global history__other_type8, history__other_type8_x, history__other_type8_y, history__other_type8_z
        global history__vocals_type2, history__vocals_type2_x, history__vocals_type2_y, history__vocals_type2_z, history__vocals_type3, history__vocals_type3_x, history__vocals_type3_y, history__vocals_type3_z
        global history__vocals_type4, history__vocals_type4_x, history__vocals_type4_y, history__vocals_type4_z, history__vocals_type5, history__vocals_type5_x, history__vocals_type5_y, history__vocals_type5_z
        global history__vocals_type6, history__vocals_type6_x, history__vocals_type6_y, history__vocals_type6_z, history__vocals_type7, history__vocals_type7_x, history__vocals_type7_y, history__vocals_type7_z
        global history__vocals_type8, history__vocals_type8_x, history__vocals_type8_y, history__vocals_type8_z
        global measurements, queue, span, start_time, history_et, delay, dt, span, t0, t1, meas_span_a, meas_span_e, read_elapsed, current_frame
        global _track_beat_count, _track1_beat_count, _track_beat_type2, _track2_beat_count, _track3_beat_count, _track4_beat_count, current_frame, current_time, _track1_beat_rms, _track2_beat_rms, _track3_beat_rms, _track4_beat_rms
        global _track1_beat_types, _track2_beat_types, _track3_beat_types, _track4_beat_types
        global intensity, beat_offset, delay_multiplier
        #measurements = queue.get()
        
        delta_a = meas_span_a
        delta_e = meas_span_e
        
        last_span = span
        #beat_offset = dt*(intensity - (intensity_max + intensity_min)/2)/2
        beat_offset = 0
        
        _track1_beat_rms = _track2_beat_rms = _track3_beat_rms = _track4_beat_rms = 0.0


        if(_track1_beat_count < _track1_rms.shape[0] - 1):
                _track1_beat_count = ((span - delay*delay_multiplier))*_track1_rms_ratio
                _track1_type_count = ((span - delay*delay_multiplier))*_track1_fps

                _track1_beat_rms = _track1_rms[int(_track1_beat_count - 1 - beat_offset)]*((1.0/draw_fps)-draw_elapsed)*draw_fps + _track1_rms[int(_track1_beat_count - beat_offset)]*draw_elapsed*draw_fps
                _track1_beat_types = _track1_types[int(_track1_type_count - 1 - beat_offset),:]*((1.0/draw_fps)-draw_elapsed)*draw_fps + _track1_types[int(_track1_type_count - 1 - beat_offset),:]*draw_elapsed*draw_fps

        if(_track2_beat_count < _track2_rms.shape[0] - 1):
                _track2_beat_count = ((span - delay*delay_multiplier))*_track2_rms_ratio
                _track2_type_count = ((span - delay*delay_multiplier))*_track2_fps

                _track2_beat_rms = _track2_rms[int(_track2_beat_count - 1 - beat_offset)]*((1.0/draw_fps)-draw_elapsed)*draw_fps + _track2_rms[int(_track2_beat_count - beat_offset)]*draw_elapsed*draw_fps
                _track2_beat_types = _track2_types[int(_track2_type_count - 1 - beat_offset),:]*((1.0/draw_fps)-draw_elapsed)*draw_fps + _track2_types[int(_track2_type_count - 1 - beat_offset),:]*draw_elapsed*draw_fps

        if(_track3_beat_count < _track3_rms.shape[0] - 1):
                _track3_beat_count = ((span - delay*delay_multiplier))*_track3_rms_ratio
                _track3_type_count = ((span - delay*delay_multiplier))*_track3_fps

                _track3_beat_rms = _track3_rms[int(_track3_beat_count - 1 - beat_offset)]*((1.0/draw_fps)-draw_elapsed)*draw_fps + _track3_rms[int(_track3_beat_count - beat_offset)]*draw_elapsed*draw_fps
                _track3_beat_types = _track3_types[int(_track3_type_count - 1 - beat_offset),:]*((1.0/draw_fps)-draw_elapsed)*draw_fps + _track3_types[int(_track3_type_count - 1 - beat_offset),:]*draw_elapsed*draw_fps

        if(_track4_beat_count < _track4_rms.shape[0] - 1):
                _track4_beat_count = ((span - delay*delay_multiplier))*_track4_rms_ratio
                _track4_type_count = ((span - delay*delay_multiplier))*_track4_fps

                _track4_beat_rms = _track4_rms[int(_track4_beat_count - 1 - beat_offset)]*((1.0/draw_fps)-draw_elapsed)*draw_fps + _track4_rms[int(_track4_beat_count - beat_offset)]*draw_elapsed*draw_fps
                _track4_beat_types = _track4_types[int(_track4_type_count - 1 - beat_offset),:]*((1.0/draw_fps)-draw_elapsed)*draw_fps + _track4_types[int(_track4_type_count - 1 - beat_offset),:]*draw_elapsed*draw_fps

                #_track1_beat_rms[1] = _track1_beat_rms[1]
        
        delay_multiplier_target = 0.75
        
        delay_multiplier_target += 0.0*((0.2+intensity)*(_track1_beat_rms*_track2_beat_rms + _track1_beat_rms+_track2_beat_rms) - (0.4+(1-intensity))*(_track3_beat_rms+_track4_beat_rms + _track3_beat_rms*_track4_beat_rms))
        
        
        
        delay_multiplier += 0.0*(delay_multiplier_target - delay_multiplier)
        
        #print(delay_multiplier)
        
        span = last_span
        
        intensity_index = int(((span - delay*delay_multiplier))*_track4_fps)
        
        if(intensity_index < _track_intensity.shape[0]):
            intensity = _track_intensity[intensity_index]
        else:
            intensity = _track_intensity[-1]

        beat_offset *= 0.0

        
        #history_at = 1.00*np.roll(history_at,-1, axis = 0)

        ##Drums
        history__drums_type1 = 1.00*np.roll(history__drums_type1,-1, axis = 0)
        history__drums_type1_x = 1.00*np.roll(history__drums_type1_x,-1, axis = 0)
        history__drums_type1_y = 1.00*np.roll(history__drums_type1_y,-1, axis = 0)
        history__drums_type1_z = 1.00*np.roll(history__drums_type1_y,-1, axis = 0)

        history__drums_type2 = 1.00*np.roll(history__drums_type2,-1, axis = 0)
        history__drums_type2_x = 1.00*np.roll(history__drums_type2_x,-1, axis = 0)
        history__drums_type2_y = 1.00*np.roll(history__drums_type2_y,-1, axis = 0)
        history__drums_type2_z = 1.00*np.roll(history__drums_type2_y,-1, axis = 0)

        history__drums_type3 = 1.00*np.roll(history__drums_type3,-1, axis = 0)
        history__drums_type3_x = 1.00*np.roll(history__drums_type3_x,-1, axis = 0)
        history__drums_type3_y = 1.00*np.roll(history__drums_type3_y,-1, axis = 0)
        history__drums_type3_z = 1.00*np.roll(history__drums_type3_y,-1, axis = 0)

        history__drums_type4 = 1.00*np.roll(history__drums_type4,-1, axis = 0)
        history__drums_type4_x = 1.00*np.roll(history__drums_type4_x,-1, axis = 0)
        history__drums_type4_y = 1.00*np.roll(history__drums_type4_y,-1, axis = 0)
        history__drums_type4_z = 1.00*np.roll(history__drums_type4_y,-1, axis = 0)


        ##Bass
        history__bass_type1 = 1.00*np.roll(history__bass_type1,-1, axis = 0)
        history__bass_type1_x = 1.00*np.roll(history__bass_type1_x,-1, axis = 0)
        history__bass_type1_y = 1.00*np.roll(history__bass_type1_y,-1, axis = 0)
        history__bass_type1_z = 1.00*np.roll(history__bass_type1_z,-1, axis = 0)

        history__bass_type2 = 1.00*np.roll(history__bass_type2,-1, axis = 0)
        history__bass_type2_x = 1.00*np.roll(history__bass_type2_x,-1, axis = 0)
        history__bass_type2_y = 1.00*np.roll(history__bass_type2_y,-1, axis = 0)
        history__bass_type2_z = 1.00*np.roll(history__bass_type2_z,-1, axis = 0)

        history__bass_type3 = 1.00*np.roll(history__bass_type3,-1, axis = 0)
        history__bass_type3_x = 1.00*np.roll(history__bass_type3_x,-1, axis = 0)
        history__bass_type3_y = 1.00*np.roll(history__bass_type3_y,-1, axis = 0)
        history__bass_type3_z = 1.00*np.roll(history__bass_type3_z,-1, axis = 0)

        history__bass_type4 = 1.00*np.roll(history__bass_type4,-1, axis = 0)
        history__bass_type4_x = 1.00*np.roll(history__bass_type4_x,-1, axis = 0)
        history__bass_type4_y = 1.00*np.roll(history__bass_type4_y,-1, axis = 0)
        history__bass_type4_z = 1.00*np.roll(history__bass_type4_z,-1, axis = 0)


        ##Other
        history__other_type1 = 1.00*np.roll(history__other_type1,-1, axis = 0)
        history__other_type1_x = 1.00*np.roll(history__other_type1_x,-1, axis = 0)
        history__other_type1_y = 1.00*np.roll(history__other_type1_y,-1, axis = 0)
        history__other_type1_z = 1.00*np.roll(history__other_type1_z,-1, axis = 0)

        history__other_type2 = 1.00*np.roll(history__other_type2,-1, axis = 0)
        history__other_type2_x = 1.00*np.roll(history__other_type2_x,-1, axis = 0)
        history__other_type2_y = 1.00*np.roll(history__other_type2_y,-1, axis = 0)
        history__other_type2_z = 1.00*np.roll(history__other_type2_z,-1, axis = 0)

        history__other_type3 = 1.00*np.roll(history__other_type3,-1, axis = 0)
        history__other_type3_x = 1.00*np.roll(history__other_type3_x,-1, axis = 0)
        history__other_type3_y = 1.00*np.roll(history__other_type3_y,-1, axis = 0)
        history__other_type3_z = 1.00*np.roll(history__other_type3_z,-1, axis = 0)

        history__other_type4 = 1.00*np.roll(history__other_type4,-1, axis = 0)
        history__other_type4_x = 1.00*np.roll(history__other_type4_x,-1, axis = 0)
        history__other_type4_y = 1.00*np.roll(history__other_type4_y,-1, axis = 0)
        history__other_type4_z = 1.00*np.roll(history__other_type4_z,-1, axis = 0)


        ##Vocals
        history__vocals_type1 = 1.0001*np.roll(history__vocals_type1,-1, axis = 0)
        history__vocals_type1_x = 1.0001*np.roll(history__vocals_type1_x,-1, axis = 0)
        history__vocals_type1_y = 1.0001*np.roll(history__vocals_type1_y,-1, axis = 0)
        history__vocals_type1_z = 1.0001*np.roll(history__vocals_type1_z,-1, axis = 0)

        history__vocals_type2 = 1.0001*np.roll(history__vocals_type2,-1, axis = 0)
        history__vocals_type2_x = 1.0001*np.roll(history__vocals_type2_x,-1, axis = 0)
        history__vocals_type2_y = 1.0001*np.roll(history__vocals_type2_y,-1, axis = 0)
        history__vocals_type2_z = 1.0001*np.roll(history__vocals_type2_z,-1, axis = 0)

        history__vocals_type3 = 1.0001*np.roll(history__vocals_type3,-1, axis = 0)
        history__vocals_type3_x = 1.0001*np.roll(history__vocals_type3_x,-1, axis = 0)
        history__vocals_type3_y = 1.0001*np.roll(history__vocals_type3_y,-1, axis = 0)
        history__vocals_type3_z = 1.0001*np.roll(history__vocals_type3_z,-1, axis = 0)

        history__vocals_type4 = 1.0001*np.roll(history__vocals_type4,-1, axis = 0)
        history__vocals_type4_x = 1.0001*np.roll(history__vocals_type4_x,-1, axis = 0)
        history__vocals_type4_y = 1.0001*np.roll(history__vocals_type4_y,-1, axis = 0)
        history__vocals_type4_z = 1.0001*np.roll(history__vocals_type4_z,-1, axis = 0)

        history_at = 1.00*np.roll(history_at,-1, axis = 0)
        history_ax = 1.00*np.roll(history_ax,-1, axis = 0)
        history_ay = 1.00*np.roll(history_ay,-1, axis = 0)
        history_az = 1.00*np.roll(history_az,-1, axis = 0)
        
        history_et = 1.00*np.roll(history_et,-1, axis = 0)

        
        
        delta_a = meas_span_a - delta_a
        delta_e = meas_span_e - delta_e

        #print("meas_span_e:" + str(meas_span_e) + "  span:" + str(8*span))
        history_et[-1] = 0.0 # stamp - span

        ##Drums
        history__drums_type1[-1] = (history__drums_type1[-2] + (0.00 +(0.04+dt*10)*intensity*( _track1_beat_rms)*((2.0*_track1_beat_rms*_track1_beat_rms +_track4_beat_rms)/ ((1+_track3_beat_rms + _track2_beat_rms))))*(_track1_beat_types[0]  - history__drums_type1[-2])) 

        history__drums_type2[-1] = (history__drums_type2[-2] + (0.00 +(0.04+dt*10)*intensity*( _track1_beat_rms)*((2.0*_track1_beat_rms*_track1_beat_rms +_track4_beat_rms)/ ((1+_track3_beat_rms + _track2_beat_rms))))*(_track1_beat_types[1]  - history__drums_type2[-2])) 

        history__drums_type3[-1] = (history__drums_type3[-2] + (0.00 +(0.04+dt*10)*intensity*( _track1_beat_rms)*((2.0*_track1_beat_rms*_track1_beat_rms +_track4_beat_rms)/ ((1+_track3_beat_rms + _track2_beat_rms))))*(_track1_beat_types[2]  - history__drums_type3[-2])) 

        history__drums_type4[-1] = (history__drums_type4[-2] + (0.00 +(0.04+dt*10)*intensity*( _track1_beat_rms)*((2.0*_track1_beat_rms*_track1_beat_rms +_track4_beat_rms)/ ((1+_track3_beat_rms + _track2_beat_rms))))*(_track1_beat_types[3]  - history__drums_type4[-2])) 


        history__drums_type1_x = history_et 
        history__drums_type1_y = history__drums_type1_z = (history__drums_type1) 

        history__drums_type2_x = history_et 
        history__drums_type2_y = history__drums_type2_z = (history__drums_type2) 

        history__drums_type3_x = history_et 
        history__drums_type3_y = history__drums_type3_z = (history__drums_type3)

        history__drums_type4_x = history_et 
        history__drums_type4_y = history__drums_type4_z = (history__drums_type4)


        history_et[-1] = 0.0
        ##Bass
        history__bass_type1[-1] = (history__bass_type1[-2] + (0.00 +(0.04+dt*10)*intensity*( _track2_beat_rms)*((_track3_beat_rms + 2.0*_track2_beat_rms*_track2_beat_rms)/ ((1+_track1_beat_rms +_track4_beat_rms))))*(_track2_beat_types[0])  - history__bass_type1[-2]) 

        history__bass_type2[-1] = (history__bass_type2[-2] + (0.00 +(0.04+dt*10)*intensity*( _track2_beat_rms)*((_track3_beat_rms + 2.0*_track2_beat_rms*_track2_beat_rms)/ ((1+_track1_beat_rms +_track4_beat_rms))))*(_track2_beat_types[1])  - history__bass_type2[-2]) 

        history__bass_type3[-1] = (history__bass_type3[-2] + (0.00 +(0.04+dt*10)*intensity*( _track2_beat_rms)*((_track3_beat_rms + 2.0*_track2_beat_rms*_track2_beat_rms)/ ((1+_track1_beat_rms +_track4_beat_rms))))*(_track2_beat_types[2])  - history__bass_type3[-2]) 

        history__bass_type4[-1] = (history__bass_type4[-2] + (0.00 +(0.04+dt*10)*intensity*( _track2_beat_rms)*((_track3_beat_rms + 2.0*_track2_beat_rms*_track2_beat_rms)/ ((1+_track1_beat_rms +_track4_beat_rms))))*(_track2_beat_types[3])  - history__bass_type4[-2]) 


        history__bass_type1_x = history_et 
        history__bass_type1_y = history__bass_type1_z = (history__bass_type1)

        history__bass_type2_x = history_et 
        history__bass_type2_y = history__bass_type2_z = (history__bass_type2)

        history__bass_type3_x = history_et 
        history__bass_type3_y = history__bass_type3_z = (history__bass_type3)

        history__bass_type4_x = history_et 
        history__bass_type4_y = history__bass_type4_z = (history__bass_type4)


        history_et[-1] = 0.0
        ##Other
        history__other_type1[-1] = (history__other_type1[-2] + (0.00 +(0.04+dt*10)*intensity*( _track3_beat_rms)*((2.0*_track3_beat_rms*_track3_beat_rms + _track2_beat_rms)/ ((1+_track1_beat_rms +_track4_beat_rms))))*(_track3_beat_types[0]  - history__other_type1[-2]))  

        history__other_type2[-1] = (history__other_type2[-2] + (0.00 +(0.04+dt*10)*intensity*( _track3_beat_rms)*((2.0*_track3_beat_rms*_track3_beat_rms + _track2_beat_rms)/ ((1+_track1_beat_rms +_track4_beat_rms))))*(_track3_beat_types[1]  - history__other_type2[-2]))    

        history__other_type3[-1] = (history__other_type3[-2] + (0.00 +(0.04+dt*10)*intensity*( _track3_beat_rms)*((2.0*_track3_beat_rms*_track3_beat_rms + _track2_beat_rms)/ ((1+_track1_beat_rms +_track4_beat_rms))))*(_track3_beat_types[2]  - history__other_type3[-2]))  

        history__other_type4[-1] = (history__other_type4[-2] + (0.00 +(0.04+dt*10)*intensity*( _track3_beat_rms)*((2.0*_track3_beat_rms*_track3_beat_rms + _track2_beat_rms)/ ((1+_track1_beat_rms +_track4_beat_rms))))*(_track3_beat_types[3]  - history__other_type4[-2]))  


        history__other_type1_x = history_et 
        history__other_type1_y = history__other_type1_z = (history__other_type1) 

        history__other_type2_x = history_et 
        history__other_type2_y = history__other_type2_z = (history__other_type2) 

        history__other_type3_x = history_et 
        history__other_type3_y = history__other_type3_z = (history__other_type3)

        history__other_type4_x = history_et 
        history__other_type4_y = history__other_type4_z = (history__other_type4)

        
        history_et[-1] = 0.0
        ##Vocals
        history__vocals_type1[-1] = (history__vocals_type1[-2] + (0.00 +(0.08+dt*10)*intensity*( _track4_beat_rms)*((_track1_beat_rms +2.0*_track4_beat_rms*_track4_beat_rms)/ ((1+_track3_beat_rms + _track2_beat_rms))))*(_track4_beat_types[0]  - history__vocals_type1[-2]))  

        history__vocals_type2[-1] = (history__vocals_type2[-2] + (0.00 +(0.08+dt*10)*intensity*( _track4_beat_rms)*((_track1_beat_rms +2.0*_track4_beat_rms*_track4_beat_rms)/ ((1+_track3_beat_rms + _track2_beat_rms))))*(_track4_beat_types[1]  - history__vocals_type2[-2]))  

        history__vocals_type3[-1] = (history__vocals_type3[-2] + (0.00 +(0.08+dt*10)*intensity*( _track4_beat_rms)*((_track1_beat_rms +2.0*_track4_beat_rms*_track4_beat_rms)/ ((1+_track3_beat_rms + _track2_beat_rms))))*(_track4_beat_types[2]  - history__vocals_type3[-2]))  

        history__vocals_type4[-1] = (history__vocals_type4[-2] + (0.00 +(0.08+dt*10)*intensity*( _track4_beat_rms)*((_track1_beat_rms +2.0*_track4_beat_rms*_track4_beat_rms)/ ((1+_track3_beat_rms + _track2_beat_rms))))*(_track4_beat_types[3]  - history__vocals_type4[-2]))  

        history__vocals_type1_x = history_et 
        history__vocals_type1_y = history__vocals_type1_z = (history__vocals_type1)

        history__vocals_type2_x = history_et 
        history__vocals_type2_y = history__vocals_type2_z = (history__vocals_type2)

        history__vocals_type3_x = history_et 
        history__vocals_type3_y = history__vocals_type3_z = (history__vocals_type3)

        history__vocals_type4_x = history_et 
        history__vocals_type4_y = history__vocals_type4_z = (history__vocals_type4)


        
        



@c.connect
def on_frame(i):
    global draw_elapsed, read_elapsed,  dt, pos, history__drums_type1, history__drums_type1_x, history__drums_type1_y, history__drums_type2, history__drums_type2_x, history__drums_type2_y,  history__drums_type3
    global start_time, delay, dt, span, t0, t1, history_et, history_at
    global _track1_beat_rms, measurements, beat_offset, draw_fps, draw_mode

    if(i == 0):
            
            t0 = start_time = time.perf_counter()
            
            aio.play(play_data, 44100, blocking = False)
            #time.sleep(wait_time)
            delay = time.perf_counter() - start_time - wait_time
            print("delay:")
            print(delay)
            
            


    #main_read()
    main_write()
    #print(measurements)
    t1 = time.perf_counter()
    dt = t1 - t0
    t0 = t1

    span += dt
    history_et -= dt
    history_at -= dt

    read_elapsed += dt 
    draw_elapsed += dt

    draw_fps = draw_fps + (256-draw_fps)*0.9  
    
    if(intensity > 0.5*intensity_max or intensity < -0.5*intensity_min):
        #draw_mode = 0
        draw_fps = 1
    #elif(intensity < intensity_cutoff - 0.01*(intensity_max-intensity_min)):
    #    draw_mode = 1
    #else:
    #    draw_mode = 2
    
    #if(intensity > intensity_cutoff + 0.5*(intensity_max-intensity_min) or intensity < intensity_cutoff - 0.5*(intensity_max-intensity_min)):
    #    #draw_mode = 1
    #    draw_fps = 1
    #    #draw_mode = 1
    ##else:
    #    #draw_mode = 1
        
    if(draw_elapsed > 1/(1+draw_fps)):
        draw()  
        draw_elapsed = 0
        
##GUI
#gui = c.gui("GUI")
#
#b = gui.control("button", "Mode")
#@b.connect
#def on_change(value):
#    global draw_mode
#    # We update the marker positions.
#    #draw_mode = (draw_mode + 1)%2

run()
c.close()