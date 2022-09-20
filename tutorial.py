import madmom
import numpy as np
from pathlib import Path
from os import path
import sys
import time
from utilities import *

extensions = ["mp3", "wav", "ogg", "flac", "m4a"]  # we will look for all those file types.

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
        
        
        
_track_intensity = np.loadtxt(track_path+'intensity.csv', delimiter=',')
intensity_max = np.max(_track_intensity)
intensity_min = np.min(_track_intensity)
#intensity_cutoff = np.mean(_track_intensity)/(intensity_max - intensity_min) - 0.2*((np.abs(intensity_max) - np.abs(intensity_min))/(intensity_max + intensity_min))*(intensity_max - intensity_min)
intensity_cutoff = np.mean(_track_intensity)*2.5

_track1_rms = np.loadtxt(track_path+'drums_trend.csv', delimiter=',')
_track2_rms = np.loadtxt(track_path+'bass_trend.csv', delimiter=',')
_track3_rms = np.loadtxt(track_path+'other_trend.csv', delimiter=',')
_track4_rms = np.loadtxt(track_path+'vocals_trend.csv', delimiter=',')


_track1_types = np.loadtxt(track_path+'drums_beats_cd_2.csv', delimiter=',')
_track2_types = np.loadtxt(track_path+'bass_beats_cd_2.csv', delimiter=',')
_track3_types = np.loadtxt(track_path+'other_beats_cd_2.csv', delimiter=',')
_track4_types = np.loadtxt(track_path+'vocals_beats_cd_2.csv', delimiter=',')

superflux_v0_norm = np.loadtxt(track_path+"superflux_drums_norm.csv", delimiter=",")
superflux_v1_norm = np.loadtxt(track_path+"superflux_bass_norm.csv", delimiter=",")
superflux_v2_norm = np.loadtxt(track_path+"superflux_other_norm.csv", delimiter=",")
superflux_v3_norm = np.loadtxt(track_path+"superflux_vocals_norm.csv", delimiter=",")

print("0: " + files[0])
print("1: " + files[1])
print("2: " + files[2])
print("3: " + files[3])
#data, samplingrate = aio.load_audio(audio_path+'.mp3')
play_data_2 = madmom.audio.signal.Signal(audio_path, dtype = np.float32,num_channels=2)
play_data = madmom.audio.signal.Signal(audio_path, dtype = np.float32,num_channels=1)
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

_track_fps = play_data.sample_rate

from screeninfo import get_monitors
from datoviz import canvas, run, colormap
import audioio as aio


screen = get_monitors()[0]
aspect_ratio = screen.width / screen.height

c = canvas(show_fps=True, width = screen.width*1.18, height = screen.height*1.18)
ctx = c.gpu().context()
panel = c.scene().panel(controller='camera', depth_test = True)

visual_0 = panel.visual('path', transform=None, depth_test=True)

visual_1 = panel.visual('marker', transform=None, depth_test=True)

visual_2 = panel.visual('text', transform=None, depth_test=True)




global read_fps, draw_fps
read_fps = 256
draw_fps = 4*read_fps
wait_time = 0.0

#timing
global start_time, delay, dt, span, t0, t1, meas_span_a, meas_span_e, read_elapsed, draw_elapsed
start_time = time.perf_counter()
t0 = start_time
t1 = start_time
delay = dt = span = meas_span_a = meas_span_e = read_elapsed = draw_elapsed = 0.0
global delay_multiplier , draw_mode, ready_switch
global offset
offset = 0.15

#_history
global history__v0_type1, history__v0_type1_x, history__v0_type1_y, history__v0_type2, history__v0_type2_x, history__v0_type2_y,  history__v0_type3, history__v0_type3_x, history__v0_type3_y, history__v0_type4, history__v0_type4_x, history__v0_type4_y, history_et
global history_ax, history_ay, history_az, history_at

n = 256
visual_0.data('length', np.array([n, n, n, n, n, n, n, n, n, n, n, n, n, n, n, n])) # n, n, n, n
#                               
x = np.linspace(-1.0, 1.0, n) - 2.0
y = np.zeros(n)
z = x*y

pos = np.c_[x, y, z]  # an (N, 3) array with the coordinates of the path vertices.
pos_a = np.c_[np.ones(1), np.ones(1), np.ones(1)]

history_et = history_at = np.array(pos[:,0])
history_ax = history_ay = history_az = np.array(pos[:,1])



fade = np.arange(n)
zeros = np.zeros(n)

global Lines
Lines = np.arange(16)*1.0 + 1.0


##Yellow
cmap_mycmap_y_1_5 = np.c_[fade,fade, fade*0.01, fade]
ctx.colormap('mycmap_y_1_5', cmap_mycmap_y_1_5.astype(np.uint8))




##Drums 
cmap_mycmap_v0_1_5 = np.c_[fade*0.99, fade*0.01, fade*0.02, fade]
ctx.colormap('mycmap_v0_1_5', cmap_mycmap_v0_1_5.astype(np.uint8))


cmap_mycmap_v0_2_6 = np.c_[fade*0.7, fade*0.01, fade*0.02, fade]
ctx.colormap('mycmap_v0_2_6', cmap_mycmap_v0_2_6.astype(np.uint8))


cmap_mycmap_v0_3_7 = np.c_[fade*0.6, fade*0.01, fade*0.02, fade]
ctx.colormap('mycmap_v0_3_7', cmap_mycmap_v0_3_7.astype(np.uint8))


cmap_mycmap_v0_4_8 = np.c_[fade*0.5, fade*0.01, fade*0.02, fade]
ctx.colormap('mycmap_v0_4_8', cmap_mycmap_v0_4_8.astype(np.uint8))


##Bass 
cmap_mycmap_v1_1_5 = np.c_[fade*0.4, fade*0.3, 0.3*fade, fade]
ctx.colormap('mycmap_v1_1_5', cmap_mycmap_v1_1_5.astype(np.uint8))


cmap_mycmap_v1_2_6 = np.c_[fade*0.35, fade*0.2, 0.2*fade, fade]
ctx.colormap('mycmap_v1_2_6', cmap_mycmap_v1_2_6.astype(np.uint8))


cmap_mycmap_v1_3_7 = np.c_[fade*0.3, fade*0.05, 0.05*fade, fade]
ctx.colormap('mycmap_v1_3_7', cmap_mycmap_v1_3_7.astype(np.uint8))


cmap_mycmap_v1_4_8 = np.c_[fade*0.25, fade*0.05, 0.05*fade, fade]
ctx.colormap('mycmap_v1_4_8', cmap_mycmap_v1_4_8.astype(np.uint8))


##Other 
cmap_mycmap_v2_1_5 = np.c_[fade*0.01, fade*0.1, 0.99*fade, fade]
ctx.colormap('mycmap_v2_1_5', cmap_mycmap_v2_1_5.astype(np.uint8))


cmap_mycmap_v2_2_6 = np.c_[fade*0.01, fade*0.1, 0.8*fade, fade]
ctx.colormap('mycmap_v2_2_6', cmap_mycmap_v2_2_6.astype(np.uint8))


cmap_mycmap_v2_3_7 = np.c_[fade*0.01, fade*0.01, 0.6*fade, fade]
ctx.colormap('mycmap_v2_3_7', cmap_mycmap_v2_3_7.astype(np.uint8))


cmap_mycmap_v2_4_8 = np.c_[fade*0.01, fade*0.01 , 0.5*fade, fade]
ctx.colormap('mycmap_v2_4_8', cmap_mycmap_v2_4_8.astype(np.uint8))


##Vocals 
cmap_mycmap_w_1_5 = np.c_[fade,fade, fade, fade]
ctx.colormap('mycmap_w_1_5', cmap_mycmap_w_1_5.astype(np.uint8))

cmap_mycmap_w_2_6 = np.c_[fade*0.8,fade*0.8, fade*0.8, fade*0.8]
ctx.colormap('mycmap_w_2_6', cmap_mycmap_w_2_6.astype(np.uint8))


#cmap_mycmap_v3_2_6 = np.c_[0.925*fade,0.925*fade, 0.925*fade, fade]
#ctx.colormap('mycmap_v3_2_6', cmap_mycmap_v3_2_6.astype(np.uint8))


#cmap_mycmap_v3_3_7 = np.c_[0.7*fade,0.7*fade, 0.7*fade, fade]
#ctx.colormap('mycmap_v3_3_7', cmap_mycmap_v3_3_7.astype(np.uint8))


#cmap_mycmap_v3_4_8 = np.c_[0.5*fade,0.5*fade, 0.5*fade, fade]
#ctx.colormap('mycmap_v3_4_8', cmap_mycmap_v3_4_8.astype(np.uint8))




N = 4*64



# Add a first column Drums
history__v0_type1 = history__v0_type1_x = history__v0_type1_y = history__v0_type1_z = np.array(pos[:,1])
history__v0_type2 = history__v0_type2_x = history__v0_type2_y = history__v0_type2_z = np.array(pos[:,1])
history__v0_type3 = history__v0_type3_x = history__v0_type3_y = history__v0_type3_z = np.array(pos[:,1])
history__v0_type4 = history__v0_type4_x = history__v0_type4_y = history__v0_type4_z = np.array(pos[:,1])

# Add second column Bass
history__v1_type1 = history__v1_type1_x = history__v1_type1_y = history__v1_type1_z = np.array(pos[:,1])
history__v1_type2 = history__v1_type2_x = history__v1_type2_y = history__v1_type2_z = np.array(pos[:,1])
history__v1_type3 = history__v1_type3_x = history__v1_type3_y = history__v1_type3_z = np.array(pos[:,1])
history__v1_type4 = history__v1_type4_x = history__v1_type4_y = history__v1_type4_z = np.array(pos[:,1])

# Add third column Other
history__v2_type1 = history__v2_type1_x = history__v2_type1_y = history__v2_type1_z = np.array(pos[:,1])
history__v2_type2 = history__v2_type2_x = history__v2_type2_y = history__v2_type2_z = np.array(pos[:,1])
history__v2_type3 = history__v2_type3_x = history__v2_type3_y =  history__v2_type3_z = np.array(pos[:,1])
history__v2_type4 = history__v2_type4_x = history__v2_type4_y = history__v2_type4_z = np.array(pos[:,1])



# Add forth column Vocals
history__v3_type1 = history__v3_type1_x = history__v3_type1_y = history__v3_type1_z = np.array(pos[:,1])
history__v3_type2 = history__v3_type2_x = history__v3_type2_y = history__v3_type2_z = np.array(pos[:,1])
history__v3_type3 = history__v3_type3_x = history__v3_type3_y =  history__v3_type3_z = np.array(pos[:,1])
history__v3_type4 = history__v3_type4_x = history__v3_type4_y =  history__v3_type4_z = np.array(pos[:,1])




delay_multiplier = 0.75
ready_switch = 20.0
draw_mode = 0

print(_track1_rms.shape, _track1_rms_ratio, (data__track1.shape[0]/data__track1.sample_rate))


global intensity, beat_offset,  current_frame, current_time,_track_beat_count, _track1_beat_count,_track_beat_type2, _track1_beat_rms, _track2_beat_count, _track2_beat_rms, _track3_beat_count, _track3_beat_rms
current_frame = current_time = measurements = _track_beat_count = _track1_beat_count = _track_beat_type2= _track2_beat_count = _track3_beat_count = _track4_beat_count =  _track1_beat_rms =  _track2_beat_rms = _track3_beat_rms = _track4_beat_rms = intensity = beat_offset = 0.0



data = np.hstack((track_data, data__track1))
data = np.hstack((data, data__track2))
data = np.hstack((data, data__track3))
data = np.hstack((data, data__track4))




def main_write():
        global statevars, s1, sv, si_vals
        global pos, pos_a, history_ax, history_ay, history_az, history_at, history__v0_type1, history__v0_type1_x, history__v0_type1_y, history__v0_type1_z
        global history__v1_type1, history__v1_type1_x, history__v1_type1_y, history__v1_type1_z
        global history__v2_type1, history__v2_type1_x, history__v2_type1_y, history__v2_type1_z
        global history__v3_type1, history__v3_type1_x, history__v3_type1_y, history__v3_type1_z
        global history__v0_type2, history__v0_type2_x, history__v0_type2_y, history__v0_type2_z, history__v0_type3, history__v0_type3_x, history__v0_type3_y, history__v0_type3_z
        global history__v0_type4, history__v0_type4_x, history__v0_type4_y, history__v0_type4_z, history__v0_type5, history__v0_type5_x, history__v0_type5_y, history__v0_type5_z
        global history__v0_type6, history__v0_type6_x, history__v0_type6_y, history__v0_type6_z, history__v0_type7, history__v0_type7_x, history__v0_type7_y, history__v0_type7_z
        global history__v0_type8, history__v0_type8_x, history__v0_type8_y, history__v0_type8_z
        global history__v1_type2, history__v1_type2_x, history__v1_type2_y, history__v1_type2_z, history__v1_type3, history__v1_type3_x, history__v1_type3_y, history__v1_type3_z
        global history__v1_type4, history__v1_type4_x, history__v1_type4_y, history__v1_type4_z, history__v1_type5, history__v1_type5_x, history__v1_type5_y, history__v1_type5_z
        global history__v1_type6, history__v1_type6_x, history__v1_type6_y, history__v1_type6_z, history__v1_type7, history__v1_type7_x, history__v1_type7_y, history__v1_type7_z
        global history__v1_type8, history__v1_type8_x, history__v1_type8_y, history__v1_type8_z
        global history__v2_type2, history__v2_type2_x, history__v2_type2_y, history__v2_type2_z, history__v2_type3, history__v2_type3_x, history__v2_type3_y, history__v2_type3_z
        global history__v2_type4, history__v2_type4_x, history__v2_type4_y, history__v2_type4_z, history__v2_type5, history__v2_type5_x, history__v2_type5_y, history__v2_type5_z
        global history__v2_type6, history__v2_type6_x, history__v2_type6_y, history__v2_type6_z, history__v2_type7, history__v2_type7_x, history__v2_type7_y, history__v2_type7_z
        global history__v2_type8, history__v2_type8_x, history__v2_type8_y, history__v2_type8_z
        global history__v3_type2, history__v3_type2_x, history__v3_type2_y, history__v3_type2_z, history__v3_type3, history__v3_type3_x, history__v3_type3_y, history__v3_type3_z
        global history__v3_type4, history__v3_type4_x, history__v3_type4_y, history__v3_type4_z, history__v3_type5, history__v3_type5_x, history__v3_type5_y, history__v3_type5_z
        global history__v3_type6, history__v3_type6_x, history__v3_type6_y, history__v3_type6_z, history__v3_type7, history__v3_type7_x, history__v3_type7_y, history__v3_type7_z
        global history__v3_type8, history__v3_type8_x, history__v3_type8_y, history__v3_type8_z
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

        ###type1
        ##v0
        history__v0_type1 = 1.0001*np.roll(history__v0_type1,-1, axis = 0)
        history__v0_type1_x = 1.0001*np.roll(history__v0_type1_x,-1, axis = 0)
        history__v0_type1_y = 1.0001*np.roll(history__v0_type1_y,-1, axis = 0)
        history__v0_type1_z = 1.0001*np.roll(history__v0_type1_z,-1, axis = 0)
        
        ##v1
        history__v1_type1 = 1.0001*np.roll(history__v1_type1,-1, axis = 0)
        history__v1_type1_x = 1.0001*np.roll(history__v1_type1_x,-1, axis = 0)
        history__v1_type1_y = 1.0001*np.roll(history__v1_type1_y,-1, axis = 0)
        history__v1_type1_z = 1.0001*np.roll(history__v1_type1_z,-1, axis = 0)
        
        ##v2
        history__v2_type1 = 1.0001*np.roll(history__v2_type1,-1, axis = 0)
        history__v2_type1_x = 1.0001*np.roll(history__v2_type1_x,-1, axis = 0)
        history__v2_type1_y = 1.0001*np.roll(history__v2_type1_y,-1, axis = 0)
        history__v2_type1_z = 1.0001*np.roll(history__v2_type1_z,-1, axis = 0)
        
        ##v3
        history__v3_type1 = 1.0001*np.roll(history__v3_type1,-1, axis = 0)
        history__v3_type1_x = 1.0001*np.roll(history__v3_type1_x,-1, axis = 0)
        history__v3_type1_y = 1.0001*np.roll(history__v3_type1_y,-1, axis = 0)
        history__v3_type1_z = 1.0001*np.roll(history__v3_type1_z,-1, axis = 0)
        
        history__v0_type1_y[-1] = 0
        history__v1_type1_y[-1] = 0
        history__v2_type1_y[-1] = 0
        history__v3_type1_y[-1] = 0
        
        
        ###type2
        ##v0
        history__v0_type2 = 1.0001*np.roll(history__v0_type2,-1, axis = 0)
        history__v0_type2_x = 1.0001*np.roll(history__v0_type2_x,-1, axis = 0)
        history__v0_type2_y = 1.0001*np.roll(history__v0_type2_y,-1, axis = 0)
        history__v0_type2_z = 1.0001*np.roll(history__v0_type2_z,-1, axis = 0)
        
        ##v1
        history__v1_type2 = 1.0001*np.roll(history__v1_type2,-1, axis = 0)
        history__v1_type2_x = 1.0001*np.roll(history__v1_type2_x,-1, axis = 0)
        history__v1_type2_y = 1.0001*np.roll(history__v1_type2_y,-1, axis = 0)
        history__v1_type2_z = 1.0001*np.roll(history__v1_type2_z,-1, axis = 0)
        
        ##v2
        history__v2_type2 = 1.0001*np.roll(history__v2_type2,-1, axis = 0)
        history__v2_type2_x = 1.0001*np.roll(history__v2_type2_x,-1, axis = 0)
        history__v2_type2_y = 1.0001*np.roll(history__v2_type2_y,-1, axis = 0)
        history__v2_type2_z = 1.0001*np.roll(history__v2_type2_z,-1, axis = 0)
        
        ##v3
        history__v3_type2 = 1.0001*np.roll(history__v3_type2,-1, axis = 0)
        history__v3_type2_x = 1.0001*np.roll(history__v3_type2_x,-1, axis = 0)
        history__v3_type2_y = 1.0001*np.roll(history__v3_type2_y,-1, axis = 0)
        history__v3_type2_z = 1.0001*np.roll(history__v3_type2_z,-1, axis = 0)
        
        history__v0_type2_y[-1] = 0
        history__v1_type2_y[-1] = 0
        history__v2_type2_y[-1] = 0
        history__v3_type2_y[-1] = 0
        
        
        ###type3
        ##v0
        history__v0_type3 = 1.0001*np.roll(history__v0_type3,-1, axis = 0)
        history__v0_type3_x = 1.0001*np.roll(history__v0_type3_x,-1, axis = 0)
        history__v0_type3_y = 1.0001*np.roll(history__v0_type3_y,-1, axis = 0)
        history__v0_type3_z = 1.0001*np.roll(history__v0_type3_z,-1, axis = 0)
        
        ##v1
        history__v1_type3 = 1.0001*np.roll(history__v1_type3,-1, axis = 0)
        history__v1_type3_x = 1.0001*np.roll(history__v1_type3_x,-1, axis = 0)
        history__v1_type3_y = 1.0001*np.roll(history__v1_type3_y,-1, axis = 0)
        history__v1_type3_z = 1.0001*np.roll(history__v1_type3_z,-1, axis = 0)
        
        ##v2
        history__v2_type3 = 1.0001*np.roll(history__v2_type3,-1, axis = 0)
        history__v2_type3_x = 1.0001*np.roll(history__v2_type3_x,-1, axis = 0)
        history__v2_type3_y = 1.0001*np.roll(history__v2_type3_y,-1, axis = 0)
        history__v2_type3_z = 1.0001*np.roll(history__v2_type3_z,-1, axis = 0)
        
        ##v3
        history__v3_type3 = 1.0001*np.roll(history__v3_type3,-1, axis = 0)
        history__v3_type3_x = 1.0001*np.roll(history__v3_type3_x,-1, axis = 0)
        history__v3_type3_y = 1.0001*np.roll(history__v3_type3_y,-1, axis = 0)
        history__v3_type3_z = 1.0001*np.roll(history__v3_type3_z,-1, axis = 0)
        
        history__v0_type3_y[-1] = 0
        history__v1_type3_y[-1] = 0
        history__v2_type3_y[-1] = 0
        history__v3_type3_y[-1] = 0
        
        ###type4
        ##v0
        history__v0_type4 = 1.0001*np.roll(history__v0_type4,-1, axis = 0)
        history__v0_type4_x = 1.0001*np.roll(history__v0_type4_x,-1, axis = 0)
        history__v0_type4_y = 1.0001*np.roll(history__v0_type4_y,-1, axis = 0)
        history__v0_type4_z = 1.0001*np.roll(history__v0_type4_z,-1, axis = 0)
        
        ##v1
        history__v1_type4 = 1.0001*np.roll(history__v1_type4,-1, axis = 0)
        history__v1_type4_x = 1.0001*np.roll(history__v1_type4_x,-1, axis = 0)
        history__v1_type4_y = 1.0001*np.roll(history__v1_type4_y,-1, axis = 0)
        history__v1_type4_z = 1.0001*np.roll(history__v1_type4_z,-1, axis = 0)
        
        ##v2
        history__v2_type4 = 1.0001*np.roll(history__v2_type4,-1, axis = 0)
        history__v2_type4_x = 1.0001*np.roll(history__v2_type4_x,-1, axis = 0)
        history__v2_type4_y = 1.0001*np.roll(history__v2_type4_y,-1, axis = 0)
        history__v2_type4_z = 1.0001*np.roll(history__v2_type4_z,-1, axis = 0)
        
        ##v3
        history__v3_type4 = 1.0001*np.roll(history__v3_type4,-1, axis = 0)
        history__v3_type4_x = 1.0001*np.roll(history__v3_type4_x,-1, axis = 0)
        history__v3_type4_y = 1.0001*np.roll(history__v3_type4_y,-1, axis = 0)
        history__v3_type4_z = 1.0001*np.roll(history__v3_type4_z,-1, axis = 0)
        
        history__v0_type4_y[-1] = 0
        history__v1_type4_y[-1] = 0
        history__v2_type4_y[-1] = 0
        history__v3_type4_y[-1] = 0
        

        history_et = 1.0001*np.roll(history_et,-1, axis = 0)
        span = last_span
        
        intensity_index = int(((span - delay*delay_multiplier))*_track4_fps)
        if(intensity_index < _track_intensity.shape[0]):
            intensity = _track_intensity[intensity_index]
        else:
            intensity = _track_intensity[-1]
            
            
        #beat_offset = dt*(intensity - (intensity_max + intensity_min)/2)/2
        beat_offset = 0
        
        _track_beat_count = int(((span - delay*delay_multiplier))*_track_fps)
        _track1_beat_count = int(((span - delay*delay_multiplier))*_track1_rms_ratio)
        _track2_beat_count = int(((span - delay*delay_multiplier))*_track2_rms_ratio)
        _track3_beat_count = int(((span - delay*delay_multiplier))*_track3_rms_ratio)
        _track4_beat_count = int(((span - delay*delay_multiplier))*_track4_rms_ratio)
        _track1_type_count = int(((span - delay*delay_multiplier))*_track1_fps)
        _track2_type_count = int(((span - delay*delay_multiplier))*_track2_fps)
        _track3_type_count = int(((span - delay*delay_multiplier))*_track3_fps)
        _track4_type_count = int(((span - delay*delay_multiplier))*_track4_fps)
        
        if(_track1_beat_count < _track1_rms.shape[0]-1):
            _track1_beat_rms = _track1_rms[_track1_beat_count]
        if(_track2_beat_count < _track2_rms.shape[0]-1):
            _track2_beat_rms = _track2_rms[_track2_beat_count]
        if(_track3_beat_count < _track3_rms.shape[0]-1):
            _track3_beat_rms = _track3_rms[_track3_beat_count]
        if(_track4_beat_count < _track4_rms.shape[0]-1):
            _track4_beat_rms = _track4_rms[_track4_beat_count]
            
        
        
        
        if(_track1_beat_count < _track1_rms.shape[0]-1):
            _track1_beat_count = int((span - delay*delay_multiplier)*_track1_rms_ratio)
            _track1_type_count = int((span - delay*delay_multiplier)*_track1_fps)
            _track1_beat_types = _track1_types[int(_track1_type_count - 1 - beat_offset),:]*((1.0/draw_fps)-draw_elapsed)*draw_fps + _track1_types[int(_track1_type_count- beat_offset),:]*draw_elapsed*draw_fps

        if(_track2_beat_count < _track2_rms.shape[0]-1):
            _track2_beat_count = int((span - delay*delay_multiplier)*_track2_rms_ratio)
            _track2_type_count = int((span - delay*delay_multiplier)*_track2_fps)
            _track2_beat_types = _track2_types[int(_track2_type_count - 1 - beat_offset),:]*((1.0/draw_fps)-draw_elapsed)*draw_fps + _track2_types[int(_track2_type_count- beat_offset),:]*draw_elapsed*draw_fps

        if(_track3_beat_count < _track3_rms.shape[0]-1):
            _track3_beat_count = int((span - delay*delay_multiplier)*_track3_rms_ratio)
            _track3_type_count = int((span - delay*delay_multiplier)*_track3_fps)
            _track3_beat_types = _track3_types[int(_track3_type_count - 1 - beat_offset),:]*((1.0/draw_fps)-draw_elapsed)*draw_fps + _track3_types[int(_track3_type_count- beat_offset),:]*draw_elapsed*draw_fps

        if(_track4_beat_count < _track4_rms.shape[0]-1):
            _track4_beat_count = int((span - delay*delay_multiplier)*_track4_rms_ratio)
            _track4_type_count = int((span - delay*delay_multiplier)*_track4_fps)
            _track4_beat_types = _track4_types[int(_track4_type_count - 1 - beat_offset),:]*((1.0/draw_fps)-draw_elapsed)*draw_fps + _track4_types[int(_track4_type_count- beat_offset),:]*draw_elapsed*draw_fps
                
                
                
        if(draw_mode == 0):
            if(_track_beat_count < play_data.shape[0]-256):
                history__v0_type1_y[-1] = play_data[_track_beat_count]
            ###Update Lines
            for i in range (Lines.shape[0]-1):
                force_total = 0
                if(i == 0):
                    force_total += history__v0_type1_y[-1]
                    force_total += np.abs(Lines[i+1] - Lines[i]) - 1
                elif(i == Lines.shape[0]-1):
                    force_total +=  1 - np.abs(Lines[i] - Lines[i-1])
                else:
                    force_total += np.abs(Lines[i+1] - Lines[i]) - 1
                    force_total +=  1 - np.abs(Lines[i] - Lines[i-1])
                force_total += 0.001*((i+1) - Lines[i])**(3)
                Lines[i] += 0.95*force_total
            
        if(draw_mode == 1):
            if(_track_beat_count < play_data.shape[0]-256):
                history__v0_type1_y[-1] = play_data[_track_beat_count]
                
        if(draw_mode == 2):
            if(_track_beat_count < play_data.shape[0]-256):
                history__v0_type1_y[-1] = data__track1[_track_beat_count]
                history__v1_type1_y[-1] = data__track2[_track_beat_count]
                history__v2_type1_y[-1] = data__track3[_track_beat_count]
                history__v3_type1_y[-1] = data__track4[_track_beat_count]
                
        if(draw_mode == 3):
            if(_track1_beat_count < _track1_rms.shape[0]-1):
                history__v0_type1_y[-1] = _track1_beat_rms
                history__v1_type1_y[-1] = _track2_beat_rms
                history__v2_type1_y[-1] = _track3_beat_rms
                history__v3_type1_y[-1] = _track4_beat_rms
                
        if(draw_mode == 4):
            if(_track1_beat_count < superflux_v0_norm.shape[0]-1):
                history__v0_type1_y[-1] = superflux_v0_norm[_track1_beat_count]
                history__v1_type1_y[-1] = superflux_v1_norm[_track2_beat_count]
                history__v2_type1_y[-1] = superflux_v2_norm[_track3_beat_count]
                history__v3_type1_y[-1] = superflux_v3_norm[_track4_beat_count]
                
        if(draw_mode == 5 or draw_mode == 6 or draw_mode == 7):
            if(_track1_beat_count < superflux_v0_norm.shape[0]-1):
                ###From here
                ##Drums
                history__v0_type1[-1] = (history__v0_type1[-2] + (0.2)*(_track1_beat_types[0]  - history__v0_type1[-2])) 
                history__v0_type2[-1] = (history__v0_type2[-2] + (0.2)*(_track1_beat_types[1]  - history__v0_type2[-2])) 
                history__v0_type3[-1] = (history__v0_type3[-2] + (0.2)*(_track1_beat_types[2]  - history__v0_type3[-2])) 
                history__v0_type4[-1] = (history__v0_type4[-2] + (0.2)*(_track1_beat_types[3]  - history__v0_type4[-2])) 
                
                if(draw_mode > 5):
                    history__v0_type1[-1] *= _track1_beat_rms
                    history__v0_type2[-1] *= _track1_beat_rms
                    history__v0_type3[-1] *= _track1_beat_rms
                    history__v0_type4[-1] *= _track1_beat_rms
                    
        
                history__v0_type1_x = history_et 
                history__v0_type1_y = (history__v0_type1) 
        
                history__v0_type2_x = history_et 
                history__v0_type2_y = (history__v0_type2) 
        
                history__v0_type3_x = history_et 
                history__v0_type3_y = (history__v0_type3)
        
                history__v0_type4_x = history_et 
                history__v0_type4_y = (history__v0_type4)
        
        
                history_et[-1] = 0
                ##Bass
                history__v1_type1[-1] = (history__v1_type1[-2] + (0.2)*(_track2_beat_types[0])  - history__v1_type1[-2]) 
                history__v1_type2[-1] = (history__v1_type2[-2] + (0.2)*(_track2_beat_types[1])  - history__v1_type2[-2]) 
                history__v1_type3[-1] = (history__v1_type3[-2] + (0.2)*(_track2_beat_types[2])  - history__v1_type3[-2]) 
                history__v1_type4[-1] = (history__v1_type4[-2] + (0.2)*(_track2_beat_types[3])  - history__v1_type4[-2]) 
                
                if(draw_mode > 5):
                    history__v1_type1[-1] *= _track2_beat_rms
                    history__v1_type2[-1] *= _track2_beat_rms
                    history__v1_type3[-1] *= _track2_beat_rms
                    history__v1_type4[-1] *= _track2_beat_rms
        
        
                history__v1_type1_x = history_et 
                history__v1_type1_y = (history__v1_type1)
        
                history__v1_type2_x = history_et 
                history__v1_type2_y = (history__v1_type2)
        
                history__v1_type3_x = history_et 
                history__v1_type3_y = (history__v1_type3)
        
                history__v1_type4_x = history_et 
                history__v1_type4_y = (history__v1_type4)
        
        
                history_et[-1] = 0
                ##Other
                history__v2_type1[-1] = (history__v2_type1[-2] + (0.2)*(_track3_beat_types[0]  - history__v2_type1[-2]))  
                history__v2_type2[-1] = (history__v2_type2[-2] + (0.2)*(_track3_beat_types[1]  - history__v2_type2[-2]))    
                history__v2_type3[-1] = (history__v2_type3[-2] + (0.2)*(_track3_beat_types[2]  - history__v2_type3[-2]))  
                history__v2_type4[-1] = (history__v2_type4[-2] + (0.2)*(_track3_beat_types[3]  - history__v2_type4[-2]))  
                
                if(draw_mode > 5):
                    history__v2_type1[-1] *= _track3_beat_rms
                    history__v2_type2[-1] *= _track3_beat_rms
                    history__v2_type3[-1] *= _track3_beat_rms
                    history__v2_type4[-1] *= _track3_beat_rms
        
                history__v2_type1_x = history_et 
                history__v2_type1_y = (history__v2_type1) 
        
                history__v2_type2_x = history_et 
                history__v2_type2_y = (history__v2_type2) 
        
                history__v2_type3_x = history_et 
                history__v2_type3_y = (history__v2_type3)
        
                history__v2_type4_x = history_et 
                history__v2_type4_y = (history__v2_type4)
        
                
                history_et[-1] = 0
                ##Vocals
                history__v3_type1[-1] = (history__v3_type1[-2] + (0.2)*(_track4_beat_types[0]  - history__v3_type1[-2]))  
                history__v3_type2[-1] = (history__v3_type2[-2] + (0.2)*(_track4_beat_types[1]  - history__v3_type2[-2]))  
                history__v3_type3[-1] = (history__v3_type3[-2] + (0.2)*(_track4_beat_types[2]  - history__v3_type3[-2]))  
                history__v3_type4[-1] = (history__v3_type4[-2] + (0.2)*(_track4_beat_types[3]  - history__v3_type4[-2]))  
                
                if(draw_mode > 5):
                    history__v3_type1[-1] *= _track4_beat_rms
                    history__v3_type2[-1] *= _track4_beat_rms
                    history__v3_type3[-1] *= _track4_beat_rms
                    history__v3_type4[-1] *= _track4_beat_rms
        
                history__v3_type1_x = history_et 
                history__v3_type1_y = (history__v3_type1)
        
                history__v3_type2_x = history_et 
                history__v3_type2_y = (history__v3_type2)
        
                history__v3_type3_x = history_et 
                history__v3_type3_y = (history__v3_type3)
        
                history__v3_type4_x = history_et 
                history__v3_type4_y = (history__v3_type4)
                
        if(draw_mode == 6 or draw_mode == 7):  ##Mode 3 x Mode 5 i.e. Cluster x RMs
            ##Drums
            history__v0_type1[-1] = (history__v0_type1[-2] + (0.00 +(0.04+dt*10)*intensity*( _track1_beat_rms)*((2.0*_track1_beat_rms*_track1_beat_rms +_track4_beat_rms)/ ((1+_track3_beat_rms + _track2_beat_rms))))*(_track1_beat_types[0]  - history__v0_type1[-2])) 
            history__v0_type2[-1] = (history__v0_type2[-2] + (0.00 +(0.04+dt*10)*intensity*( _track1_beat_rms)*((2.0*_track1_beat_rms*_track1_beat_rms +_track4_beat_rms)/ ((1+_track3_beat_rms + _track2_beat_rms))))*(_track1_beat_types[1]  - history__v0_type2[-2])) 
            history__v0_type3[-1] = (history__v0_type3[-2] + (0.00 +(0.04+dt*10)*intensity*( _track1_beat_rms)*((2.0*_track1_beat_rms*_track1_beat_rms +_track4_beat_rms)/ ((1+_track3_beat_rms + _track2_beat_rms))))*(_track1_beat_types[2]  - history__v0_type3[-2])) 
            history__v0_type4[-1] = (history__v0_type4[-2] + (0.00 +(0.04+dt*10)*intensity*( _track1_beat_rms)*((2.0*_track1_beat_rms*_track1_beat_rms +_track4_beat_rms)/ ((1+_track3_beat_rms + _track2_beat_rms))))*(_track1_beat_types[3]  - history__v0_type4[-2])) 
            
            ##Bass
            history__v1_type1[-1] = (history__v1_type1[-2] + (0.00 +(0.04+dt*10)*intensity*( _track2_beat_rms)*((_track3_beat_rms + 2.0*_track2_beat_rms*_track2_beat_rms)/ ((1+_track1_beat_rms +_track4_beat_rms))))*(_track2_beat_types[0])  - history__v1_type1[-2]) 
            history__v1_type2[-1] = (history__v1_type2[-2] + (0.00 +(0.04+dt*10)*intensity*( _track2_beat_rms)*((_track3_beat_rms + 2.0*_track2_beat_rms*_track2_beat_rms)/ ((1+_track1_beat_rms +_track4_beat_rms))))*(_track2_beat_types[1])  - history__v1_type2[-2]) 
            history__v1_type3[-1] = (history__v1_type3[-2] + (0.00 +(0.04+dt*10)*intensity*( _track2_beat_rms)*((_track3_beat_rms + 2.0*_track2_beat_rms*_track2_beat_rms)/ ((1+_track1_beat_rms +_track4_beat_rms))))*(_track2_beat_types[2])  - history__v1_type3[-2]) 
            history__v1_type4[-1] = (history__v1_type4[-2] + (0.00 +(0.04+dt*10)*intensity*( _track2_beat_rms)*((_track3_beat_rms + 2.0*_track2_beat_rms*_track2_beat_rms)/ ((1+_track1_beat_rms +_track4_beat_rms))))*(_track2_beat_types[3])  - history__v1_type4[-2]) 
    
    
            ##Other
            history__v2_type1[-1] = (history__v2_type1[-2] + (0.00 +(0.04+dt*10)*intensity*( _track3_beat_rms)*((2.0*_track3_beat_rms*_track3_beat_rms + _track2_beat_rms)/ ((1+_track1_beat_rms +_track4_beat_rms))))*(_track3_beat_types[0]  - history__v2_type1[-2]))  
            history__v2_type2[-1] = (history__v2_type2[-2] + (0.00 +(0.04+dt*10)*intensity*( _track3_beat_rms)*((2.0*_track3_beat_rms*_track3_beat_rms + _track2_beat_rms)/ ((1+_track1_beat_rms +_track4_beat_rms))))*(_track3_beat_types[1]  - history__v2_type2[-2]))    
            history__v2_type3[-1] = (history__v2_type3[-2] + (0.00 +(0.04+dt*10)*intensity*( _track3_beat_rms)*((2.0*_track3_beat_rms*_track3_beat_rms + _track2_beat_rms)/ ((1+_track1_beat_rms +_track4_beat_rms))))*(_track3_beat_types[2]  - history__v2_type3[-2]))  
            history__v2_type4[-1] = (history__v2_type4[-2] + (0.00 +(0.04+dt*10)*intensity*( _track3_beat_rms)*((2.0*_track3_beat_rms*_track3_beat_rms + _track2_beat_rms)/ ((1+_track1_beat_rms +_track4_beat_rms))))*(_track3_beat_types[3]  - history__v2_type4[-2]))  
    
            ##Vocals
            history__v3_type1[-1] = (history__v3_type1[-2] + (0.00 +(0.04+dt*10)*intensity*( _track4_beat_rms)*((_track1_beat_rms +2.0*_track4_beat_rms*_track4_beat_rms)/ ((1+_track3_beat_rms + _track2_beat_rms))))*(_track4_beat_types[0]  - history__v3_type1[-2]))  
            history__v3_type2[-1] = (history__v3_type2[-2] + (0.00 +(0.04+dt*10)*intensity*( _track4_beat_rms)*((_track1_beat_rms +2.0*_track4_beat_rms*_track4_beat_rms)/ ((1+_track3_beat_rms + _track2_beat_rms))))*(_track4_beat_types[1]  - history__v3_type2[-2]))  
            history__v3_type3[-1] = (history__v3_type3[-2] + (0.00 +(0.04+dt*10)*intensity*( _track4_beat_rms)*((_track1_beat_rms +2.0*_track4_beat_rms*_track4_beat_rms)/ ((1+_track3_beat_rms + _track2_beat_rms))))*(_track4_beat_types[2]  - history__v3_type3[-2]))  
            history__v3_type4[-1] = (history__v3_type4[-2] + (0.00 +(0.04+dt*10)*intensity*( _track4_beat_rms)*((_track1_beat_rms +2.0*_track4_beat_rms*_track4_beat_rms)/ ((1+_track3_beat_rms + _track2_beat_rms))))*(_track4_beat_types[3]  - history__v3_type4[-2]))  




                
        
        history_et[-1] = 0.0
        
        history__v0_type1_x = history__v1_type1_x = history__v2_type1_x = history__v3_type1_x = history_et 
        
        #print(history__v0_type1_y)
        
        
        #print(Lines)
        



def draw():
    global _track_beat_count, _track1_beat_count, _track_beat_type2, _track2_beat_count, _track3_beat_count, _track4_beat_count, current_frame, current_time, _track1_beat_rms, _track2_beat_rms, _track3_beat_rms, _track4_beat_rms, Lines
    global  _track1_beat_types, _track2_beat_types, _track3_beat_types, _track4_beat_types, ready_switch
    ###Lines
    
    #pos[:,2] = aspect_ratio*history__v0_type1_z
    
    pos[:,0] = history__v0_type1_x*10.0 + 2.0
    
    if(draw_mode == 0):
        pos[:,0] = 1+0.5*history__v0_type1_y[-1]
        pos[:,1] = 2*(0.85+2*history__v0_type1_x*aspect_ratio)
        color = colormap(np.linspace(0, 1, n), vmin=0, vmax=1, cmap='mycmap_y_1_5')
        draw_Path(visual_0, pos, 8, color, 0)
        
        for i in range(Lines.shape[0]-1):
            pos[:,0] = 0.5 - 0.2*Lines[i]
            draw_Path(visual_0, pos, 8, color, 1)
        
    elif(draw_mode == 1):
        pos[:,0] = history__v0_type1_x*20.0 + 2.0
        pos[:,1] = aspect_ratio*history__v0_type1_y*0.25
        color = colormap(np.linspace(0, 1, n), vmin=0, vmax=1, cmap='mycmap_y_1_5')
        draw_Path(visual_0, pos, 8, color, 0)
    elif(draw_mode == 5):
        pass
    else:
        pos[:,1] = aspect_ratio*history__v0_type1_y*0.125 + 1.2
        color = colormap(np.linspace(0, 1, n), vmin=0, vmax=1, cmap='mycmap_v0_1_5')
        draw_Path(visual_0, pos, 8, color, 0)
    #color[:,-1] = np.max([32, color[:,-1].all()]) + 256*np.abs(history__v0_type1_y)
    
    
    
    pos[:,0] = history__v1_type1_x*10.0 + 2.0
    
    if(draw_mode == 0):
        pos[:,1] = aspect_ratio*history__v1_type1_y*0.25 -5.0
    elif(draw_mode == 1):
        pos[:,1] = aspect_ratio*history__v1_type1_y*0.25 -5.0
        draw_Path(visual_0, pos, 8, color, 1)
    elif(draw_mode == 5):
        pass
    else:
        pos[:,1] = pos[:,1] = aspect_ratio*history__v1_type1_y*0.125 + 0.4
        color = colormap(np.linspace(0, 1, n), vmin=0, vmax=1, cmap='mycmap_v1_1_5')
        draw_Path(visual_0, pos, 8, color, 1)
    #color[:,-1] = np.max([32, color[:,-1].all()]) + 256*np.abs(history__v1_type1_y)
    
    
    
    pos[:,0] = history__v2_type1_x*10.0 + 2.0
    
    if(draw_mode == 0):
        pos[:,1] = aspect_ratio*history__v2_type1_y*0.25 -5.0
    elif(draw_mode == 1):
        pos[:,1] = aspect_ratio*history__v2_type1_y*0.25 -5.0
        draw_Path(visual_0, pos, 8, color, 1)
    elif(draw_mode == 5):
        pass
    else:
        pos[:,1] = pos[:,1] = aspect_ratio*history__v2_type1_y*0.125 - 0.4
        color = colormap(np.linspace(0, 1, n), vmin=0, vmax=1, cmap='mycmap_v2_1_5')
        draw_Path(visual_0, pos, 8, color, 1)
    #color[:,-1] = np.max([32, color[:,-1].all()]) + 256*np.abs(history__v2_type1_y)
    
    
    
    pos[:,0] = history__v3_type1_x*10.0 + 2.0
    
    if(draw_mode == 0):
        pos[:,1] = aspect_ratio*history__v3_type1_y*0.25 -5.0
    elif(draw_mode == 1):
        pos[:,1] = aspect_ratio*history__v3_type1_y*0.25 -5.0
        draw_Path(visual_0, pos, 8, color, 1)
        
        for i in range(Lines.shape[0]-4):
            pos[:,0] = 0.5 - 0.2*Lines[i]
            pos[:,1] = -5.0
            draw_Path(visual_0, pos, 8, color, 1)
    elif(draw_mode == 5 or draw_mode == 6 or draw_mode == 7):
        multiplier_temp = 0.5
        ##Drums
        pos[:,0] = history__v0_type1_x*10.0
        pos[:,1] = -1.5+multiplier_temp*aspect_ratio*history__v0_type1_y/np.sqrt(1e-6+np.abs(history__v0_type1_y))
        pos[:,2] = aspect_ratio*history__v0_type1_z
        color = colormap(np.linspace(0, 1, n), vmin=0, vmax=1, cmap='mycmap_v0_1_5')
        draw_Path(visual_0, pos, 8, color, 0)
    
        pos[:,0] = history__v0_type2_x*10.0
        pos[:,1] = -1.5+multiplier_temp*aspect_ratio*history__v0_type2_y/np.sqrt(1e-6+np.abs(history__v0_type2_y))
        pos[:,2] = aspect_ratio*history__v0_type2_z
        color = colormap(np.linspace(0, 1, n), vmin=0, vmax=1, cmap='mycmap_v0_2_6')
        draw_Path(visual_0, pos, 8, color, 1)
        
    
        pos[:,0] = history__v0_type3_x*10.0
        pos[:,1] = -1.5+multiplier_temp*aspect_ratio*history__v0_type3_y/np.sqrt(1e-6+np.abs(history__v0_type3_y))
        pos[:,2] = aspect_ratio*history__v0_type3_z
        color = colormap(np.linspace(0, 1, n), vmin=0, vmax=1, cmap='mycmap_v0_3_7')
        draw_Path(visual_0, pos, 8, color, 1)
    
        pos[:,0] = history__v0_type4_x*10.0
        pos[:,1] = -1.5+multiplier_temp*aspect_ratio*history__v0_type4_y/np.sqrt(1e-6+np.abs(history__v0_type4_y))
        pos[:,2] = aspect_ratio*history__v0_type4_z
        color = colormap(np.linspace(0, 1, n), vmin=0, vmax=1, cmap='mycmap_v0_4_8')
        draw_Path(visual_0, pos, 8, color, 1)
        
        
        ##Bass
        pos[:,0] = history__v1_type1_x*10.0
        pos[:,1] = -1.5+multiplier_temp*aspect_ratio*history__v1_type1_y/np.sqrt(1e-6+np.abs(history__v1_type1_y))
        pos[:,2] = aspect_ratio*history__v1_type1_z
        color = colormap(np.linspace(0, 1, n), vmin=0, vmax=1, cmap='mycmap_v1_1_5')
        draw_Path(visual_0, pos, 8, color, 1)
    
        pos[:,0] = history__v1_type2_x*10.0
        pos[:,1] = -1.5+multiplier_temp*aspect_ratio*history__v1_type2_y/np.sqrt(1e-6+np.abs(history__v1_type2_y))
        pos[:,2] = aspect_ratio*history__v1_type2_z
        color = colormap(np.linspace(0, 1, n), vmin=0, vmax=1, cmap='mycmap_v1_2_6')
        draw_Path(visual_0, pos, 8, color, 1)
        
    
        pos[:,0] = history__v1_type3_x*10.0
        pos[:,1] = -1.5+multiplier_temp*aspect_ratio*history__v1_type3_y/np.sqrt(1e-6+np.abs(history__v1_type3_y))
        pos[:,2] = aspect_ratio*history__v1_type3_z
        color = colormap(np.linspace(0, 1, n), vmin=0, vmax=1, cmap='mycmap_v1_3_7')
        draw_Path(visual_0, pos, 8, color, 1)
    
        pos[:,0] = history__v1_type4_x*10.0
        pos[:,1] = -1.5+multiplier_temp*aspect_ratio*history__v1_type4_y/np.sqrt(1e-6+np.abs(history__v1_type4_y))
        pos[:,2] = aspect_ratio*history__v1_type4_z
        color = colormap(np.linspace(0, 1, n), vmin=0, vmax=1, cmap='mycmap_v1_4_8')
        draw_Path(visual_0, pos, 8, color, 1)
    
        
        ##Other
        pos[:,0] = history__v2_type1_x*10.0
        pos[:,1] = -1.5+multiplier_temp*aspect_ratio*history__v2_type1_y/np.sqrt(1e-6+np.abs(history__v2_type1_y))
        pos[:,2] = aspect_ratio*history__v2_type1_z
        color = colormap(np.linspace(0, 1, n), vmin=0, vmax=1, cmap='mycmap_v2_1_5')
        draw_Path(visual_0, pos, 8, color, 1)
    
        pos[:,0] = history__v2_type2_x*10.0
        pos[:,1] = -1.5+multiplier_temp*aspect_ratio*history__v2_type2_y/np.sqrt(1e-6+np.abs(history__v2_type2_y))
        pos[:,2] = aspect_ratio*history__v2_type2_z
        color = colormap(np.linspace(0, 1, n), vmin=0, vmax=1, cmap='mycmap_v2_2_6')
        draw_Path(visual_0, pos, 8, color, 1)
        
    
        pos[:,0] = history__v2_type3_x*10.0
        pos[:,1] = -1.5+multiplier_temp*aspect_ratio*history__v2_type3_y/np.sqrt(1e-6+np.abs(history__v2_type3_y))
        pos[:,2] = aspect_ratio*history__v2_type3_z
        color = colormap(np.linspace(0, 1, n), vmin=0, vmax=1, cmap='mycmap_v2_3_7')
        draw_Path(visual_0, pos, 8, color, 1)
    
        pos[:,0] = history__v2_type4_x*10.0
        pos[:,1] = -1.5+multiplier_temp*aspect_ratio*history__v2_type4_y/np.sqrt(1e-6+np.abs(history__v2_type4_y))
        pos[:,2] = aspect_ratio*history__v2_type4_z
        color = colormap(np.linspace(0, 1, n), vmin=0, vmax=1, cmap='mycmap_v2_4_8')
        draw_Path(visual_0, pos, 8, color, 1)
    
        
        ##Vocals
        pos[:,0] = history__v3_type1_x*10.0
        pos[:,1] = -1.5+multiplier_temp*aspect_ratio*history__v3_type1_y/np.sqrt(1e-6+np.abs(history__v3_type1_y))
        pos[:,2] = aspect_ratio*history__v3_type1_z
        color = colormap(np.linspace(0, 1, n), vmin=0, vmax=1, cmap='mycmap_w_1_5')
        draw_Path(visual_0, pos, 8, color, 1)
    
    
        pos[:,0] = history__v3_type2_x*10.0
        pos[:,1] = -1.5+multiplier_temp*aspect_ratio*history__v3_type2_y/np.sqrt(1e-6+np.abs(history__v3_type2_y))
        pos[:,2] = aspect_ratio*history__v3_type2_z
        color = colormap(np.linspace(0, 1, n), vmin=0, vmax=1, cmap='mycmap_w_2_6')
        draw_Path(visual_0, pos, 8, color, 1)
        
    
        pos[:,0] = history__v3_type3_x*10.0
        pos[:,1] = -1.5+multiplier_temp*aspect_ratio*history__v3_type3_y/np.sqrt(1e-6+np.abs(history__v3_type3_y))
        pos[:,2] = aspect_ratio*history__v3_type3_z
        color = colormap(np.linspace(0, 1, n), vmin=0, vmax=1, cmap='mycmap_w_1_5')
        draw_Path(visual_0, pos, 8, color, 1)
    
        pos[:,0] = history__v3_type4_x*10.0
        pos[:,1] = -1.5+multiplier_temp*aspect_ratio*history__v3_type4_y/np.sqrt(1e-6+np.abs(history__v3_type4_y))
        pos[:,2] = aspect_ratio*history__v3_type4_z
        color = colormap(np.linspace(0, 1, n), vmin=0, vmax=1, cmap='mycmap_w_2_6')
        draw_Path(visual_0, pos, 8, color, 1)
    else:
        pos[:,1] = pos[:,1] = aspect_ratio*history__v3_type1_y*0.125 - 1.2
        color = colormap(np.linspace(0, 1, n), vmin=0, vmax=1, cmap='mycmap_w_1_5')
        draw_Path(visual_0, pos, 8, color, 1)
    #color[:,-1] = np.max([32, color[:,-1].all()]) + 256*np.abs(history__v3_type1_y )
        
        for i in range(Lines.shape[0]-4):
            pos[:,0] = 0.5 - 0.2*Lines[i]
            pos[:,1] = -5.0
            draw_Path(visual_0, pos, 8, color, 1)
        
        
    pos_2 = pos
    
    pos_2[-1,0] = history__v0_type1_x[-1]*10.0 + 2.5
    ms = np.array([32+ 64*history__v0_type1_y[-1]])
    ###Dots
    if(draw_mode == 7):
        ###V0
        x = np.hstack((history__v0_type1_y[-256:], history__v0_type2_y[-256:], history__v0_type3_y[-256:], history__v0_type4_y[-256:]))
    
        
        y = np.pi*np.cos((intensity*x+57.3*x)) * (_track1_beat_rms /(1+ (_track2_beat_rms+_track3_beat_rms+_track4_beat_rms)))
        x = np.pi*np.sin((intensity*x+57.3*x)) * (_track1_beat_rms /(1+ (_track2_beat_rms+_track3_beat_rms+_track4_beat_rms))) + 1
        
        z = 0.20*(x*y)
        pos_2 = np.c_[x, y, z]
    
    
        color_values = -z * (ms + 1)  + 1.0
        
        alpha = 15*color_values
        # (N, 4) array of uint8
        color = colormap(color_values, vmin=0, vmax=1, alpha=alpha, cmap='mycmap_v0_2_6')
        draw_Dots_at(visual_1, pos_2, alpha, color, 0)
    
    
        ###V1
        x = np.hstack((history__v1_type1_y[-256:], history__v1_type2_y[-256:], history__v1_type3_y[-256:], history__v1_type4_y[-256:]))
    
            
        y = np.pi*np.cos((intensity*x+57.3*x)) * (_track2_beat_rms /(1+ (_track1_beat_rms+_track3_beat_rms+_track4_beat_rms)))
        x = np.pi*np.sin((intensity*x+57.3*x)) * (_track2_beat_rms /(1+ (_track1_beat_rms+_track3_beat_rms+_track4_beat_rms))) + 1
        
        z = 0.20*(x*y)
        pos_2 = np.c_[x, y, z]
        
    
    
        color_values = -z * (ms + 1)  + 1.0
        
        alpha = 15*color_values
        # (N, 4) array of uint8
        color = colormap(color_values, vmin=0, vmax=1, alpha=alpha, cmap='mycmap_v1_2_6')
    
        draw_Dots_at(visual_1, pos_2, alpha, color, 1)
        
        
    
        ###V2
        x = np.hstack((history__v2_type1_y[-256:], history__v2_type2_y[-256:], history__v2_type3_y[-256:], history__v2_type4_y[-256:]))
    
            
        y = np.pi*(intensity+0.25)*np.cos((intensity*x+57.3*x)) * (_track3_beat_rms /(1+ (_track1_beat_rms+_track2_beat_rms+_track4_beat_rms))) + (intensity+5/8)*np.sin(2/3*(intensity*x+57.3*x)) * (_track3_beat_rms /(1+ (_track1_beat_rms+_track2_beat_rms+_track4_beat_rms)))
        x = np.pi*(intensity+0.25)*np.sin((intensity*x+57.3*x)) * (_track3_beat_rms /(1+ (_track1_beat_rms+_track2_beat_rms+_track4_beat_rms))) + (intensity+5/8)*np.cos(2/3*(intensity*x+57.3*x)) * (_track3_beat_rms /(1+ (_track1_beat_rms+_track2_beat_rms+_track4_beat_rms))) + 1
    
        z = 0.20*(x*y)
        pos_2 = np.c_[x, y, z]
        
    
    
        color_values = -z * (ms + 1)  + 1.0
        
        alpha = 15*color_values
        # (N, 4) array of uint8
        color = colormap(color_values, vmin=0, vmax=1, alpha=alpha, cmap='mycmap_v2_2_6')
        draw_Dots_at(visual_1, pos_2, alpha, color, 1)
    
    
    
        ###V3
        x = np.hstack((history__v3_type1_y[-256:], history__v3_type2_y[-256:], history__v3_type3_y[-256:], history__v3_type4_y[-256:]))
        #print(x)
        
        y = np.pi*(intensity+0.25)*np.cos((intensity*x+57.3*x)) * (_track4_beat_rms /(1+ (_track1_beat_rms+_track2_beat_rms+_track3_beat_rms))) + (intensity+5/8)*np.sin(2/3*(intensity*x+57.3*x)) * (_track4_beat_rms /(1+ (_track1_beat_rms+_track2_beat_rms+_track3_beat_rms)))
        x = np.pi*(intensity+0.25)*np.sin((intensity*x+57.3*x)) * (_track4_beat_rms /(1+ (_track1_beat_rms+_track2_beat_rms+_track3_beat_rms))) + (intensity+5/8)*np.cos(2/3*(intensity*x+57.3*x)) * (_track4_beat_rms /(1+ (_track1_beat_rms+_track2_beat_rms+_track3_beat_rms))) + 1
    
        z = 0.20*(x*y)
        pos_2 = np.c_[x, y, z]
        #ms = 15*np.ones(N)
    
    
        color_values = -z * (ms + 1)  + 1.0
        
        alpha = 15*color_values
        # (N, 4) array of uint8
        color = colormap(color_values, vmin=0, vmax=1, alpha=alpha, cmap='mycmap_v3_3_7')
        draw_Dots_at(visual_1, pos_2, alpha, color, 1)
    
    
    pos_2 = pos
    
    pos_2[-1,0] = history__v0_type1_x[-1]*10.0 + 2.5
    ms = np.array([32+ 64*history__v0_type1_y[-1]])
    
    
    ###Rest
    
    pos_2 = pos
    
    pos_2[-1,0] = history__v0_type1_x[-1]*10.0 + 2.5
    ms = np.array([32+ 64*history__v0_type1_y[-1]])
    if(draw_mode == 0):
        pos_2[-1,0] = history__v0_type1_x[-1]*10.0 + 2.75
        pos[:,1] = -5
        color = colormap(np.linspace(0, 1, n), vmin=0, vmax=1, cmap='mycmap_w_1_5')
        ms = np.array([64+ 128*history__v0_type1_y[-1]])
    elif(draw_mode == 1):
        pos_2[-1,0] = history__v0_type1_x[-1]*10.0 + 2.75
        pos[:,1] = 0
        color = colormap(np.linspace(0, 1, n), vmin=0, vmax=1, cmap='mycmap_w_1_5')
        ms = np.array([64+ 128*history__v0_type1_y[-1]])
    else:
        pos[:,1] = 1.2
        color = colormap(np.linspace(0, 1, n), vmin=0, vmax=1, cmap='mycmap_v0_1_5')
    #color[:,-1] = 140+128*np.abs(history__v0_type1_y)
    
    if(draw_mode == 7):
        draw_Dots_at(visual_1, pos_2[-1,:], ms, color[-1], 1)
    else:
        draw_Dots_at(visual_1, pos_2[-1,:], ms, color[-1], 0)
    
    
    
    
    
    ####Text + marker
    if(draw_mode == 5):
        s = str(np.argmax(_track1_beat_types))
    else:
        s = str(round(history__v0_type1_y[-1]/(1e-6+ np.max(history__v0_type1_y)),2))
        
    
    if(draw_mode == 0):
        s = "f (Hz) < perceptible"
        draw_Text_at(visual_2, s, np.array([2.5,1.0,0]),32*(1+round(history__v0_type1_y[-1]/(1e-6+ np.max(history__v0_type1_y)),2)), color[-1,:], 0)
    elif(draw_mode == 1):
        draw_Text_at(visual_2, s, pos_2[-1,:] + np.array([0.75,0,0]), 40, color[-1,:], 0)
    else:
        draw_Text_at(visual_2, s, pos_2[-1,:] + np.array([0.5,0,0]), 32, color[-1,:], 0)
        
    
    
    pos_2[-1,0] = history__v1_type1_x[-1]*10.0 + 2.5
    
    if(draw_mode == 0):
        pos[:,1] = -5.0
    elif(draw_mode == 1):
        pos[:,1] = -5.0
    else:
        pos[:,1] = 0.4
    color = colormap(np.linspace(0, 1, n), vmin=0, vmax=1, cmap='mycmap_v1_1_5')
    #color[:,-1] = 140+128*np.abs(history__v1_type1_y)
    ms = np.array([32+ 64*history__v1_type1_y[-1]])
    
    draw_Dots_at(visual_1, pos_2[-1,:], ms, color[-1], 1)
    
    
    
    if(draw_mode == 5):
        s = str(np.argmax(_track2_beat_types))
    else:
        s = str(round(history__v1_type1_y[-1]/(1e-6+ np.max(history__v1_type1_y)),2))
    draw_Text_at(visual_2, s, pos_2[-1,:] + np.array([0.5,0,0]), 32, color[-1,:], 1)
    

    
    pos_2[-1,0] = history__v2_type1_x[-1]*10.0 + 2.5
    
    if(draw_mode == 0):
        pos[:,1] = -5.0
    elif(draw_mode == 1):
        pos[:,1] = -5.0
    else:
        pos[:,1] = -0.4
    color = colormap(np.linspace(0, 1, n), vmin=0, vmax=1, cmap='mycmap_v2_1_5')
    #color[:,-1] = 140+128*np.abs(history__v2_type1_y)
    ms = np.array([32+ 64*history__v2_type1_y[-1]])
    
    draw_Dots_at(visual_1, pos_2[-1,:], ms, color[-1], 1)
    
    if(draw_mode == 5):
        s = str(np.argmax(_track3_beat_types))
    else:
        s = str(round(history__v2_type1_y[-1]/(1e-6+ np.max(history__v2_type1_y)),2))
    draw_Text_at(visual_2, s, pos_2[-1,:] + np.array([0.5,0,0]), 32, color[-1,:], 1)
    
    
    pos_2[-1,0] = history__v3_type1_x[-1]*10.0 + 2.5
    
    if(draw_mode == 0):
        pos[:,1] = -5.0
    elif(draw_mode == 1):
        pos[:,1] = -5.0
    else:
        pos[:,1] = -1.2
    ms = np.array([32+ 64*history__v3_type1_y[-1]])
    color = colormap(np.linspace(0, 1, n), vmin=0, vmax=1, cmap='mycmap_w_1_5')
    #color[:,-1] = 140+128*np.abs(history__v3_type1_y )
    ms = np.array([32+ 32*history__v3_type1_y[-1]])
    
    draw_Dots_at(visual_1, pos_2[-1,:], ms, color[-1], 1)
    
    if(draw_mode == 5):
        s = str(np.argmax(_track4_beat_types))
    else:
        s = str(round(history__v3_type1_y[-1]/(1e-6+ np.max(history__v3_type1_y)),2))
        
    draw_Text_at(visual_2, s, pos_2[-1,:] + np.array([0.5,0,0]), 32, color[-1,:], 1)
    
    
    


@c.connect
def on_frame(i):
    global draw_elapsed, read_elapsed,  dt, pos, history__v0_type1, history__v0_type1_x, history__v0_type1_y, history__v0_type2, history__v0_type2_x, history__v0_type2_y,  history__v0_type3
    global start_time, delay, dt, span, t0, t1, history_et, history_at
    global _track1_beat_rms, measurements, beat_offset, draw_fps, draw_mode, ready_switch

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
    ready_switch -= dt

    read_elapsed += dt 
    draw_elapsed += dt

    draw_fps = draw_fps + (256-draw_fps)*0.9  
    #
    if(intensity > 0.01*intensity_max or intensity < -0.01*intensity_min):
        if(ready_switch < 0):
            draw_mode = (draw_mode + 1)%7
            ready_switch = 20.0/(1+np.sqrt(draw_mode))
        #draw_mode = 0
        #draw_fps = 1
    #elif(intensity < intensity_cutoff - 0.01*(intensity_max-intensity_min)):
    #    draw_mode = 1
    #else:
    #    draw_mode = 2
    
    #if(intensity > intensity_cutoff + 0.4*(intensity_max-intensity_min) or intensity < intensity_cutoff - 0.4*(intensity_max-intensity_min)):
    #    #draw_mode = 1
    #    draw_fps = 1
    #    #draw_mode = 1
    ##else:
    #    #draw_mode = 1
        
    if(draw_elapsed > 1/(1+draw_fps)):
        draw()  
        draw_elapsed = 0
        
##GUI
gui = c.gui("GUI")

b = gui.control("button", "Mode")
@b.connect
def on_change(value):
    global draw_mode
    # We update the marker positions.
    draw_mode = (draw_mode + 1)%7

run()
c.close()
