import madmom
import numpy as np

import os
import sys 

from pathlib import Path  

import musicsections


extensions = ["mp3", "wav", "ogg", "flac"]  # we will look for all those file types.

    
def rhythmic_patterns_analysis(input_file, annotations_file, delimiter = ',',
                               downbeat_label = '.1', n_tatums = 4, n_clusters = 4):
    '''Rhythmic patterns analysis

    :parameters:
      - input_file : str
          path to input audio file (wav, mp3, m4a, flac, etc.)
      - annotations_file : str
          path to the annotations file (txt, csv, etc)
      - delimiter: str
          delimiter string to process the annotations file
      - downbeat_label: str
          string to look for in the label data to select downbeats
      - n_tatums: int
          number of tatums (subdivisions) per tactus beat
      - n_clusters: int
          number of clusters for rhythmic patterns clustering
    '''

    # =================== MAIN PROCESSING ===================
    import carat
    # 1. load the wav file
    print('Loading audio file ...', input_file)
    sr = 44100
    y = madmom.audio.signal.Signal(input_file, sample_rate = sr,dtype = np.float32,num_channels=1)

    # 2. load beat and downbeat annotations
    print('Loading beat and downbeat annotations ...', annotations_file)
    beats, _ = carat.annotations.load_beats(annotations_file, delimiter=delimiter)
    downbeats, _ = carat.annotations.load_downbeats(annotations_file, delimiter=delimiter,
                                                    downbeat_label='1')
    # number of beats per bar
    n_beats = int(round(beats.size/downbeats.size))

    # 3. compute accentuation feature
    print('Computing accentuation feature ...')
    acce, times, _ = carat.features.accentuation_feature(y, sr, minfreq=20, maxfreq=200)

    # 4. compute feature map
    print('Computing feature map ...')
    map_acce, _, _, _ = carat.features.feature_map(acce, times, beats, downbeats,
                                                   n_beats=n_beats, n_tatums=n_tatums)
    
    # 5. cluster rhythmic patterns
    print('Clustering rhythmic patterns ...')
    cluster_labs, centroids, _ = carat.clustering.rhythmic_patterns(map_acce, n_clusters=n_clusters)
    
    track_path = str(input_file)
    annotation_path = str(annotations_file)
    sub = ""
    
    if(annotation_path[-5:] == "2.csv"):
        if(track_path[-9:] == "drums.mp3" or track_path[-9:] == "drums.wav"):
            track_path = track_path [:-9]
            sub = "drums_beats_c2.csv"
        elif(track_path[-8:] == "bass.mp3" or track_path[-8:] == "bass.wav"):
            track_path = track_path [:-8]
            sub = "bass_beats_c2.csv"
        elif(track_path[-9:] == "other.mp3" or track_path[-9:] == "other.wav"):
            track_path = track_path [:-9]
            sub = "other_beats_c2.csv"
        elif(track_path[-10:] == "vocals.mp3" or track_path[-10:] == "vocals.wav"):
            track_path = track_path [:-10]
            sub = "vocals_beats_c2.csv"
        print(track_path)
        
    else:
        if(track_path[-9:] == "drums.mp3" or track_path[-9:] == "drums.wav"):
            track_path = track_path [:-9]
            sub = "drums_beats_c.csv"
        elif(track_path[-8:] == "bass.mp3" or track_path[-8:] == "bass.wav"):
            track_path = track_path [:-8]
            sub = "bass_beats_c.csv"
        elif(track_path[-9:] == "other.mp3" or track_path[-9:] == "other.wav"):
            track_path = track_path [:-9]
            sub = "other_beats_c.csv"
        elif(track_path[-10:] == "vocals.mp3" or track_path[-10:] == "vocals.wav"):
            track_path = track_path [:-10]
            sub = "vocals_beats_c.csv"
        print(track_path)
    
    np.savetxt(track_path+sub, cluster_labs, delimiter=",")
    
def Process():
    import os
    import sys   
       
    stems = sys.argv[1:]
    
    for stem in stems:
        process(stem)
    
def process(stem = ""):
    import madmom
    import numpy as np
    from sklearn.cluster import KMeans, AgglomerativeClustering, FeatureAgglomeration
    
    import os
    import sys   
    
    stem = stem
    
    from os import path

    basepath = path.dirname(__file__)

    track_path = basepath + "/separated/mdx_extra/" + stem + "/"
    audio_path = basepath + "/" + stem
    
    
    #in_path = basepath
        
    
    #out_path = in_path + "/separated/"
    
    
    def find_files(in_path):
        out_2 = []
        for file in Path(in_path).iterdir():
            if file.suffix.lower().lstrip(".") in extensions:
                out_2.append(file)
        return out_2
        
    files = [str(f) for f in find_files(track_path)]
    
    deepsim_model_folder = basepath+"/models/deepsim/"
    fewshot_model_folder = basepath+"/models/fewshot/"
    
    model_deepsim = musicsections.load_deepsim_model(deepsim_model_folder)
    model_fewshot = musicsections.load_fewshot_model(fewshot_model_folder)
    
    signal_drums = madmom.audio.signal.Signal(files[1], sample_rate = 44100,dtype = np.float32,num_channels=1)
    signal_bass = madmom.audio.signal.Signal(files[0], sample_rate = 44100,dtype = np.float32,num_channels=1)
    signal_other = madmom.audio.signal.Signal(files[2], sample_rate = 44100,dtype = np.float32,num_channels=1)
    signal_vocals = madmom.audio.signal.Signal(files[3], sample_rate = 44100,dtype = np.float32,num_channels=1)
    
    fs_drums = madmom.audio.signal.FramedSignal(signal_drums, frame_size=2048, hop_size = 441)
    fs_bass = madmom.audio.signal.FramedSignal(signal_bass, frame_size=2048, hop_size = 441)
    fs_other = madmom.audio.signal.FramedSignal(signal_other, frame_size=2048, hop_size = 441)
    fs_vocals = madmom.audio.signal.FramedSignal(signal_vocals, frame_size=2048, hop_size = 441)
    
    spec_drums = madmom.audio.spectrogram.Spectrogram(fs_drums)
    spec_bass = madmom.audio.spectrogram.Spectrogram(fs_bass)
    spec_other = madmom.audio.spectrogram.Spectrogram(fs_other) 
    spec_vocals = madmom.audio.spectrogram.Spectrogram(fs_vocals)
    
    log_filt_spec_drums = madmom.audio.spectrogram.LogarithmicFilteredSpectrogram(spec_drums, num_bands=8)
    log_filt_spec_bass = madmom.audio.spectrogram.LogarithmicFilteredSpectrogram(spec_bass, num_bands=8)
    log_filt_spec_other = madmom.audio.spectrogram.LogarithmicFilteredSpectrogram(spec_other, num_bands=8)
    log_filt_spec_vocals = madmom.audio.spectrogram.LogarithmicFilteredSpectrogram(spec_vocals, num_bands=8)
    
    superflux_drums = madmom.features.onsets.superflux(log_filt_spec_drums)
    superflux_bass = madmom.features.onsets.superflux(log_filt_spec_bass)
    superflux_other = madmom.features.onsets.superflux(log_filt_spec_other)
    superflux_vocals = madmom.features.onsets.superflux(log_filt_spec_vocals)
    
    superflux_drums_norm = superflux_drums/superflux_drums.max()
    superflux_bass_norm = superflux_bass/superflux_bass.max()
    superflux_other_norm = superflux_other/superflux_other.max()
    superflux_vocals_norm = superflux_vocals/superflux_vocals.max()
    
    np.savetxt(track_path+"superflux_drums_norm.csv", superflux_drums_norm, delimiter=",")
    np.savetxt(track_path+"superflux_bass_norm.csv", superflux_bass_norm, delimiter=",")
    np.savetxt(track_path+"superflux_other_norm.csv", superflux_other_norm, delimiter=",")
    np.savetxt(track_path+"superflux_vocals_norm.csv", superflux_vocals_norm, delimiter=",")
    
    
    proc = madmom.features.beats.RNNBeatProcessor(post_processor = None)
    
    predictions_drums = proc(track_path+"drums.wav")
    predictions_bass = proc(track_path+"bass.wav")
    predictions_other = proc(track_path+"other.wav")
    predictions_vocals = proc(track_path+"vocals.wav")
    
    import scipy.stats
    
    mm_proc = madmom.features.beats.MultiModelSelectionProcessor(num_ref_predictions = None)
    
    drums_beats = mm_proc(predictions_drums)
    bass_beats = mm_proc(predictions_bass)
    other_beats = mm_proc(predictions_other)
    vocals_beats = mm_proc(predictions_vocals)
    
    when_beats_drums = madmom.features.beats.BeatTrackingProcessor(fps=200)(drums_beats)
    drums_res = scipy.stats.linregress(np.arange(len(when_beats_drums)), when_beats_drums)
    
    drums_first = drums_res.intercept
    beat_step_drums = drums_res.slope
    
    print(beat_step_drums, 60/beat_step_drums)
    
    when_beats_bass = madmom.features.beats.BeatTrackingProcessor(fps=200)(bass_beats)
    bass_res = scipy.stats.linregress(np.arange(len(when_beats_bass)), when_beats_bass)
    
    bass_first = bass_res.intercept
    beat_step_bass = bass_res.slope
    
    print(beat_step_bass, 60/beat_step_bass)
    
    when_beats_other = madmom.features.beats.BeatTrackingProcessor(fps=200)(other_beats)
    other_res = scipy.stats.linregress(np.arange(len(when_beats_other)), when_beats_other)
    
    other_first = other_res.intercept
    beat_step_other = other_res.slope
    
    print(beat_step_other, 60/beat_step_other)
    
    when_beats_vocals = madmom.features.beats.BeatTrackingProcessor(fps=200)(vocals_beats)
    vocals_res = scipy.stats.linregress(np.arange(len(when_beats_vocals)), when_beats_vocals)
    
    vocals_first = vocals_res.intercept
    beat_step_vocals = vocals_res.slope
    
    print(beat_step_vocals, 60/beat_step_vocals)
    
    
    drums_beats = np.array(np.where(drums_beats > 0.25))[0]/(drums_beats.shape[0]/(signal_drums.shape[0]/signal_drums.sample_rate))
    bass_beats = np.array(np.where(bass_beats > 0.1))[0]/(bass_beats.shape[0]/(signal_bass.shape[0]/signal_bass.sample_rate))
    other_beats = np.array(np.where(other_beats > 0.15))[0]/(other_beats.shape[0]/(signal_other.shape[0]/signal_other.sample_rate))
    ######!!!!!!!!!!Adjust 0.0 up for lyrics
    vocals_beats = np.array(np.where(vocals_beats > 0.15))[0]/(vocals_beats.shape[0]/(signal_vocals.shape[0]/signal_vocals.sample_rate))
    
    frac = 1.0/4.0
    
    drums_beat_count = 1
    
    for i in range(drums_beats.shape[0]-1):
        if((drums_beats[i+1] - drums_beats[i]) > beat_step_drums*frac):
            drums_beat_count += 1
            
    drums_beats_adj = np.zeros((drums_beat_count, 2))
    
    drums_beats_adj[0] = [drums_beats[0],1]
    
    current_beat = 0
    
    for i in range(drums_beats.shape[0]-1):
        if((drums_beats[i+1] - drums_beats[i]) > beat_step_drums*frac):
            current_beat += 1
            drums_beats_adj[current_beat] =  [drums_beats[i+1], 1]
            
    labels_drums = (drums_beats_adj[:,1]).astype(str)
    drums_beats_adj[:,1] = labels_drums
    
    if(drums_beats_adj[0,0] ==  0.0):
        np.savetxt(track_path+"drums_beats_2.csv", drums_beats_adj[1:], delimiter=",")
    else:
        np.savetxt(track_path+"drums_beats_2.csv", drums_beats_adj, delimiter=",")
    
    
    bass_beat_count = 1
    
    for i in range(bass_beats.shape[0]-1):
        if((bass_beats[i+1] - bass_beats[i]) > beat_step_bass*frac):
            bass_beat_count += 1
            
    bass_beats_adj = np.zeros((bass_beat_count, 2))
    
    bass_beats_adj[0] = [bass_beats[0],1]
    
    current_beat = 0
    
    for i in range(bass_beats.shape[0]-1):
        if((bass_beats[i+1] - bass_beats[i]) > beat_step_bass*frac):
            current_beat += 1
            bass_beats_adj[current_beat] =  [bass_beats[i+1], 1]
            
    labels_bass = (bass_beats_adj[:,1]).astype(str)
    bass_beats_adj[:,1] = labels_bass
    
    if(bass_beats_adj[0,0] ==  0.0):
        np.savetxt(track_path+"bass_beats_2.csv", bass_beats_adj[1:], delimiter=",")
    else:
        np.savetxt(track_path+"bass_beats_2.csv", bass_beats_adj, delimiter=",")
    
    other_beat_count = 1
    
    for i in range(other_beats.shape[0]-1):
        if((other_beats[i+1] - other_beats[i]) > beat_step_other*frac):
            other_beat_count += 1
            
    other_beats_adj = np.zeros((other_beat_count, 2))
    
    other_beats_adj[0] = [other_beats[0],1]
    
    current_beat = 0
    
    for i in range(other_beats.shape[0]-1):
        if((other_beats[i+1] - other_beats[i]) > beat_step_other*frac):
            current_beat += 1
            other_beats_adj[current_beat] =  [other_beats[i+1], 1]
            
    labels_other = (other_beats_adj[:,1]).astype(str)
    other_beats_adj[:,1] = labels_other
    
    if(other_beats_adj[0,0] ==  0.0):
        np.savetxt(track_path+"other_beats_2.csv", other_beats_adj[1:], delimiter=",")
    else:
        np.savetxt(track_path+"other_beats_2.csv", other_beats_adj, delimiter=",")
    
    
    if(vocals_beats.size < 20):
        drums_beats_temp =  np.array(np.where(mm_proc(predictions_drums) > 0.5))[0]
        bass_beats_temp  = np.array(np.where(mm_proc(predictions_bass) > 0.1))[0]
        other_beats_temp =  np.array(np.where(mm_proc(predictions_other) > 0.1))[0] 
        vocals_beats_temp =   np.sort(np.hstack((drums_beats_temp, bass_beats_temp, other_beats_temp)))
    
        combined_max_ind = np.max((drums_beats_temp.max(),bass_beats_temp.max(),other_beats_temp.max()))
    
        drums_beats_instr = drums_beats_temp/(drums_beats_temp.shape[0]/(signal_drums.shape[0]/signal_drums.sample_rate))
        bass_beats_instr = bass_beats_temp/(bass_beats_temp.shape[0]/(signal_bass.shape[0]/signal_bass.sample_rate))
        other_beats_instr = other_beats_temp/(other_beats_temp.shape[0]/(signal_other.shape[0]/signal_other.sample_rate))
        ######!!!!!!!!!!Adjust 0.0 up for lyrics
        vocals_beats_instr = vocals_beats_temp/(combined_max_ind/(signal_vocals.shape[0]/signal_vocals.sample_rate))
        vocals_beats = vocals_beats_instr
    
    vocals_beat_count = 1
    
    for i in range(vocals_beats.shape[0]-1):
        if((vocals_beats[i+1] - vocals_beats[i]) > beat_step_vocals*frac):
            vocals_beat_count += 1
            
    vocals_beats_adj = np.zeros((vocals_beat_count, 2))
    
    vocals_beats_adj[0] = [vocals_beats[0],1]
    
    current_beat = 0
    
    for i in range(vocals_beats.shape[0]-1):
        if((vocals_beats[i+1] - vocals_beats[i]) > beat_step_vocals*frac):
            current_beat += 1
            vocals_beats_adj[current_beat] =  [vocals_beats[i+1], 1]
            
    labels_vocals = (vocals_beats_adj[:,1]).astype(str)
    vocals_beats_adj[:,1] = labels_vocals
    
    if(vocals_beats_adj[0,0] ==  0.0):
        np.savetxt(track_path+"vocals_beats_2.csv", vocals_beats_adj[1:], delimiter=",")
    else:
        np.savetxt(track_path+"vocals_beats_2.csv", vocals_beats_adj, delimiter=",")
    
    
    
    drums_beats = mm_proc(predictions_drums)
    bass_beats = mm_proc(predictions_bass)
    other_beats = mm_proc(predictions_other)
    vocals_beats = mm_proc(predictions_vocals)
    
    drums_beats = np.array(np.where(drums_beats > 0.25))[0]/(drums_beats.shape[0]/(signal_drums.shape[0]/signal_drums.sample_rate))
    bass_beats = np.array(np.where(bass_beats > 0.1))[0]/(bass_beats.shape[0]/(signal_bass.shape[0]/signal_bass.sample_rate))
    other_beats = np.array(np.where(other_beats > 0.15))[0]/(other_beats.shape[0]/(signal_other.shape[0]/signal_other.sample_rate))
    ######!!!!!!!!!!Adjust 0.0 up for lyrics
    vocals_beats = np.array(np.where(vocals_beats > 0.15))[0]/(vocals_beats.shape[0]/(signal_vocals.shape[0]/signal_vocals.sample_rate))
    
    frac = 1.0/16.0
    
    drums_beat_count = 1
    
    for i in range(drums_beats.shape[0]-1):
        if((drums_beats[i+1] - drums_beats[i]) > beat_step_drums*frac):
            drums_beat_count += 1
            
    drums_beats_adj = np.zeros((drums_beat_count, 2))
    
    drums_beats_adj[0] = [drums_beats[0],1]
    
    current_beat = 0
    
    for i in range(drums_beats.shape[0]-1):
        if((drums_beats[i+1] - drums_beats[i]) > beat_step_drums*frac):
            current_beat += 1
            drums_beats_adj[current_beat] =  [drums_beats[i+1], 1]
            
    labels_drums = (drums_beats_adj[:,1]).astype(str)
    drums_beats_adj[:,1] = labels_drums
    
    if(drums_beats_adj[0,0] ==  0.0):
        np.savetxt(track_path+"drums_beats.csv", drums_beats_adj[1:], delimiter=",")
    else:
        np.savetxt(track_path+"drums_beats.csv", drums_beats_adj, delimiter=",")
    
    bass_beat_count = 1
    
    for i in range(bass_beats.shape[0]-1):
        if((bass_beats[i+1] - bass_beats[i]) > beat_step_bass*frac):
            bass_beat_count += 1
            
    bass_beats_adj = np.zeros((bass_beat_count, 2))
    
    bass_beats_adj[0] = [bass_beats[0],1]
    
    current_beat = 0
    
    for i in range(bass_beats.shape[0]-1):
        if((bass_beats[i+1] - bass_beats[i]) > beat_step_bass*frac):
            current_beat += 1
            bass_beats_adj[current_beat] =  [bass_beats[i+1], 1]
            
    labels_bass = (bass_beats_adj[:,1]).astype(str)
    bass_beats_adj[:,1] = labels_bass
    
    if(bass_beats_adj[0,0] ==  0.0):
        np.savetxt(track_path+"bass_beats.csv", bass_beats_adj[1:], delimiter=",")
    else:
        np.savetxt(track_path+"bass_beats.csv", bass_beats_adj, delimiter=",")
    
    other_beat_count = 1
    
    for i in range(other_beats.shape[0]-1):
        if((other_beats[i+1] - other_beats[i]) > beat_step_other*frac):
            other_beat_count += 1
            
    other_beats_adj = np.zeros((other_beat_count, 2))
    
    other_beats_adj[0] = [other_beats[0],1]
    
    current_beat = 0
    
    for i in range(other_beats.shape[0]-1):
        if((other_beats[i+1] - other_beats[i]) > beat_step_other*frac):
            current_beat += 1
            other_beats_adj[current_beat] =  [other_beats[i+1], 1]
            
    labels_other = (other_beats_adj[:,1]).astype(str)
    other_beats_adj[:,1] = labels_other
    
    if(other_beats_adj[0,0] ==  0.0):
        np.savetxt(track_path+"other_beats.csv", other_beats_adj[1:], delimiter=",")
    else:
        np.savetxt(track_path+"other_beats.csv", other_beats_adj, delimiter=",")
    
    
    
    if(vocals_beats.size < 20):
        
        drums_beats_temp =  np.array(np.where(mm_proc(predictions_drums) > 0.5))[0]
        bass_beats_temp  = np.array(np.where(mm_proc(predictions_bass) > 0.1))[0]
        other_beats_temp =  np.array(np.where(mm_proc(predictions_other) > 0.1))[0] 
        vocals_beats_temp =   np.sort(np.hstack((drums_beats_temp, bass_beats_temp, other_beats_temp)))
    
        combined_max_ind = np.max((drums_beats_temp.max(),bass_beats_temp.max(),other_beats_temp.max()))
    
        drums_beats_instr = drums_beats_temp/(drums_beats_temp.shape[0]/(signal_drums.shape[0]/signal_drums.sample_rate))
        bass_beats_instr = bass_beats_temp/(bass_beats_temp.shape[0]/(signal_bass.shape[0]/signal_bass.sample_rate))
        other_beats_instr = other_beats_temp/(other_beats_temp.shape[0]/(signal_other.shape[0]/signal_other.sample_rate))
        ######!!!!!!!!!!Adjust 0.0 up for lyrics
        vocals_beats_instr = vocals_beats_temp/(combined_max_ind/(signal_vocals.shape[0]/signal_vocals.sample_rate))
        vocals_beats = vocals_beats_instr
    
        #print(vocals_beats)
    
    vocals_beat_count = 1
    
    for i in range(vocals_beats.shape[0]-1):
        if((vocals_beats[i+1] - vocals_beats[i]) > beat_step_vocals*frac):
            vocals_beat_count += 1
            
    vocals_beats_adj = np.zeros((vocals_beat_count, 2))
    
    vocals_beats_adj[0] = [vocals_beats[0],1]
    
    current_beat = 0
    
    for i in range(vocals_beats.shape[0]-1):
        if((vocals_beats[i+1] - vocals_beats[i]) > beat_step_vocals*frac):
            current_beat += 1
            vocals_beats_adj[current_beat] =  [vocals_beats[i+1], 1]
            
    labels_vocals = (vocals_beats_adj[:,1]).astype(str)
    vocals_beats_adj[:,1] = labels_vocals
    
    if(vocals_beats_adj[0,0] ==  0.0):
        np.savetxt(track_path+"vocals_beats.csv", vocals_beats_adj[1:], delimiter=",")
    else:
        np.savetxt(track_path+"vocals_beats.csv", vocals_beats_adj, delimiter=",")
    
    
    
    input_file_drums = track_path + "drums.wav" 
    annotations_file_drums = track_path + "drums_beats.csv"
    annotations_file_drums_2 = track_path + "drums_beats_2.csv"
    
    input_file_bass = track_path + "bass.wav" 
    annotations_file_bass = track_path + "bass_beats.csv"
    annotations_file_bass_2 = track_path + "bass_beats_2.csv"
    
    input_file_other = track_path + "other.wav" 
    annotations_file_other = track_path + "other_beats.csv"
    annotations_file_other_2 = track_path + "other_beats_2.csv"
    
    input_file_vocals = track_path + "vocals.wav" 
    annotations_file_vocals = track_path + "vocals_beats.csv"
    annotations_file_vocals_2 = track_path + "vocals_beats_2.csv"
    
    
    rhythmic_patterns_analysis(input_file_drums, annotations_file_drums)
    rhythmic_patterns_analysis(input_file_drums, annotations_file_drums_2)
    rhythmic_patterns_analysis(input_file_bass, annotations_file_bass)
    rhythmic_patterns_analysis(input_file_bass, annotations_file_bass_2)
    rhythmic_patterns_analysis(input_file_other, annotations_file_other)
    rhythmic_patterns_analysis(input_file_other, annotations_file_other_2)
    rhythmic_patterns_analysis(input_file_vocals, annotations_file_vocals)
    rhythmic_patterns_analysis(input_file_vocals, annotations_file_vocals_2)
    
    drums_loudness = fs_drums.rms()
    bass_loudness = fs_bass.rms()
    other_loudness = fs_other.rms()
    vocals_loudness = fs_vocals.rms()
    
    combined_rms = (drums_loudness + bass_loudness + other_loudness + vocals_loudness)
    
    
    
    drums_delta = np.sqrt(drums_loudness) + drums_loudness*np.abs(np.gradient(drums_loudness))
    bass_delta = np.sqrt(bass_loudness) + bass_loudness*np.abs(np.gradient(bass_loudness))
    other_delta = np.sqrt(other_loudness) + other_loudness*np.abs(np.gradient(other_loudness))
    vocals_delta = np.sqrt(vocals_loudness) + vocals_loudness*np.abs(np.gradient(vocals_loudness))
    
    drums_delta /= drums_delta.max()
    bass_delta /= bass_delta.max()
    other_delta /= other_delta.max()
    vocals_delta /= vocals_delta.max()
    
    
    import numpy as np
    
    
    
    
    spec_drums = madmom.audio.spectrogram.Spectrogram(fs_drums)
    spec_bass = madmom.audio.spectrogram.Spectrogram(fs_bass)
    spec_other = madmom.audio.spectrogram.Spectrogram(fs_other) 
    spec_vocals = madmom.audio.spectrogram.Spectrogram(fs_vocals)
    
    log_filt_spec_drums = madmom.audio.spectrogram.LogarithmicFilteredSpectrogram(spec_drums, num_bands=8)
    log_filt_spec_bass = madmom.audio.spectrogram.LogarithmicFilteredSpectrogram(spec_bass, num_bands=8)
    log_filt_spec_other = madmom.audio.spectrogram.LogarithmicFilteredSpectrogram(spec_other, num_bands=8)
    log_filt_spec_vocals = madmom.audio.spectrogram.LogarithmicFilteredSpectrogram(spec_vocals, num_bands=8)
    
    _track1_type2 = np.loadtxt(track_path+'drums_beats_c.csv', delimiter=',')
    _track2_type2 = np.loadtxt(track_path+'bass_beats_c.csv', delimiter=',')
    _track3_type2 = np.loadtxt(track_path+'other_beats_c.csv', delimiter=',')
    _track4_type2 = np.loadtxt(track_path+'vocals_beats_c.csv', delimiter=',')
    
    #_track1_beats = np.loadtxt(track_path+'drums_beats_2.csv', delimiter=',')[:,0]
    #_track2_beats = np.loadtxt(track_path+'bass_beats_2.csv', delimiter=',')[:,0]
    #_track3_beats = np.loadtxt(track_path+'other_beats_2.csv', delimiter=',')[:,0]
    #_track4_beats = np.loadtxt(track_path+'vocals_beats_2.csv', delimiter=',')[:,0]
    
    _track1_beats = np.loadtxt(track_path+'drums_beats.csv', delimiter=',')[:,0]
    _track2_beats = np.loadtxt(track_path+'bass_beats.csv', delimiter=',')[:,0]
    _track3_beats = np.loadtxt(track_path+'other_beats.csv', delimiter=',')[:,0]
    _track4_beats = np.loadtxt(track_path+'vocals_beats.csv', delimiter=',')[:,0]
    
    track_length = signal_drums.shape[0]/signal_drums.sample_rate
    
    segmentations_drums, features_drums = musicsections.segment_file(
    files[1], 
    deepsim_model=model_deepsim,
    fewshot_model=model_fewshot,
    min_duration=1,
    mu=0.5,
    gamma=0.5,
    beats_alg="madmom",
    beats_file=None)
    #musicsections.plot_segmentation(segmentations_drums)
    ###    
    segmentations_bass, features_bass = musicsections.segment_file(
        files[0], 
        deepsim_model=model_deepsim,
        fewshot_model=model_fewshot,
        min_duration=1,
        mu=0.5,
        gamma=0.5,
        beats_alg="madmom",
        beats_file=None)
    #musicsections.plot_segmentation(segmentations_bass)
    ###
    segmentations_other, features_other = musicsections.segment_file(
        files[2], 
        deepsim_model=model_deepsim,
        fewshot_model=model_fewshot,
        min_duration=1,
        mu=0.5,
        gamma=0.5,
        beats_alg="madmom",
        beats_file=None)
    #musicsections.plot_segmentation(segmentations_other)
    ###
    segmentations_vocals, features_vocals = musicsections.segment_file(
        files[3], 
        deepsim_model=model_deepsim,
        fewshot_model=model_fewshot,
        min_duration=1,
        mu=0.5,
        gamma=0.5,
        beats_alg="madmom",
        beats_file=None)
    #musicsections.plot_segmentation(segmentations_vocals)
    

    
    drums_1to8 = np.zeros((drums_delta.shape[0], int(_track1_type2.max() - _track1_type2.min() + 1)))
    bass_1to8 = np.zeros((bass_delta.shape[0], int(_track2_type2.max() - _track2_type2.min() + 1)))
    other_1to8 = np.zeros((other_delta.shape[0], int(_track3_type2.max() - _track3_type2.min() + 1)))
    vocals_1to8 = np.zeros((vocals_delta.shape[0], int(_track4_type2.max() - _track4_type2.min() + 1)))
    
    last_t = 0
    last_delta = 0
    
    for i in range(0, _track1_type2.shape[0]):
        current_t = _track1_beats[i]
        current_delta = current_t*(drums_delta.shape[0]/track_length)
        #print(current_t,current_delta)
        current_type = _track1_type2[i]
        drums_1to8[int(last_delta):int(current_delta), int(current_type)] = current_type
        #drums_delta[int(last_delta):int(current_delta)]
        #drums_1to8[int(last_delta):int(current_delta), int(current_type)] = i/10
        
        last_t = current_t
        last_delta = current_delta
        
    last_t = 0
    last_delta = 0
    
    for i in range(0, _track2_type2.shape[0]):
        current_t = _track2_beats[i]
        current_delta = current_t*(bass_delta.shape[0]/track_length)
        #print(current_t,current_delta)
        current_type = _track2_type2[i]
        bass_1to8[int(last_delta):int(current_delta), int(current_type)] = current_type
        #bass_1to8[int(last_delta):int(current_delta), int(current_type)] = i/10
        
        last_t = current_t
        last_delta = current_delta
        
    last_t = 0
    last_delta = 0
    
    for i in range(0, _track3_type2.shape[0]):
        current_t = _track3_beats[i]
        current_delta = current_t*(other_delta.shape[0]/track_length)
        #print(current_t,current_delta)
        current_type = _track3_type2[i]
        other_1to8[int(last_delta):int(current_delta), int(current_type)] = current_type
        #other_1to8[int(last_delta):int(current_delta), int(current_type)] = i/10
        
        last_t = current_t
        last_delta = current_delta
        
    
    last_t = 0
    last_delta = 0
    
    for i in range(0, _track4_type2.shape[0]):
        current_t = _track4_beats[i]
        current_delta = current_t*(vocals_delta.shape[0]/track_length)
        #print(current_t,current_delta)
        current_type = _track4_type2[i]
        vocals_1to8[int(last_delta):int(current_delta), int(current_type)] = current_type
        #vocals_1to8[int(last_delta):int(current_delta), int(current_type)] = i/10
        
        last_t = current_t
        last_delta = current_delta
        
    np.savetxt(track_path+"drums_beats_cd.csv", drums_1to8, delimiter=",")
    np.savetxt(track_path+"bass_beats_cd.csv", bass_1to8, delimiter=",")
    np.savetxt(track_path+"other_beats_cd.csv", other_1to8, delimiter=",")
    np.savetxt(track_path+"vocals_beats_cd.csv", vocals_1to8, delimiter=",")
    
    kernel_size = 8
    kernel = np.ones(kernel_size) / kernel_size
    drums_delta = np.convolve(drums_delta, kernel, mode='same')
    bass_delta = np.convolve(bass_delta, kernel, mode='same')
    other_delta = np.convolve(other_delta, kernel, mode='same')
    vocals_delta = np.convolve(vocals_delta, kernel, mode='same')
    
    np.savetxt(track_path+"drums_trend.csv", drums_delta, delimiter=",")
    np.savetxt(track_path+"bass_trend.csv", bass_delta, delimiter=",")
    np.savetxt(track_path+"other_trend.csv", other_delta, delimiter=",")
    np.savetxt(track_path+"vocals_trend.csv", vocals_delta, delimiter=",")

    #drums_1to8_2 = np.zeros((drums_delta.shape[0], int(_track1_type2.max() - _track1_type2.min() + 1)))
    #bass_1to8_2 = np.zeros((bass_delta.shape[0], int(_track2_type2.max() - _track2_type2.min() + 1)))
    #other_1to8_2 = np.zeros((other_delta.shape[0], int(_track3_type2.max() - _track3_type2.min() + 1)))
    #vocals_1to8_2 = np.zeros((vocals_delta.shape[0], int(_track4_type2.max() - _track4_type2.min() + 1)))
    #
    #drums_1to8_2 = drums_1to8.copy()
    #bass_1to8_2 = bass_1to8.copy()
    #other_1to8_2 = other_1to8.copy()
    #vocals_1to8_2 = vocals_1to8.copy()*2
    #
    #
    #
    #drums_1to8_2 = np.gradient(drums_1to8_2)
    #
    #bass_1to8_2 = np.gradient(bass_1to8_2)
    #
    #other_1to8_2 = np.gradient(other_1to8_2)
    #
    #vocals_1to8_2 = np.gradient(vocals_1to8_2)
    #
    #
    #drums_1to8_2 = (drums_1to8_2[0]+drums_1to8_2[1]).cumsum(axis=1)/(drums_1to8_2[0]+drums_1to8_2[1]).sum()
    #bass_1to8_2 = (bass_1to8_2[0]+bass_1to8_2[1]).cumsum(axis=1)/(bass_1to8_2[0]+bass_1to8_2[1]).sum()
    #other_1to8_2 = (other_1to8_2[0]+other_1to8_2[1]).cumsum(axis=1)/(other_1to8_2[0]+other_1to8_2[1]).sum()
    #vocals_1to8_2 = (vocals_1to8_2[0]+vocals_1to8_2[1]).cumsum(axis=1)/(vocals_1to8_2[0]+vocals_1to8_2[1]).sum()
    #
    #drums_1to8_2 *= (drums_1to8_2[0]+drums_1to8_2[0]).cumsum(axis=0)/(drums_1to8_2[0]+drums_1to8_2[1]).sum(axis=0)
    #bass_1to8_2 *= (bass_1to8_2[0]+bass_1to8_2[0]).cumsum(axis=0)/(bass_1to8_2[0]+bass_1to8_2[1]).sum(axis=0)
    #other_1to8_2 *= (other_1to8_2[0]+other_1to8_2[0]).cumsum(axis=0)/(other_1to8_2[0]+other_1to8_2[1]).sum(axis=0)
    #vocals_1to8_2 *= (vocals_1to8_2[0]+vocals_1to8_2[0]).cumsum(axis=0)/(vocals_1to8_2[0]+vocals_1to8_2[1]).sum(axis=0)
    #
    #drums_1to8_2 /= np.sqrt(1+np.abs(drums_1to8_2))
    #bass_1to8_2 /= np.sqrt(1+np.abs(bass_1to8_2))
    #other_1to8_2 /= np.sqrt(1+np.abs(other_1to8_2))
    #vocals_1to8_2 /= np.sqrt(1+np.abs(vocals_1to8_2))
    #
    #drums_1to8_2 /= ((drums_1to8_2).max()-drums_1to8_2.min())
    #bass_1to8_2 /= ((bass_1to8_2).max()-bass_1to8_2.min())
    #other_1to8_2 /= ((other_1to8_2).max()-other_1to8_2.min())
    #vocals_1to8_2 /= ((vocals_1to8_2).max()-vocals_1to8_2.min())
    #
    ##drums_1to8_2 += drums_1to8_2*drums_1to8.copy()
    ##bass_1to8_2 += bass_1to8_2*bass_1to8.copy()
    ##other_1to8_2 += other_1to8_2*other_1to8.copy()
    ##vocals_1to8_2 += vocals_1to8_2*vocals_1to8.copy()
    #
    #drums_1to8_2 /= np.power(1.0+np.abs(drums_1to8_2),4)
    #bass_1to8_2 /= np.power(1.0+np.abs(bass_1to8_2),4)
    #other_1to8_2 /= np.power(1.0+np.abs(other_1to8_2),4)
    #vocals_1to8_2 /= np.power(1.0+np.abs(other_1to8_2),4)
    #
    #drums_1to8_2 /= 1.0*((drums_1to8_2).max()-(drums_1to8_2).min())
    #bass_1to8_2 /= 1.0*((bass_1to8_2).max()-(bass_1to8_2).min())
    #other_1to8_2 /= 1.5*((other_1to8_2).max()-(other_1to8_2).min())
    #vocals_1to8_2 /= 1.0*((vocals_1to8_2).max()-(vocals_1to8_2).min())
    #
    #np.savetxt(track_path+"drums_beats_cd.csv", drums_1to8_2, delimiter=",")
    #np.savetxt(track_path+"bass_beats_cd.csv", bass_1to8_2, delimiter=",")
    #np.savetxt(track_path+"other_beats_cd.csv", other_1to8_2, delimiter=",")
    #np.savetxt(track_path+"vocals_beats_cd.csv", vocals_1to8_2, delimiter=",")
    
    _track1_type2 = np.loadtxt(track_path+'drums_beats_c2.csv', delimiter=',')
    _track2_type2 = np.loadtxt(track_path+'bass_beats_c2.csv', delimiter=',')
    _track3_type2 = np.loadtxt(track_path+'other_beats_c2.csv', delimiter=',')
    _track4_type2 = np.loadtxt(track_path+'vocals_beats_c2.csv', delimiter=',')
    
    #_track1_beats = np.loadtxt(track_path+'drums_beats_2.csv', delimiter=',')[:,0]
    #_track2_beats = np.loadtxt(track_path+'bass_beats_2.csv', delimiter=',')[:,0]
    #_track3_beats = np.loadtxt(track_path+'other_beats_2.csv', delimiter=',')[:,0]
    #_track4_beats = np.loadtxt(track_path+'vocals_beats_2.csv', delimiter=',')[:,0]
    
    _track1_beats = np.loadtxt(track_path+'drums_beats_2.csv', delimiter=',')[:,0]
    _track2_beats = np.loadtxt(track_path+'bass_beats_2.csv', delimiter=',')[:,0]
    _track3_beats = np.loadtxt(track_path+'other_beats_2.csv', delimiter=',')[:,0]
    _track4_beats = np.loadtxt(track_path+'vocals_beats_2.csv', delimiter=',')[:,0]
    
    track_length = signal_drums.shape[0]/signal_drums.sample_rate
    
    drums_1to8 = np.zeros((drums_delta.shape[0], int(_track1_type2.max() - _track1_type2.min() + 1)))
    bass_1to8 = np.zeros((bass_delta.shape[0], int(_track2_type2.max() - _track2_type2.min() + 1)))
    other_1to8 = np.zeros((other_delta.shape[0], int(_track3_type2.max() - _track3_type2.min() + 1)))
    vocals_1to8 = np.zeros((vocals_delta.shape[0], int(_track4_type2.max() - _track4_type2.min() + 1)))
    
    last_t = 0
    last_delta = 0
    
    for i in range(0, _track1_type2.shape[0]):
        current_t = _track1_beats[i]
        current_delta = current_t*(drums_delta.shape[0]/track_length)
        #print(current_t,current_delta)
        current_type = _track1_type2[i]
        drums_1to8[int(last_delta):int(current_delta), int(current_type)] = current_type
        #drums_1to8[int(last_delta):int(current_delta), int(current_type)] = i/10
        
        last_t = current_t
        last_delta = current_delta
        
    last_t = 0
    last_delta = 0
    
    for i in range(0, _track2_type2.shape[0]):
        current_t = _track2_beats[i]
        current_delta = current_t*(bass_delta.shape[0]/track_length)
        #print(current_t,current_delta)
        current_type = _track2_type2[i]
        bass_1to8[int(last_delta):int(current_delta), int(current_type)] = current_type
        #bass_1to8[int(last_delta):int(current_delta), int(current_type)] = i/10
        
        last_t = current_t
        last_delta = current_delta
        
    last_t = 0
    last_delta = 0
    
    for i in range(0, _track3_type2.shape[0]):
        current_t = _track3_beats[i]
        current_delta = current_t*(other_delta.shape[0]/track_length)
        #print(current_t,current_delta)
        current_type = _track3_type2[i]
        other_1to8[int(last_delta):int(current_delta), int(current_type)] = current_type
        #other_1to8[int(last_delta):int(current_delta), int(current_type)] = i/10
        
        last_t = current_t
        last_delta = current_delta
        
    
    last_t = 0
    last_delta = 0
    
    for i in range(0, _track4_type2.shape[0]):
        current_t = _track4_beats[i]
        current_delta = current_t*(vocals_delta.shape[0]/track_length)
        #print(current_t,current_delta)
        current_type = _track4_type2[i]
        vocals_1to8[int(last_delta):int(current_delta), int(current_type)] = current_type
        #vocals_1to8[int(last_delta):int(current_delta), int(current_type)] = i/10
        
        last_t = current_t
        last_delta = current_delta
        
    np.savetxt(track_path+"drums_beats_cd_2.csv", drums_1to8, delimiter=",")
    np.savetxt(track_path+"bass_beats_cd_2.csv", bass_1to8, delimiter=",")
    np.savetxt(track_path+"other_beats_cd_2.csv", other_1to8, delimiter=",")
    np.savetxt(track_path+"vocals_beats_cd_2.csv", vocals_1to8, delimiter=",")
    
    
    
    #drums_1to8_2 = np.zeros((drums_delta.shape[0], int(_track1_type2.max() - _track1_type2.min() + 1)))
    #bass_1to8_2 = np.zeros((bass_delta.shape[0], int(_track2_type2.max() - _track2_type2.min() + 1)))
    #other_1to8_2 = np.zeros((other_delta.shape[0], int(_track3_type2.max() - _track3_type2.min() + 1)))
    #vocals_1to8_2 = np.zeros((vocals_delta.shape[0], int(_track4_type2.max() - _track4_type2.min() + 1)))
    #
    #drums_1to8_2 = drums_1to8.copy()
    #bass_1to8_2 = bass_1to8.copy()
    #other_1to8_2 = other_1to8.copy()
    #vocals_1to8_2 = vocals_1to8.copy()*2
    #
    #
    #
    #drums_1to8_2 = np.gradient(drums_1to8_2)
    #
    #bass_1to8_2 = np.gradient(bass_1to8_2)
    #
    #other_1to8_2 = np.gradient(other_1to8_2)
    #
    #vocals_1to8_2 = np.gradient(vocals_1to8_2)
    #
    #
    #drums_1to8_2 = (drums_1to8_2[0]+drums_1to8_2[1]).cumsum(axis=1)/(drums_1to8_2[0]+drums_1to8_2[1]).sum()
    #bass_1to8_2 = (bass_1to8_2[0]+bass_1to8_2[1]).cumsum(axis=1)/(bass_1to8_2[0]+bass_1to8_2[1]).sum()
    #other_1to8_2 = (other_1to8_2[0]+other_1to8_2[1]).cumsum(axis=1)/(other_1to8_2[0]+other_1to8_2[1]).sum()
    #vocals_1to8_2 = (vocals_1to8_2[0]+vocals_1to8_2[1]).cumsum(axis=1)/(vocals_1to8_2[0]+vocals_1to8_2[1]).sum()
    #
    #drums_1to8_2 *= (drums_1to8_2[0]+drums_1to8_2[0]).cumsum(axis=0)/(drums_1to8_2[0]+drums_1to8_2[1]).sum(axis=0)
    #bass_1to8_2 *= (bass_1to8_2[0]+bass_1to8_2[0]).cumsum(axis=0)/(bass_1to8_2[0]+bass_1to8_2[1]).sum(axis=0)
    #other_1to8_2 *= (other_1to8_2[0]+other_1to8_2[0]).cumsum(axis=0)/(other_1to8_2[0]+other_1to8_2[1]).sum(axis=0)
    #vocals_1to8_2 *= (vocals_1to8_2[0]+vocals_1to8_2[0]).cumsum(axis=0)/(vocals_1to8_2[0]+vocals_1to8_2[1]).sum(axis=0)
    #
    #drums_1to8_2 /= np.sqrt(1+np.abs(drums_1to8_2))
    #bass_1to8_2 /= np.sqrt(1+np.abs(bass_1to8_2))
    #other_1to8_2 /= np.sqrt(1+np.abs(other_1to8_2))
    #vocals_1to8_2 /= np.sqrt(1+np.abs(vocals_1to8_2))
    #
    #drums_1to8_2 /= ((drums_1to8_2).max()-drums_1to8_2.min())
    #bass_1to8_2 /= ((bass_1to8_2).max()-bass_1to8_2.min())
    #other_1to8_2 /= ((other_1to8_2).max()-other_1to8_2.min())
    #vocals_1to8_2 /= ((vocals_1to8_2).max()-vocals_1to8_2.min())
    #
    ##drums_1to8_2 += drums_1to8_2*drums_1to8.copy()
    ##bass_1to8_2 += bass_1to8_2*bass_1to8.copy()
    ##other_1to8_2 += other_1to8_2*other_1to8.copy()
    ##vocals_1to8_2 += vocals_1to8_2*vocals_1to8.copy()
    #
    #drums_1to8_2 /= np.power(1.0+np.abs(drums_1to8_2),4)
    #bass_1to8_2 /= np.power(1.0+np.abs(bass_1to8_2),4)
    #other_1to8_2 /= np.power(1.0+np.abs(other_1to8_2),4)
    #vocals_1to8_2 /= np.power(1.0+np.abs(other_1to8_2),4)
    #
    #drums_1to8_2 /= 1.0*((drums_1to8_2).max()-(drums_1to8_2).min())
    #bass_1to8_2 /= 1.0*((bass_1to8_2).max()-(bass_1to8_2).min())
    #other_1to8_2 /= 1.5*((other_1to8_2).max()-(other_1to8_2).min())
    #vocals_1to8_2 /= 1.0*((vocals_1to8_2).max()-(vocals_1to8_2).min())
    #
    #np.savetxt(track_path+"drums_beats_cd_2.csv", drums_1to8_2, delimiter=",")
    #np.savetxt(track_path+"bass_beats_cd_2.csv", bass_1to8_2, delimiter=",")
    #np.savetxt(track_path+"other_beats_cd_2.csv", other_1to8_2, delimiter=",")
    #np.savetxt(track_path+"vocals_beats_cd_2.csv", vocals_1to8_2, delimiter=",")
    
    
    
    _track1_type = np.loadtxt(track_path+'drums_beats_cd.csv', delimiter=',')
    _track2_type = np.loadtxt(track_path+'bass_beats_cd.csv', delimiter=',')
    _track3_type = np.loadtxt(track_path+'other_beats_cd.csv', delimiter=',')
    _track4_type = np.loadtxt(track_path+'vocals_beats_cd.csv', delimiter=',')
    
    _track1_type2 = np.loadtxt(track_path+'drums_beats_cd_2.csv', delimiter=',')
    _track2_type2 = np.loadtxt(track_path+'bass_beats_cd_2.csv', delimiter=',')
    _track3_type2 = np.loadtxt(track_path+'other_beats_cd_2.csv', delimiter=',')
    _track4_type2 = np.loadtxt(track_path+'vocals_beats_cd_2.csv', delimiter=',')
    
    _track1_flux = np.loadtxt(track_path+'superflux_drums_norm.csv', delimiter=',')
    _track2_flux = np.loadtxt(track_path+'superflux_bass_norm.csv', delimiter=',')
    _track3_flux = np.loadtxt(track_path+'superflux_other_norm.csv', delimiter=',')
    _track4_flux = np.loadtxt(track_path+'superflux_vocals_norm.csv', delimiter=',')
    
    drums_1to8 = np.zeros((drums_delta.shape[0], int(_track1_type2.max() - _track1_type2.min() + 1)))
    bass_1to8 = np.zeros((bass_delta.shape[0], int(_track2_type2.max() - _track2_type2.min() + 1)))
    other_1to8 = np.zeros((other_delta.shape[0], int(_track3_type2.max() - _track3_type2.min() + 1)))
    vocals_1to8 = np.zeros((vocals_delta.shape[0], int(_track4_type2.max() - _track4_type2.min() + 1)))
    
    
    last_t = 0
    last_delta = 0
    
    for i in range(len(segmentations_drums[drums_1to8.shape[1]-1][0])):
        current_t = segmentations_drums[drums_1to8.shape[1]-1][0][i][1]
        current_delta = current_t*(drums_delta.shape[0]/track_length)
        #print(current_t,current_delta)
        current_type = int(segmentations_drums[drums_1to8.shape[1]-1][1][i])
        drums_1to8[int(last_delta):int(current_delta), int(current_type)] = current_type
        #drums_delta[int(last_delta):int(current_delta)]
        #drums_1to8[int(last_delta):int(current_delta), int(current_type)] = i/10
        
        last_t = current_t
        last_delta = current_delta
        
    last_t = 0
    last_delta = 0
    
    for i in range(len(segmentations_bass[bass_1to8.shape[1]-1][0])):
        current_t = segmentations_bass[bass_1to8.shape[1]-1][0][i][1]
        current_delta = current_t*(bass_delta.shape[0]/track_length)
        #print(current_t,current_delta)
        current_type = int(segmentations_bass[bass_1to8.shape[1]-1][1][i])
        bass_1to8[int(last_delta):int(current_delta), int(current_type)] = current_type
        #bass_delta[int(last_delta):int(current_delta)]
        #bass_1to8[int(last_delta):int(current_delta), int(current_type)] = i/10
        
        last_t = current_t
        last_delta = current_delta
        
    last_t = 0
    last_delta = 0
    
    for i in range(len(segmentations_other[other_1to8.shape[1]-1][0])):
        current_t = segmentations_other[other_1to8.shape[1]-1][0][i][1]
        current_delta = current_t*(other_delta.shape[0]/track_length)
        #print(current_t,current_delta)
        current_type = int(segmentations_other[other_1to8.shape[1]-1][1][i])
        other_1to8[int(last_delta):int(current_delta), int(current_type)] = current_type
        #other_delta[int(last_delta):int(current_delta)]
        #other_1to8[int(last_delta):int(current_delta), int(current_type)] = i/10
        
        last_t = current_t
        last_delta = current_delta
        
    
    last_t = 0
    last_delta = 0
    
    for i in range(len(segmentations_vocals[vocals_1to8.shape[1]-1][0])):
        current_t = segmentations_vocals[vocals_1to8.shape[1]-1][0][i][1]
        current_delta = current_t*(vocals_delta.shape[0]/track_length)
        #print(current_t,current_delta)
        current_type = int(segmentations_vocals[vocals_1to8.shape[1]-1][1][i])
        vocals_1to8[int(last_delta):int(current_delta), int(current_type)] = current_type
        #vocals_delta[int(last_delta):int(current_delta)]
        #vocals_1to8[int(last_delta):int(current_delta), int(current_type)] = i/10
        
        last_t = current_t
        last_delta = current_delta
    
    _track1_type3 = _track1_type + 10.0*_track1_type2
    _track2_type3 = _track2_type + 10.0*_track2_type2
    _track3_type3 = _track3_type + 10.0*_track3_type2
    _track4_type3 = _track4_type + 10.0*_track4_type2
    
    _track1_type3 /= 2*((_track1_type3).max())
    _track2_type3 /= 2*((_track2_type3).max())
    _track3_type3 /= 2*((_track3_type3).max())
    _track4_type3 /= 1.5*((_track4_type3).max())
    
    for i in range(4):
        _track1_type3[:,i] += ((1+_track1_type[:,i])*(1+drums_1to8[:,i])-1)*(1+ 0.4*(drums_delta * _track1_flux))
    
    for i in range(4):
        _track2_type3[:,i] += ((1+_track2_type[:,i])*(1+bass_1to8[:,i])-1)*(1+0.4*(bass_delta * _track2_flux))
        
    for i in range(4):
        _track3_type3[:,i] += ((1+_track3_type[:,i])*(1+other_1to8[:,i])-1)*(1+0.4*(other_delta * _track3_flux))
        
    for i in range(4):
        _track4_type3[:,i] += ((1+_track4_type[:,i])*(1+vocals_1to8[:,i])-1)*(1+0.4*(vocals_delta * _track4_flux))



    _track1_type3 = (_track1_type3 - _track1_type3.min())/(1+(_track1_type3).max())
    _track2_type3 = (_track2_type3 - _track2_type3.min())/(1+(_track2_type3).max())
    _track3_type3 = (_track3_type3 - _track3_type3.min())/(1+(_track3_type3).max())
    _track4_type3 = (_track4_type3 - _track4_type3.min())/(1+(_track4_type3).max())
    
    _track1_type3 = np.sqrt(_track1_type3)
    _track2_type3 = np.sqrt(_track2_type3)
    _track3_type3 = np.sqrt(_track3_type3)
    _track4_type3 = np.sqrt(_track4_type3)
    
    np.savetxt(track_path+"drums_beats_cd.csv", _track1_type3, delimiter=",")
    np.savetxt(track_path+"bass_beats_cd.csv", _track2_type3, delimiter=",")
    np.savetxt(track_path+"other_beats_cd.csv", _track3_type3, delimiter=",")
    np.savetxt(track_path+"vocals_beats_cd.csv", _track4_type3, delimiter=",")

    _track1_type = np.loadtxt(track_path+'drums_beats_cd.csv', delimiter=',')
    _track2_type = np.loadtxt(track_path+'bass_beats_cd.csv', delimiter=',')
    _track3_type = np.loadtxt(track_path+'other_beats_cd.csv', delimiter=',')
    _track4_type = np.loadtxt(track_path+'vocals_beats_cd.csv', delimiter=',')
    
    _track_type = np.hstack((_track1_type, _track2_type, _track3_type, _track4_type))
    
    c_method = FeatureAgglomeration(n_clusters=8)
    c_method = c_method.fit(_track_type)
    c_labs = c_method.transform(_track_type)
    
    c_method = AgglomerativeClustering(n_clusters=8, affinity='euclidean', linkage='complete', compute_full_tree = True)
    c_labs = c_method.fit_predict(c_labs)
    
    _track1_type_2 = _track1_type.copy()
    _track2_type_2 = _track2_type.copy()
    _track3_type_2 = _track3_type.copy()
    _track4_type_2 = _track4_type.copy()
    
    for i in range(_track_type.shape[0]):
        _track1_type_2[i] *= c_labs[i]
        _track2_type_2[i] *= c_labs[i]
        _track3_type_2[i] *= c_labs[i]
        _track4_type_2[i] *= c_labs[i]
        
    _track_type_2 = np.hstack((_track1_type_2, _track2_type_2, _track3_type_2, _track4_type_2))
    _track_type_2 = (c_labs*(_track_type_2).sum(axis=1))
    _track_type_2 /= (_track_type_2.max() - _track_type_2.min())
    
    np.savetxt(track_path+"intensity.csv", _track_type_2, delimiter=",")






    
    

    
    
if __name__ == "__main__":
    import madmom
    import numpy as np
    
    import os
    import sys    
    from sklearn.cluster import KMeans, AgglomerativeClustering, FeatureAgglomeration