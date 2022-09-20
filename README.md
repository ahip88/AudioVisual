<h1>
  <a href="#"><img alt="ahiP" src="Cave-4.0-bg.jpg" width="100%"/></a>
</h1>

# AudioVisual

Testbed for deep-generative AudioVisuals


### Dependencies
* Python 3.8
* Datoviz
* madmom
* carat
* musicsections
* screeninfo
* audioio
* sklearn

## Getting Started
Install Datoviz by following https://datoviz.org/tutorials/install/ or

For Linux:
```
 pip install http://dl.datoviz.org/v0.1.0-alpha.1/datoviz-0.1.0a1-cp38-cp38-manylinux_2_24_x86_64.whl
```

For macOS:
```
 pip install http://dl.datoviz.org/v0.1.0-alpha.1/datoviz-0.1.0a1-cp38-cp38-macosx_10_14_x86_64.whl
```

For Windows:
```
 pip install http://dl.datoviz.org/v0.1.0-alpha.1/datoviz-0.1.0a1-cp38-cp38-win_amd64.whl
```



followed by 
```
git clone https://github.com/ahip88/AudioVisual.git
cd AudioVisual
pip install -r requirements.txt
```

### Executing Code

* To run the tutorial:
```
python tutorial.py When_They_Come_For_Me
```

* For the AudioVisual only:
```
python audio_visual_v5.py sync_test
```

* To process your own songs, place them in AudioVisual Folder and run:
```
python Process.py File
```
Do Not include the extension, it will be searched for.

## Issues
If it's lagging (fps < 300) you're probably being spammed vklite validation warnings.
Follow the instructions to build datoviz from Source (updating anything Vulkan related)


## Authors

ahip88
ahi.p88@gmail.com

## Version History

    * Demo/Tutorial



## Acknowledgments

* [Datoviz](https://datoviz.org/)
* [madmom](https://github.com/CPJKU/madmom)
* [carat](https://github.com/mrocamora/carat)
* [musicsections](https://github.com/justinsalamon/musicseg_deepemb)
* [demucs](https://github.com/facebookresearch/demucs)
