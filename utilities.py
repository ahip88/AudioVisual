import numpy as np


def draw_Text_at(visual_, str_, loc_, ts_, color_, _type):
    s = str_
    if(_type == 0):
        visual_.data('glyph', np.array([ord(i) - 32 for i in s], dtype=np.uint16))
        visual_.data('pos', loc_)
        visual_.data('length', np.array([len(s)], dtype=np.uint32))
        visual_.data('color', color_)
        visual_.data('text_size', np.array([[ts_]], dtype=np.float32))
    else:
        visual_.append('glyph', np.array([ord(i) - 32 for i in s], dtype=np.uint16))
        visual_.append('pos', loc_)
        visual_.append('length', np.array([len(s)], dtype=np.uint32))
        visual_.append('color', color_)
        visual_.append('text_size', np.array([[ts_]], dtype=np.float32))
        
        
        
        
def draw_Dots_at(visual_, loc_, ms_, color_, _type):
    if(_type == 0):
        visual_.data('pos', loc_)
        visual_.data('ms', ms_)
        visual_.data('color', color_)
    else:
        visual_.append('pos', loc_)
        visual_.append('ms', ms_)
        visual_.append('color', color_)
        
        
        
def draw_Path(visual_, pos_, linewidth_, color_, _type):
    if(_type==0):
        visual_.data('linewidth', np.array([linewidth_]))
        visual_.data('pos', pos_)
        visual_.data('color', color_)
    else:
        visual_.append('linewidth', np.array([linewidth_]))
        visual_.append('pos', pos_)
        visual_.append('color', color_)