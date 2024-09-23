'''
Copyright (c) 2021 Ariel University.
This is a slightly modified version, the original model can be found here:
https://github.com/ArielCyber/OSF-EIMTC/blob/main/src/EIMTC/modals/_lopez_protocol_header_fields_TDL.py
'''

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,GRU,Bidirectional,ReLU,Flatten, Dense
from tensorflow.keras.constraints import max_norm 
from ._utils import stack

class LopezTDLModality:
    '''
    Lopez-Protocol-header-Fields TDL deep learning nerual network uses for Packet Header Fields vector input.
    Classifciation type: Packet Header Fields
    Input: two-dimensional vector of number of packets and 3 features from each packet.

    Extraction plugin: ProtocolHeaderFieldsExtended
    
    Inspired by the paper: "Network Traffic Classifier With Convolutional and Recurrent Neural Networks for Internet of Things"
    '''
    def __init__(self,packet_count=32) -> None:
        input_layer_protocol_fields_modality = Input(shape=(packet_count,3), name='input_protocol_fields_TDL')
        self.model = Model(
            #name='Lopez protocol header fields TDL modality', #python gods don't like the spaces, replaced with underscore
            name='Lopez_protocol_header_fields_TDL_modality',
            inputs=input_layer_protocol_fields_modality,
            outputs=stack([
                input_layer_protocol_fields_modality,
                Bidirectional(GRU(64, return_sequences=True, kernel_constraint=max_norm(3))),
                ReLU(),
                Flatten(),
                Dense(3, activation='softmax'), #added by me
            ])
        )
