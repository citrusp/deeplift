import keras
from keras.engine.topology import Layer
from keras import backend as K
import numpy as np
import theano
from theano import tensor 
import sys
import h5py


def dense_conversion(layer, name, verbose,
                      dense_mxts_mode, nonlinear_mxts_mode, **kwargs):
    converted_activation = activation_conversion(
                                  layer, name=name, verbose=verbose,
                                  nonlinear_mxts_mode=nonlinear_mxts_mode) 
    to_return = [blobs.Dense(
                  name=("preact_" if len(converted_activation) > 0
                        else "")+name, 
                  verbose=verbose,
                  W=layer.get_weights()[0],
                  b=layer.get_weights()[1],
                  dense_mxts_mode=dense_mxts_mode)]
    to_return.extend(converted_activation)
    return to_return

class Dense(object):

    def __init__(self, W, b, inputlayer,reference ,dense_mxts_mode, **kwargs):
        super(Dense, self).__init__(**kwargs)
        self.W = W
        self.b = b
        self.inputlayer = inputlayer
        self.outputlayer = np.dot(self.W,inputlayer)+self.b
        self.reference = reference
        self.dense_mxts_mode = dense_mxts_mode
        self.reference_out = np.dot(self.W,reference)+self.b
   

    def get_yaml_compatible_object_kwargs(self):
        kwargs_dict = super(Dense, self).\
                       get_yaml_compatible_object_kwargs()
        kwargs_dict['W'] = self.W
        kwargs_dict['b'] = self.b
        return kwargs_dict

    def _compute_shape(self, input_shape):
        return (None, self.W.shape[1])

    def _build_activation_vars(self, input_act_vars):
        return B.dot(input_act_vars, self.W) + self.b

    def _get_output_diff_from_reference_vars(self):
        diff_ref = self.outputlayer - self.reference_out
        return diff_ref

    def _build_pos_and_neg_contribs(self):
        if (self.dense_mxts_mode == "a"):
            ##DenseMxtsMode.Linear): 
            outp_diff_ref = self._get_output_diff_from_reference_vars()
            pos_W = self.W*(self.W>0.0)
            neg_W = self.W*(self.W<0.0)
            pos_contribs = (np.dot(pos_W.T,outp_diff_ref)*(self.inputlayer>0.0))+np.dot(neg_W.T,outp_diff_ref)*(self.inputlayer<0.0)
            neg_contribs = (np.dot(pos_W.T,outp_diff_ref)*(self.inputlayer<0.0))+np.dot(neg_W.T,outp_diff_ref)*(self.inputlayer>0.0)
        elif (self.dense_mxts_mode == DenseMxtsMode.SepPosAndNeg):
            #compute pos/neg contribs based on the pos/neg breakdown
            #of the input, rather than just the sign of inp_diff_ref
            inp_pos_contribs, inp_neg_contribs =\
                self._get_input_pos_and_neg_contribs()
            pos_contribs = (T.dot(inp_pos_contribs, self.W*(self.W>=0.0))+
                            T.dot(inp_neg_contribs, self.W*(self.W<0.0))) 
            neg_contribs = (T.dot(inp_neg_contribs, self.W*(self.W>=0.0))+
                            T.dot(inp_pos_contribs, self.W*(self.W<0.0))) 
        else:
            raise RuntimeError("Unsupported dense_mxts_mode: "+
                               self.dense_mxts_mode)
        return pos_contribs, neg_contribs

    def _get_mxts_increments_for_inputs(self):
        if (self.dense_mxts_mode == DenseMxtsMode.Linear): 
            #different inputs will inherit multipliers differently according
            #to the sign of inp_diff_ref (as this sign was used to determine
            #the pos_contribs and neg_contribs; there was no breakdown
            #by the pos/neg contribs of the input)
            inp_diff_ref = self._get_input_diff_from_reference_vars() 
            pos_inp_mask = inp_diff_ref > 0.0
            neg_inp_mask = inp_diff_ref < 0.0
            zero_inp_mask = T.eq(inp_diff_ref, 0.0)
            inp_mxts_increments = pos_inp_mask*(
                                    T.dot(self.get_pos_mxts(),
                                        self.W.T*(self.W.T>=0.0)) 
                                   +T.dot(self.get_neg_mxts(),
                                        self.W.T*(self.W.T<0.0)))
            inp_mxts_increments += neg_inp_mask*(
                                    T.dot(self.get_pos_mxts(),
                                        self.W.T*(self.W.T<0.0)) 
                                   +T.dot(self.get_neg_mxts(),
                                        self.W.T*(self.W.T>=0.0)))
            inp_mxts_increments += zero_inp_mask*T.dot(
                                   0.5*(self.get_pos_mxts()
                                        +self.get_neg_mxts()),self.W.T)
            #pos_mxts and neg_mxts in the input get the same multiplier
            #because the breakdown between pos and neg wasn't used to
            #compute pos_contribs and neg_contribs in the forward pass
            #(it was based entirely on inp_diff_ref)
            return inp_mxts_increments, inp_mxts_increments

        elif (self.dense_mxts_mode == DenseMxtsMode.SepPosAndNeg):
            #during the forward pass, the pos/neg contribs of the input
            #were used to determing the pos/neg contribs of the output - thus
            #during the backward pass, the pos/neg mxts will be determined
            #accordingly (i.e. for a given input, the multiplier on the
            #positive part may be different from the multiplier on the
            #negative part)
            pos_mxts_increments = (B.dot(self.get_pos_mxts(),
                                        self.W.T*(self.W.T>=0.0))
                                   +B.dot(self.get_neg_mxts(),
                                        self.W.T*(self.W.T<0.0)))
            neg_mxts_increments = (B.dot(self.get_pos_mxts(),
                                        self.W.T*(self.W.T<0.0))
                                   +B.dot(self.get_neg_mxts(),
                                        self.W.T*(self.W.T>=0.0)))
            return pos_mxts_increments, neg_mxts_increments
        else:
            raise RuntimeError("Unsupported mxts mode: "
                               +str(self.dense_mxts_mode))

def simple_deeplift(layer_para, data, reference, nb_classes):
    denselayer = Dense(W = layer_para, b = 0, inputlayer = data ,reference = reference, dense_mxts_mode ="a")
    ##layer should contain W and b
    ##DenseMxtsMode.Linear)
    pos_contribs , neg_contribs = denselayer._build_pos_and_neg_contribs()
    total_contribs = pos_contribs + neg_contribs
    print (total_contribs)
    return total_contribs






modellayerfirst = np.array([1,0.5,-1,0,0,0])
modellayerfirst = modellayerfirst.reshape((2,3))
data_a = np.array([[1,1,1],[2,2,2],[3,3,3],[4,4,4],[5,5,5],[6,6,6],[7,7,7],[8,8,8],[9,9,9],[10,10,10]])
data_a = data_a.T
reference_a = np.ones((10,3))
reference_a = reference_a.T
contrib_ref1 = simple_deeplift(layer_para=modellayerfirst, data=data_a, reference=reference_a, nb_classes=1)

#f = open("/home/gongy/ref1.txt", "w")
#contrib_ref1 = f.write
#f.close()
