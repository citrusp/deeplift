import keras
from keras.engine.topology import Layer
from keras import backend as K
import numpy as np
import theano
from theano import tensor 
import sys
import h5py

def sigmoid(x):                                        
   return 1/(1 + np.exp(-x))


class Dense(object):

    def __init__(self, W, b, inputlayer,reference ,dense_mxts_mode,activation_mode, **kwargs):
        super(Dense, self).__init__(**kwargs)
        self.W = W
        self.b = b
        self.inputlayer = inputlayer
        self.activation_mode = activation_mode
        self.reference = reference
        self.dense_mxts_mode = dense_mxts_mode
        
   

    def _compute_shape(self, input_shape):
        return (None, self.W.shape[1])

    def _build_activation_vars(self, input_act_vars):
        return np.dot(self.W, input_act_vars) + self.b

    def _get_input_diff_from_reference_vars(self):
        diff_ref = self.inputlayer - self.reference
        return diff_ref

    def _build_pos_and_neg_contribs(self):
        if (self.dense_mxts_mode == "a"):
            ##DenseMxtsMode.Linear): 
            inp_diff_ref = self._get_input_diff_from_reference_vars()
            pos_contribs = np.dot(self.W,self.inputlayer)*((np.dot(self.W,self.inputlayer))>=0.0)
            neg_contribs = np.dot(self.W,self.inputlayer)*((np.dot(self.W,self.inputlayer))<0.0)
           
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


    def _activation_contribution(self):
      linear_outp = self._build_activation_vars(self.inputlayer)
      linear_ref_outp = self._build_activation_vars(self.reference)
     
      if (self.activation_mode =="sigmoid"):
                                    #relu='relu',
                                     #prelu='prelu',
                                     #sigmoid='sigmoid',
                                     #softmax='softmax',
                                     #linear='linear'
            
            if (self.dense_mxts_mode == "a"):
            ##DenseMxtsMode.Linear):
                M_activation = (sigmoid(linear_outp)-sigmoid(linear_ref_outp))/(linear_outp-linear_ref_outp)
            elif (self.dense_mxts_mode == DenseMxtsMode.SepPosAndNeg):
                xplus = (linear_outp-linear_ref_outp)*((linear_outp-linear_ref_outp)>=0.0)
                xminus = (linear_outp-linear_ref_outp)*((linear_outp-linear_ref_outp)<0.0)
                yplus = 0.5*(sigmoid(linear_ref_outp+xplus)-sigmoid(linear_ref_outp))+0.5*(sigmoid(linear_ref_outp+xplus+xminus)-sigmoid(linear_ref_outp+xminus))
                yminus = 0.5*(sigmoid(linear_ref_outp+xminus)-sigmoid(linear_ref_outp))+0.5*(sigmoid(linear_ref_outp+xplus+xminus)-sigmoid(linear_ref_outp+xplus))
                M_xplus_yplus = yplus/xplus
                M_xminus_yminus = yminus/xminus
            else:
                raise RuntimeError("Unsupported dense_mxts_mode: "+
                                   self.dense_mxts_mode)
       

         
      elif (self.activation_mode == "linear"):
                                    #relu='relu',
                                     #prelu='prelu',
                                     #sigmoid='sigmoid',
                                     #softmax='softmax',
                                     #linear='linear'
            
            if (self.dense_mxts_mode == "a"):
               ##DenseMxtsMode.Linear):
                   linear_outp = _build_activation_vars(inputlayer)
                   linear_ref_outp = _build_activation_vars(reference)
                   M_activation = (sigmoid(linear_outp)-sigmoid(linear_ref_outp))/(linear_outp-linear_ref_outp)
                   
            elif (self.dense_mxts_mode == DenseMxtsMode.SepPosAndNeg):
                   xplus = (linear_outp-linear_ref_outp)*((linear_outp-linear_ref_outp)>=0.0)
                   xminus = (linear_outp-linear_ref_outp)*((linear_outp-linear_ref_outp)<0.0)
                   yplus = 0.5*(sigmoid(linear_ref_outp+xplus)-sigmoid(linear_ref_outp))+0.5*(sigmoid(linear_ref_outp+xplus+xminus)-sigmoid(linear_ref_outp+xminus))
                   yminus = 0.5*(sigmoid(linear_ref_outp+xminus)-sigmoid(linear_ref_outp))+0.5*(sigmoid(linear_ref_outp+xplus+xminus)-sigmoid(linear_ref_outp+xplus))
                   M_xplus_yplus = yplus/xplus
                   M_xminus_yminus = yminus/xminus
            else:
                  raise RuntimeError("Unsupported dense_mxts_mode: "+
                                      self.dense_mxts_mode)
      else:
            raise RuntimeError("Unsupported activation_mode: "+
                                  self.activation_mode)

      if (self.dense_mxts_mode == "a"):
            return M_activation 
      elif (self.dense_mxts_mode == DenseMxtsMode.SepPosAndNeg):
            return M_xplus_yplus ,M_xminus_yminus
      else:
            raise RuntimeError("Unsupported dense_mxts_mode: "+
                                   self.dense_mxts_mode)

   def _total_layer_contribution(self):
      if (denselayer.dense_mxts_mode == "a"):
         acti_contribs = denselayer._activation_contribution()
      elif (denselayer.dense_mxts_mode == DenseMxtsMode.SepPosAndNeg):
         acti_contribs_plus, acti_contribs_minus = denselayer._activation_contribution()

      M_layer = np.dot(self.W,acti_contribs)
      return M_layer          



def simple_deeplift(layer_para, data, reference, nb_classes):
   denselayer = Dense(W = layer_para, b = 0, inputlayer = data ,reference = reference, dense_mxts_mode ="a", activation_mode="sigmoid")
   ##layer should contain W and b
   ##DenseMxtsMode.Linear)
   pos_contribs , neg_contribs = denselayer._build_pos_and_neg_contribs()
   total_contribs = pos_contribs + neg_contribs

   if (denselayer.dense_mxts_mode == "a"):
      acti_contribs = denselayer._activation_contribution()
   elif (denselayer.dense_mxts_mode == DenseMxtsMode.SepPosAndNeg):
      acti_contribs_plus, acti_contribs_minus = denselayer._activation_contribution
   #print (total_contribs)
   print (acti_contribs)
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
