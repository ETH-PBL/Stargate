from nntool.api import NNGraph

from training_gate_navigator_TensorFlow import training_gate_navigator
from training_gate_classifier_PyTorch import training_gate_classifier
from nntool_gate_navigator_Tensorflow import validation_score_quantized_nav_model
from nntool_gate_classifier_PyTorch import quantize_classifier

if __name__ == "__main__":
 
    print("\n"+"#"*30)
    print("GATE_NAVIGATOR TRAINING")
    print("#"*30+"\n")
    
    training_gate_navigator()    
    validation_score_quantized_nav_model()

    print("\n"+"#"*30)
    print("GATE_CLASSIFIER TRAINING")
    print("#"*30+"\n")
    
    training_gate_classifier()
    quantize_classifier()

