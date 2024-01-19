from training_gate_navigator_TensorFlow import training_gate_navigator
from training_gate_classifier_PyTorch import training_gate_classifier

if __name__ == "__main__":
 
    print("\n"+"#"*30)
    print("GATE_NAVIGATOR TRAINING")
    print("#"*30+"\n")
    
    training_gate_navigator()
    
    
    print("\n"+"#"*30)
    print("GATE_CLASSIFIER TRAINING")
    print("#"*30+"\n")
    
    training_gate_classifier()