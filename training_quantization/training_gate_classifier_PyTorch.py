"""
This script is used to train the weights of the GateClassifier network
Adapted from https://github.com/pulp-platform/pulp-dronet/blob/master/pulp-dronet-v2/training.py

author: Konstantin Kalenberg
"""

# essentials
import configparser
from tqdm import tqdm
import os
import wandb
from datetime import datetime
import numpy as np

# torch
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
from torchinfo import summary
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score, AUROC

# imav challenge
from utility import ImavChallengeClassificationDataset, custom_bce_loss, custom_accuracy_loss, custom_f1_loss, custom_auroc
from models.gate_classifier_PyTorch_model import GateClassifier


def training(training_loader, validation_loader,config):

     # Load parameters
    verbose = config.getboolean("TRAINING_CLASSIFICATION", "VERBOSE")
    use_wandb = config.getboolean("WANDB", "USE_WANDB")
    batch_size = int(config["TRAINING_CLASSIFICATION"]["BATCH_SIZE"])
    num_channels_start = int(config["TRAINING_CLASSIFICATION"]["NUM_CHANNELS_START"])
    lr = float(config["TRAINING_CLASSIFICATION"]["LEARNING_RATE"])
    lr_decay = float(config["TRAINING_CLASSIFICATION"]["LEARNING_RATE_DECAY"])
    epochs = int(config["TRAINING_CLASSIFICATION"]["EPOCHS"])
    dropout_p = float(config["TRAINING_CLASSIFICATION"]["DROPOUT_PROB"])
    data_loading_path_classification = "../" + config["DATA_PATHS"]["DATA_LOADING_PATH_CLASSIFICATION"]

    # Unique id to save best model of this run
    unique_run_id = str(datetime.now())

    # Set up weights and biases
    if use_wandb:
        
        wandb.init(project="StarGate", group="")

        # Log important parameters to weights and biases
        wandb.config.data_path = data_loading_path_classification
        wandb.config.batch_size = batch_size
        wandb.config.num_channels_start = num_channels_start
        wandb.config.lr = lr
        wandb.config.epochs = epochs
        wandb.config.dropout_p = dropout_p

    # Select CPU or GPU as device
    #os.environ['CUDA_VISIBLE_DEVICES'] = 0,1,2,3
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("PyTorch found the following device:", device)
    #print("pyTorch version:", torch.__version__)

    # Create gate classifier model
    gate_classifier_model = GateClassifier(num_channels_start=num_channels_start, dropout_p=dropout_p)
    gate_classifier_model.to(device)

    # Create optimizer for training
    optimizer = optim.Adam(params=gate_classifier_model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=lr_decay,
                           amsgrad=False)

    # Create custom loss fct
    bce_loss = torch.nn.BCELoss().to(device)
    binary_accuracy = BinaryAccuracy().to(device)
    binary_f1_loss = BinaryF1Score().to(device)
    binary_auroc = AUROC(task='binary').to(device)

    # Print model and optimizer information
    if verbose:
        print("Gate classifier's state_dict:")
        for param_tensor in gate_classifier_model.state_dict():
            print(param_tensor, "\t\t\t", gate_classifier_model.state_dict()[param_tensor].size())
        print("Optimizer's state_dict:")
        for var_name in optimizer.state_dict():
            print(var_name, "\t", optimizer.state_dict()[var_name])

        # Print summary
        print('Gate classifier summary:')
        summary(gate_classifier_model, [(batch_size, 1, 168, 168), (batch_size, 1, 21, 21)])

    ############################################################################
    # Train/Val Loop
    ############################################################################
    # Used for saving best model of this run
    best_val_loss = float('inf')

    # Used to save the best ROC AUC and confusion matrix curve to wandb
    best_train_labels = list()
    best_train_preds = list()
    best_val_labels = list()
    best_val_preds = list()

    for epoch in range(epochs):
        print('\nEpoch: ', epoch+1, ' / ', epochs)

        # Average train losses
        train_bce = 0.0
        train_accuracy = 0.0
        train_f1 = 0.0

        # ROC AUC curve values
        train_labels = list()
        train_preds = list()

        # Training
        num_train_iterations = len(training_loader)
        gate_classifier_model.train()
        with tqdm(total=num_train_iterations, desc='Train', disable=not True) as t:
            for batch_idx, data in enumerate(training_loader):
                # Load data
                image, tof, label = data[0].to(device), data[1].to(device), data[2].to(device)

                # Forward pass
                optimizer.zero_grad()
                pred = gate_classifier_model(image, tof)

                # Loss computation
                loss_bce = custom_bce_loss(label, pred, bce_loss)
                loss_accuracy = custom_accuracy_loss(label, pred, binary_accuracy)
                loss_f1 = custom_f1_loss(label, pred, binary_f1_loss)

                # Save label and pred for ROC AUC curve and confusion matrix
                train_labels.extend(label.tolist())  # squeeze not needed as dim is [batch_size]
                train_preds.extend(torch.squeeze(pred).tolist())  # squeeze needed as dim is [batch_size, 1]

                # Save avg train mse losses
                train_bce += loss_bce.item()
                train_accuracy += loss_accuracy.item()
                train_f1 += loss_f1.item()

                # Backward pass
                loss_bce.backward()
                optimizer.step()
                t.update(1)

        # Average validation losses
        val_bce = 0.0
        val_accuracy = 0.0
        val_f1 = 0.0

        # ROC AUC curve values
        val_labels = list()
        val_preds = list()

        # Validation
        num_val_iterations = len(validation_loader)
        gate_classifier_model.eval()
        with torch.no_grad():
            for batch_idx, data in enumerate(validation_loader):
                # Load data
                image, tof, label = data[0].to(device), data[1].to(device), data[2].to(device)

                # Forward pass
                pred = gate_classifier_model(image, tof)

                # Loss computation
                loss_bce = custom_bce_loss(label, pred, bce_loss)
                loss_accuracy = custom_accuracy_loss(label, pred, binary_accuracy)
                loss_f1 = custom_f1_loss(label, pred, binary_f1_loss)

                # Save label and pred for ROC AUC curve and confusion matrix
                val_labels.extend(label.tolist())  # squeeze not needed as dim is [batch_size]
                val_preds.extend(torch.squeeze(pred).tolist())  # squeeze needed as dim is [batch_size, 1]

                # Save avg val losses
                val_bce += loss_bce.item()
                val_accuracy += loss_accuracy.item()
                val_f1 += loss_f1.item()

        # Compute average losses at end of epoch
        train_bce /= num_train_iterations
        train_accuracy /= num_train_iterations
        train_f1 /= num_train_iterations
        val_bce /= num_val_iterations
        val_accuracy /= num_val_iterations
        val_f1 /= num_val_iterations

        # Compute AUROC train/val for this epoch
        train_auroc = custom_auroc(torch.Tensor(train_labels), torch.Tensor(train_preds), binary_auroc).item()
        val_auroc = custom_auroc(torch.Tensor(val_labels), torch.Tensor(val_preds), binary_auroc).item()

        # Save model if better val loss than previous best val loss --> used later to upload to wandb
        if val_bce < best_val_loss:
            # Remove previous best if it exists
            if os.path.exists('throwaway_models/best_model_gate_classifier' + unique_run_id + '.pt'):
                print('Deleted model dict')
                os.remove('throwaway_models/best_model_gate_classifier' + unique_run_id + '.pt')

            # Save
            torch.save({'epoch': epoch, 'num_channels_start': num_channels_start, 'dropout_p': dropout_p,
                        'gate_classifier_state_dict': gate_classifier_model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                        'best_val_bce': val_bce, 'best_val_accuracy': val_accuracy, 'best_val_f1': val_f1, 'best_val_auroc': val_auroc},
                        'throwaway_models/best_model_gate_classifier' + unique_run_id + '.pt')
            print('Saved new best model')

            # Save best label and pred for ROC AUC curve and confusion matrix
            best_train_labels = train_labels
            best_train_preds = train_preds
            best_val_labels = val_labels
            best_val_preds = val_preds

            # Update new best loss
            best_val_loss = val_bce


        print('\n')
        print('Avg BCE Loss Train / Val: ', train_bce, ' / ', val_bce)
        print('Avg Accuracy Loss Train / Val: ', train_accuracy, ' / ', val_accuracy)
        print('Avg F1 Loss Val: ', train_f1, ' / ', val_f1)
        print('Avg AUROC Loss Val: ', train_auroc, ' / ', val_auroc)
        print('-------------------------------------------------------------')

        if use_wandb:
            wandb.watch(gate_classifier_model, loss_bce, log="all", log_freq=500)
            wandb.log({'epoch': epoch, 'train_bce': train_bce, 'val_bce': val_bce, 'train_accuracy': train_accuracy,
                       'val_accuracy': val_accuracy, 'train_f1': train_f1, 'val_f1': val_f1, 'train_auroc': train_auroc,
                       'val_auroc': val_auroc})
    if use_wandb:
        # Log best model to wandb artifact
        artifact_full_state_dict = wandb.Artifact('best_model_gate_classifier', type='full_state_dict')
        artifact_full_state_dict.add_file('throwaway_models/best_model_gate_classifier' + unique_run_id + '.pt')
        wandb.log_artifact(artifact_full_state_dict).wait()

        # Get the version of the saved artifact
        artifact_version = artifact_full_state_dict.version

        # Log ROC AUC curve of best model to wandb
        best_train_preds = [[1 - element, element] for element in best_train_preds]  # Need both probabilities for pos and neg class
        best_val_preds = [[1 - element, element] for element in best_val_preds]  # Need both probabilities for pos and neg class
        wandb.log({"train_roc_curve": wandb.plot.roc_curve(best_train_labels, best_train_preds)})
        wandb.log({"val_roc_curve": wandb.plot.roc_curve(best_val_labels, best_val_preds)})

        # Log confusion matrix of best model to wandb
        best_train_preds = np.array(best_train_preds)
        best_val_preds = np.array(best_val_preds)
        wandb.log({"train_confusion_mat": wandb.plot.confusion_matrix(y_true=best_train_labels, probs=best_train_preds,
                                                                      class_names=["gate", "no_gate"])})
        wandb.log({"val_confusion_mat": wandb.plot.confusion_matrix(y_true=best_val_labels, probs=best_val_preds,
                                                                    class_names=["gate", "no_gate"])})
        wandb.finish()

    best_model = GateClassifier(num_channels_start=num_channels_start, dropout_p=dropout_p)
    best_model.load_state_dict(torch.load('throwaway_models/best_model_gate_classifier' + unique_run_id + '.pt')['gate_classifier_state_dict'])
    return best_model,artifact_version


def process_loaders(config):
    # Load parameters
    data_loading_path = os.getcwd() + "/../" + config["DATA_PATHS"]["DATA_LOADING_PATH_CLASSIFICATION"]   
    num_workers = int(config["TRAINING_CLASSIFICATION"]["NUM_WORKERS"])
    batch_size = int(config["TRAINING_CLASSIFICATION"]["BATCH_SIZE"])    
    mean_image = float(config["NORMALIZATION"]["MEAN_IMAGE"])
    std_image = float(config["NORMALIZATION"]["STD_IMAGE"])
    mean_tof = float(config["NORMALIZATION"]["MEAN_TOF"])
    std_tof = float(config["NORMALIZATION"]["STD_TOF"])
    
    # Create transforms to be applied in dataloader
    standartizer_image = transforms.Normalize(mean=[mean_image], std=[std_image])  # Determined from: from utility import batch_mean_and_sd
    standartizer_tof = transforms.Normalize(mean=[mean_tof], std=[std_tof])  # Determined from: from utility import batch_mean_and_sd
    color_jitter_image = transforms.ColorJitter(brightness=(0.5, 2.0), contrast=(0.5, 2.0))
    gaussian_blur_image = transforms.GaussianBlur(kernel_size=(3, 5))
    random_invert_image = transforms.RandomInvert()
    random_adjust_sharpness_image = transforms.RandomAdjustSharpness(sharpness_factor=10)

    transforms_image_train = [transforms.ToTensor(), color_jitter_image, gaussian_blur_image, random_invert_image,
                              random_adjust_sharpness_image, standartizer_image]
    transforms_image_val = [transforms.ToTensor(), standartizer_image]
    transforms_tof = [transforms.ToTensor(), standartizer_tof]

    # Create dataloader for training and validation dataset
    normalizer_image = transforms.Normalize(mean=[mean_image], std=[std_image])  # Determined from: from utility import batch_mean_and_sd
    normalizer_tof = transforms.Normalize(mean=[mean_tof], std=[std_tof])    # Determined from: from utility import batch_mean_and_sd

    train_dataset = ImavChallengeClassificationDataset(root=data_loading_path+'training/', transforms_image=transforms_image_train,
                                                       transforms_tof=transforms_tof)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    validation_dataset = ImavChallengeClassificationDataset(root=data_loading_path+'validation/', transforms_image=transforms_image_val,
                                                            transforms_tof=transforms_tof)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, validation_loader


def convert_pretrained_pytorch_to_onnx(trained_model, artifact_version,config):
   
    

    # Export model to onnx format
    trained_model.eval()
   
    dummy_input_camera = torch.rand(1, 1, 168, 168)
    dummy_input_tof = torch.rand(1, 1, 21, 21)

    input_names = ['image', 'tof']
    output_names = ['output']

    file_path_saved_model = 'onnx_models/' + 'gate_classifier_model_'+ artifact_version + '.onnx'
    torch.onnx.export(trained_model, (dummy_input_camera, dummy_input_tof),
                      file_path_saved_model,
                      input_names=input_names, output_names=output_names, export_params=True)

    print("Model .onnx saved in: ", file_path_saved_model)
    config.set("QUANTIZATION_CLASSIFICATION","MODEL_IDENTIFIER",artifact_version)
    with open('deep_learning_config.ini', 'w') as configfile:
        config.write(configfile)


def training_gate_classifier():
    # Load config file
    config = configparser.ConfigParser(inline_comment_prefixes="#")
    config.read("deep_learning_config.ini")

    training_loader, validation_loader = process_loaders(config=config)

    print("\n\n#Training using classic method\n\n")
    model, artifact_version = training(training_loader=training_loader, validation_loader=validation_loader, config=config)

    print("\n\n#ONNX export\n\n")
    convert_pretrained_pytorch_to_onnx(trained_model=model,artifact_version=artifact_version, config=config)

