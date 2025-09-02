import torch
import torch.nn as nn
import torch.optim as optim
import sys
from tqdm import tqdm
sys.path.append('src')
from utils2 import *



class M1(nn.Module):
    def __init__(self, layer_dims):
        """
        Initializes the M1 model with the specified layer dimensions.
        Args:
            layer_dims (list of int): A list where each element represents the number 
                of neurons in the corresponding layer of the neural network. The length 
                of the list determines the number of layers, including input and output layers.
        Attributes:
            model (nn.Sequential): A sequential container of layers, including linear 
                transformations and ReLU activations for hidden layers.
        Notes:
            - The weights of the linear layers are initialized using He initialization 
              (kaiming_normal_) to improve convergence in ReLU networks.
            - The biases of the linear layers are initialized to zero.
        """
        super(M1, self).__init__()
        
        layers = []
        for i in range(1, len(layer_dims)):
            layers.append(nn.Linear(layer_dims[i-1], layer_dims[i]))
            if i < len(layer_dims) - 1:
                layers.append(nn.ReLU())
        
        self.model = nn.Sequential(*layers)
        
        # initialize weights using He initialization for ReLU networks
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        Defines the forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor to the model.

        Returns:
            torch.Tensor: Output tensor after applying the model and softmax activation 
                          along the specified dimension (dim=1).
        """
        logits = self.model(x)
        return torch.softmax(logits, dim=1)



def train_model_adam_pytorch(X_train, y_train, X_val, y_val, layer_dims, 
                            learning_rate=0.001, num_epochs=50, print_every=10,
                            beta1=0.9, beta2=0.999, epsilon=1e-8):
    """
    Trains a PyTorch model using the Adam optimizer and cross-entropy loss.
    Args:
        X_train (numpy.ndarray): Training input data.
        y_train (numpy.ndarray): Training target labels.
        X_val (numpy.ndarray): Validation input data.
        y_val (numpy.ndarray): Validation target labels.
        layer_dims (list): List of integers defining the dimensions of each layer in the model.
        learning_rate (float, optional): Learning rate for the Adam optimizer. Default is 0.001.
        num_epochs (int, optional): Number of epochs to train the model. Default is 50.
        print_every (int, optional): Frequency (in epochs) to print training and validation costs. Default is 10.
        beta1 (float, optional): Exponential decay rate for the first moment estimates in Adam. Default is 0.9.
        beta2 (float, optional): Exponential decay rate for the second moment estimates in Adam. Default is 0.999.
        epsilon (float, optional): Small constant for numerical stability in Adam. Default is 1e-8.
    Returns:
        tuple: A tuple containing:
            - model (torch.nn.Module): The trained PyTorch model.
            - train_costs (list): List of training costs (loss values) for each epoch.
            - val_costs (list): List of validation costs (loss values) for each epoch.
    """
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val)
    
    model = M1(layer_dims)
    
    criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, 
                          betas=(beta1, beta2), eps=epsilon)
    
    train_costs = []
    val_costs = []
    

    for epoch in tqdm(range(1, num_epochs + 1), desc="Training Epochs", ncols=100, unit="epoch"):
        model.train()
        
        # forward pass
        outputs = model(X_train_tensor)
        loss = criterion(torch.log(outputs + 1e-8), torch.argmax(y_train_tensor, dim=1))
        
        # backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # calculate train and validation loss
        model.eval()
        with torch.no_grad():
            # training metrics
            outputs_train = model(X_train_tensor)
            train_loss = criterion(torch.log(outputs_train + 1e-8), torch.argmax(y_train_tensor, dim=1))
            train_costs.append(train_loss.item())
            
            # validation metrics
            outputs_val = model(X_val_tensor)
            val_loss = criterion(torch.log(outputs_val + 1e-8), torch.argmax(y_val_tensor, dim=1))
            val_costs.append(val_loss.item())
        
        if print_every and epoch % print_every == 0:
            print(f"Epoch {epoch}: Train cost = {train_loss.item():.4f}, Val cost = {val_loss.item():.4f}")
    
    return model, train_costs, val_costs


def report_metrics_pytorch(model, X, y, class_names=None, plot=True):
    """
    Evaluates a PyTorch model's performance on a given dataset and reports metrics.
    Args:
        model (torch.nn.Module): The PyTorch model to evaluate.
        X (array-like): Input features, typically a NumPy array or similar structure.
        y (array-like): True labels, typically a NumPy array or similar structure.
        class_names (list, optional): List of class names for the confusion matrix. Defaults to None.
        plot (bool, optional): Whether to plot the confusion matrix. Defaults to True.
    Returns:
        tuple: A tuple containing:
            - accuracy (float): The accuracy of the model on the given dataset.
            - cross_entropy (float): The cross-entropy loss of the model on the given dataset.
    Notes:
        - The function assumes that `y` is one-hot encoded.
        - The confusion matrix is plotted only if `plot` is set to True and `class_names` is provided.
        - Adds a small constant (1e-8) to the outputs to avoid numerical instability when taking the logarithm.
    """
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y)
    
    model.eval()
    with torch.no_grad():
        outputs = model(X_tensor)
        predicted_classes = torch.argmax(outputs, dim=1)
        true_classes = torch.argmax(y_tensor, dim=1)
        
        accuracy = (predicted_classes == true_classes).float().mean().item()
        
        criterion = nn.CrossEntropyLoss()
        cross_entropy = criterion(torch.log(outputs + 1e-8), true_classes).item()
        
        if plot:
            plot_confusion_matrix(true_classes.numpy(), predicted_classes.numpy(), class_names)

    return accuracy, cross_entropy