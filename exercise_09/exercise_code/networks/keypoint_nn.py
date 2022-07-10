"""Models for facial keypoint detection"""
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F

# TODO: Choose from either model and uncomment that line
# class KeypointModel(nn.Module):
class KeypointModel(pl.LightningModule):
    """Facial keypoint detection model"""
    def __init__(self, hparams):
        """
        Initialize your model from a given dict containing all your hparams
        Warning: Don't change the method declaration (i.e. by adding more
            arguments), otherwise it might not work on the submission server
            
        NOTE: You could either choose between pytorch or pytorch lightning, 
            by switching the class name line.
        """
        super().__init__()
        self.save_hyperparameters(hparams)
        
        ########################################################################
        # TODO: Define all the layers of your CNN, the only requirements are:  #
        # 1. The network takes in a batch of images of shape (Nx1x96x96)       #
        # 2. It ends with a linear layer that represents the keypoints.        #
        # Thus, the output layer needs to have shape (Nx30),                   #
        # with 2 values representing each of the 15 keypoint (x, y) pairs      #
        #                                                                      #
        # Some layers you might consider including:                            #
        # maxpooling layers, multiple conv layers, fully-connected layers,     #
        # and other layers (such as dropout or batch normalization) to avoid   #
        # overfitting.                                                         #
        #                                                                      #
        # We would truly recommend to make your code generic, such as you      #
        # automate the calculation of the number of parameters at each layer.  #
        # You're going probably try different architecutres, and that will     #
        # allow you to be quick and flexible.                                  #
        ########################################################################
        
        def conv2d_block(in_ch, out_ch, 
                         kern = self.hparams["kernel_size"], 
                         stride = self.hparams["stride"], 
                         pad = self.hparams["padding"], 
                         pool_kern =  self.hparams["kernel_size_pool"], 
                         pool_stride = self.hparams["stride_pool"]):
            
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kern, stride, pad),
                nn.MaxPool2d(pool_kern, pool_stride),
                nn.ReLU()
            )
            
            
        self.model = nn.Sequential(
            conv2d_block( in_ch = self.hparams["in_ch_1"], 
                         out_ch = self.hparams["out_ch_1"]),
            
            conv2d_block( in_ch = self.hparams["in_ch_2"], 
                         out_ch = self.hparams["out_ch_2"]),
            
            conv2d_block( in_ch = self.hparams["in_ch_3"], 
                         out_ch = self.hparams["out_ch_3"]),
            
            conv2d_block( in_ch = self.hparams["in_ch_4"], 
                         out_ch = self.hparams["out_ch_4"]),
            
            nn.Flatten(),
            nn.Linear(self.hparams["flatten_out"], self.hparams["n_hidden"]),
            nn.ReLU(),
            nn.Linear(self.hparams["n_hidden"], self.hparams["net_out"]),
            nn.Tanh()
        )

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def forward(self, x):
        # check dimensions to use show_keypoint_predictions later
        if x.dim() == 3:
            x = torch.unsqueeze(x, 0)
        return self.model(x)

    def general_step(self, batch):
        input, y = batch["image"], batch["keypoints"]
        y_hat = self.forward(input).view(-1, 15, 2)#resize the batch data
        #squeeze mainly compresses the dimension of the data and removes the dimension of dimension 1.
        loss = F.mse_loss(torch.squeeze(y_hat), torch.squeeze(y))
        return loss
    
    def training_step(self, batch):
        loss = self.general_step(batch)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.model.parameters(), 
                                self.hparams["learning_rate"], 
                                momentum=0.9,
                                weight_decay=1e-6, 
                                nesterov=True)


class DummyKeypointModel(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.prediction = torch.tensor([[
            0.4685, -0.2319,
            -0.4253, -0.1953,
            0.2908, -0.2214,
            0.5992, -0.2214,
            -0.2685, -0.2109,
            -0.5873, -0.1900,
            0.1967, -0.3827,
            0.7656, -0.4295,
            -0.2035, -0.3758,
            -0.7389, -0.3573,
            0.0086, 0.2333,
            0.4163, 0.6620,
            -0.3521, 0.6985,
            0.0138, 0.6045,
            0.0190, 0.9076,
        ]])

    def forward(self, x):
        return self.prediction.repeat(x.size()[0], 1, 1, 1)
