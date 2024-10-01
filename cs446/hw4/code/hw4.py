import hw4_utils
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class VAE(torch.nn.Module):
    def __init__(self, lam,lrate,latent_dim,loss_fn):
        """
        Initialize the layers of your neural network

        @param lam: Hyperparameter to scale KL-divergence penalty
        @param lrate: The learning rate for the model.
        @param loss_fn: A loss function defined in the following way:
            @param x - an (N,D) tensor
            @param y - an (N,D) tensor
            @return l(x,y) an () tensor that is the mean loss
        @param latent_dim: The dimension of the latent space

        The network should have the following architecture (in terms of hidden units):
        Encoder Network:
        2 -> 50 -> ReLU -> 50 -> ReLU -> 50 -> ReLU -> (6,6) (mu_layer,logstd2_layer)

        Decoder Network:
        6 -> 50 -> ReLU -> 50 -> ReLU -> 2 -> Sigmoid

        See set_parameters() function for the exact shapes for each weight
        """
        super(VAE, self).__init__()

        self.lrate = lrate
        self.lam = lam
        self.loss_fn = loss_fn
        self.latent_dim = latent_dim
        
        # Encoder layers
        self.encoder_fc1 = nn.Linear(2, 50)
        self.encoder_fc2 = nn.Linear(50, 50)
        self.encoder_fc3 = nn.Linear(50, 50)
        self.mu_layer = nn.Linear(50, latent_dim)
        self.logstd2_layer = nn.Linear(50, latent_dim)

        # Decoder layers
        self.decoder_fc1 = nn.Linear(latent_dim, 50)
        self.decoder_fc2 = nn.Linear(50, 50)
        self.decoder_fc3 = nn.Linear(50, 2)

        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=lrate)
        
        
        
    def set_parameters(self, We1,be1, We2, be2, We3, be3, Wmu, bmu, Wstd, bstd, Wd1, bd1, Wd2, bd2, Wd3, bd3):
        """ Set the parameters of your network

        # Encoder weights:
        @param We1: an (50,2) torch tensor
        @param be1: an (50,) torch tensor
        @param We2: an (50,50) torch tensor
        @param be2: an (50,) torch tensor
        @param We3: an (50,50) torch tensor
        @param be3: an (50,) torch tensor
        @param Wmu: an (6,50) torch tensor
        @param bmu: an (6,) torch tensor
        @param Wstd: an (6,50) torch tensor
        @param bstd: an (6,) torch tensor

        # Decoder weights:
        @param Wd1: an (50,6) torch tensor
        @param bd1: an (50,) torch tensor
        @param Wd2: an (50,50) torch tensor
        @param bd2: an (50,) torch tensor
        @param Wd3: an (2,50) torch tensor
        @param bd3: an (2,) torch tensor

        """
        
        # Encoder parameters
        self.encoder_fc1.weight = nn.Parameter(We1)
        self.encoder_fc1.bias = nn.Parameter(be1)
        self.encoder_fc2.weight = nn.Parameter(We2)
        self.encoder_fc2.bias = nn.Parameter(be2)
        self.encoder_fc3.weight = nn.Parameter(We3)
        self.encoder_fc3.bias = nn.Parameter(be3)
        self.mu_layer.weight = nn.Parameter(Wmu)
        self.mu_layer.bias = nn.Parameter(bmu)
        self.logstd2_layer.weight = nn.Parameter(Wstd)
        self.logstd2_layer.bias = nn.Parameter(bstd)

        # Decoder parameters
        self.decoder_fc1.weight = nn.Parameter(Wd1)
        self.decoder_fc1.bias = nn.Parameter(bd1)
        self.decoder_fc2.weight = nn.Parameter(Wd2)
        self.decoder_fc2.bias = nn.Parameter(bd2)
        self.decoder_fc3.weight = nn.Parameter(Wd3)
        self.decoder_fc3.bias = nn.Parameter(bd3)
        
        

    def forward(self, x):
        """ A forward pass of your autoencoder

        @param x: an (N, 2) torch tensor

        # return the following in this order from left to right:
        @return y: an (N, 50) torch tensor of output from the encoder network
        @return mean: an (N,latent_dim) torch tensor of output mu layer
        @return stddev_p: an (N,latent_dim) torch tensor of output stddev layer
        @return z: an (N,latent_dim) torch tensor sampled from N(mean,exp(stddev_p/2)
        @return xhat: an (N,D) torch tensor of outputs from f_dec(z)
        """
        
        # Pass through the encoder network
        h1 = F.relu(self.encoder_fc1(x))
        h2 = F.relu(self.encoder_fc2(h1))
        h3 = F.relu(self.encoder_fc3(h2))
        mean = self.mu_layer(h3)
        logstd2 = self.logstd2_layer(h3)

        # Reparameterization trick
        stddev = torch.exp(0.5 * logstd2)
        z = mean + stddev * torch.randn_like(stddev)

        # Pass through the decoder network
        h4 = F.relu(self.decoder_fc1(z))
        h5 = F.relu(self.decoder_fc2(h4))
        xhat = torch.sigmoid(self.decoder_fc3(h5))

        # y is the output of the last encoder layer before the latent layer
        y = h3

        # Returning the values in the specified order
        return y, mean, logstd2, z, xhat
        
        

    def step(self, x):
        """
        Performs one gradient step through a batch of data x
        @param x: an (N, 2) torch tensor

        # return the following in this order from left to right:
        @return L_rec: float containing the reconstruction loss at this time step
        @return L_kl: kl divergence penalty at this time step
        @return L: total loss at this time step
        """
        
        self.optimizer.zero_grad()
        _, mean, logstd2, _, xhat = self.forward(x)
        
        # Compute the reconstruction loss
        L_rec = self.loss_fn(xhat, x).mean()
        
        # Compute the KL divergence
        L_kl = -0.5 * torch.sum(1 + logstd2 - mean.pow(2) - logstd2.exp())
        L_kl /= x.size(0) * self.latent_dim
        
        # Total loss
        L = L_rec + self.lam * L_kl
        
        # Backpropagation
        L.backward()
        self.optimizer.step()
        
        return L_rec.item(), L_kl.item(), L.item()


def fit(net,X,n_iter):
    """ Fit a VAE.  Use the full batch size.
    @param net: the VAE
    @param X: an (N, D) torch tensor
    @param n_iter: int, the number of iterations of training

    # return all of these from left to right:

    @return losses_rec: Array of reconstruction losses at the beginning and after each iteration. Ensure len(losses_rec) == n_iter
    @return losses_kl: Array of KL loss penalties at the beginning and after each iteration. Ensure len(losses_kl) == n_iter
    @return losses: Array of total loss at the beginning and after each iteration. Ensure len(losses) == n_iter
    @return Xhat: an (N,D) NumPy array of approximations to X
    @return gen_samples: an (N,D) NumPy array of N samples generated by the VAE
    """
    
    losses_rec = []
    losses_kl = []
    losses = []
    
    for i in range(n_iter):
        L_rec, L_kl, L = net.step(X)
        losses_rec.append(L_rec)
        losses_kl.append(L_kl)
        losses.append(L)
    
    net.eval()
    with torch.no_grad():
        # Generate reconstructions of X
        _, _, _, _, Xhat = net.forward(X)
        # Generate new samples from the latent space
        z = torch.randn(X.size(0), net.latent_dim)
        gen_samples = net.decoder_fc3(F.relu(net.decoder_fc2(F.relu(net.decoder_fc1(z)))))
        gen_samples = torch.sigmoid(gen_samples)
    
    # Convert to numpy arrays
    Xhat = Xhat.cpu().numpy()
    gen_samples = gen_samples.cpu().numpy()
    
    return losses_rec, losses_kl, losses, Xhat, gen_samples