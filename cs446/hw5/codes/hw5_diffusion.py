import cv2

import torch
import torch.nn as nn
import torch.optim as optim

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from hw5_utils import create_samples, generate_sigmas, plot_score



class ScoreNet(nn.Module):
    def __init__(self, n_layers=8, latent_dim=128):
        super().__init__()

        # TODO: Implement the neural network
        # The network has n_layers of linear layers. 
        # Latent dimensions are specified with latent_dim.
        # Between each two linear layers, we use Softplus as the activation layer.
        
        # Prepare to construct layers dynamically
        layers = []
        input_dim = 3  # As x has 2 dimensions and sigma has 1 dimension

        # Create a sequence of linear and softplus layers
        for _ in range(n_layers):
            layers.append(nn.Linear(input_dim, latent_dim))
            layers.append(nn.Softplus())
            input_dim = latent_dim  # Update input dimension for the next layer

        # Append final linear layer to map to the score size (2, corresponding to x)
        layers.append(nn.Linear(latent_dim, 2))

        # Wrap the list of layers into nn.Sequential
        self.net = nn.Sequential(*layers)

    def forward(self, x, sigmas):
        """.
        Parameters
        ----------
        x : torch.tensor, N x 2

        sigmas : torch.tensor of shape N x 1 or a float number
        """
        
        if isinstance(sigmas, float):
            sigmas = torch.tensor(sigmas).reshape(1, 1).repeat(x.shape[0], 1)
        if sigmas.dim() == 0:
            sigmas = sigmas.reshape(1, 1).repeat(x.shape[0], 1)
        
        # we use the trick from NCSNv2 to explicitly divide sigma
        return self.net(torch.concatenate([x, sigmas], dim=-1)) / sigmas
    
    
        # Ensure sigmas is a 2D tensor with the same second dimension as x
        if sigmas.dim() == 1:
            sigmas = sigmas.unsqueeze(1)  # Reshape from (N,) to (N, 1)

        # Concatenate x and sigmas along the last dimension
        input_tensor = torch.cat([x, sigmas], dim=1)  # Ensure concatenation along features dimension
        scores = self.net(input_tensor)  # Compute scores
        
        # Scale the scores by sigmas
        if sigmas.shape[1] == 1:  # If sigmas is (N, 1), expand it to (N, 2) for element-wise division
            sigmas = sigmas.expand_as(scores)
        return scores / sigmas


def compute_denoising_loss(scorenet, training_data, sigmas):
    """
    Compute the denoising loss.

    Parameters
    ----------
    scorenet : nn.Module
        The neural network for score prediction

    training_data : np.array, N x 2
        The training data

    sigmas : np.array, L
        The list of sigmas

    Return
    ------
    loss averaged over all training data
    """
    B, C = training_data.shape

    # TODO: Implement the denoising loss follow the steps: 
    # For each training sample x: 
    # 1. Randomly sample a sigma from sigmas
    # 2. Perturb the training sample: \tilde(x) = x + sigma * z
    # 3. Get the predicted score
    # 4. Compute the loss: 1/2 * lambda * ||score + ((\tilde(x) - x) / sigma^2)||^2
    # Return the loss averaged over all training samples
    # Note: use batch operations as much as possible to avoid iterations
    
    B, C = training_data.shape  # B is batch size, C is number of dimensions (2 in this case)

    # Randomly sample a sigma for each training sample from the list of sigmas
    indices = torch.randint(0, len(sigmas), (B,), device=training_data.device)
    sampled_sigmas = sigmas[indices]  # Get the corresponding sigmas for each sample

    # Ensure sampled_sigmas is two-dimensional (B, 1)
    if sampled_sigmas.dim() == 1:
        sampled_sigmas = sampled_sigmas.unsqueeze(1)

    # Sample z ~ N(0, I)
    z = torch.randn(B, C, device=training_data.device)

    # Perturb the training sample: x_tilde = x + sigma * z
    x_tilde = training_data + sampled_sigmas * z

    # Get the predicted score for the perturbed samples
    predicted_scores = scorenet(x_tilde, sampled_sigmas)  # Pass x_tilde and sampled_sigmas separately

    # Compute the loss
    lambda_factor = sampled_sigmas.squeeze() ** 2  # Ensure lambda_factor is not 2-dimensional
    correction_term = (x_tilde - training_data) / (sampled_sigmas ** 2)
    loss_elements = predicted_scores + correction_term
    loss = 0.5 * lambda_factor * (loss_elements ** 2).sum(dim=1)

    # Return the loss averaged over all training samples
    return loss.mean()


@torch.no_grad()
def langevin_dynamics_sample(scorenet, n_samples, sigmas, iterations=100, eps=0.00002, return_traj=False):
    """
    Sample with langevin dynamics.

    Parameters
    ----------
    scorenet : nn.Module
        The neural network for score prediction

    n_samples: int
        Number of samples to acquire

    sigmas : np.array, L
        The list of sigmas

    iterations: int
        The number of iterations for each sigma (T in Alg. 2)

    eps: float
        The parameter to control step size

    return_traj: bool, default is False
        If True, return all intermediate samples
        If False, only return the last step

    Return
    ------
    torch.Tensor in the shape of n_samples x 2 if return_traj=False
    in the shape of n_samples x (L*T) x 2 if return_traj=True
    """

    # TODO: Implement the Langevin dynamics following the steps:
    # 1. Initialize x_0 ~ N(0, I)
    # 2. Iterate through sigmas, for each sigma:
    # 3.    Compute alpha = eps * sigma^2 / sigmaL^2
    # 4.    Iterate through T steps:
    # 5.        x_t = x_{t-1} + alpha * scorenet(x_{t-1}, sigma) + sqrt(2 * alpha) * z
    # 6.    x_0 = x_T
    # 7. Return the last x_T if return_traj=False, or return all x_t
    
    # Determine the device from the first parameter of the model
    device = next(scorenet.parameters()).device

    # Initialize samples: x_0 ~ N(0, I)
    samples = torch.randn(n_samples, 2, device=device)

    if return_traj:
        trajectories = [samples.unsqueeze(1)]

    # Compute alpha for each sigma, iterating through sigmas
    sigmaL2 = sigmas[-1] ** 2
    for sigma in sigmas:
        alpha = eps * (sigma ** 2) / sigmaL2
        sigma = sigma.expand(n_samples, 2)  # Expand sigma to match the dimensions of samples

        # Iterative update of samples
        for _ in range(iterations):
            z = torch.randn_like(samples)
            score = scorenet(samples, sigma)
            samples += alpha * score + torch.sqrt(2 * alpha) * z

            if return_traj:
                trajectories.append(samples.unsqueeze(1))

        # Set x_0 for the next sigma level
        samples = samples.clone()  # Ensure updates are correct for next sigma

    if return_traj:
        return torch.cat(trajectories, dim=1)  # Concatenate along the new dimension
    else:
        return samples


def main():
    # training related hyperparams
    lr = 0.01
    n_iters = 50000
    log_freq = 1000

    # sampling related hyperparams
    n_samples = 1000
    sample_iters = 100
    sample_lr = 0.00002

    # create the training set
    training_data = torch.tensor(create_samples()).float()

    # visualize the training data
    plt.figure(figsize=(20, 5))
    plt.scatter(training_data[:, 0], training_data[:, 1])
    plt.axis('scaled')
    plt.xlim(-2.5, 2.5)
    plt.ylim(-0.6, 0.6)
    plt.show()


    # create ScoreNet and optimizer
    scorenet = ScoreNet()
    scorenet.train()
    optimizer = optim.Adam(scorenet.parameters(), lr=lr)

    # generate sigmas in descending order: sigma1 > sigma2 > ... > sigmaL
    sigmas = torch.tensor(generate_sigmas(0.3, 0.01, 10)).float()

    avg_loss = 0.
    for i_iter in range(n_iters):
        optimizer.zero_grad()
        loss = compute_denoising_loss(scorenet, training_data, sigmas)
        loss.backward()
        optimizer.step()

        avg_loss += loss.item()
        if i_iter % log_freq == log_freq - 1:
            avg_loss /= log_freq
            print(f'iter {i_iter}: loss = {avg_loss:.3f}')
            avg_loss = 0.

    torch.save(scorenet.state_dict(), 'model.ckpt')
    
    # Q5(a). visualize score function
    scorenet.eval()
    plot_score(scorenet, training_data)

    # Q5(b). sample with langevin dynamics
    samples = langevin_dynamics_sample(scorenet, n_samples, sigmas, sample_iters, sample_lr, return_traj=True).numpy()

    # plot the samples
    for step in range(0, sample_iters * len(sigmas), 200):
        plt.figure(figsize=(20, 5))
        plt.scatter(samples[:, step, 0], samples[:, step, 1], color='red')
        plt.axis('scaled')
        plt.xlim(-2.5, 2.5)
        plt.ylim(-0.6, 0.6)
        plt.title(f'Samples at step={step}')
        plt.show()

    plt.figure(figsize=(20, 5))
    plt.scatter(samples[:, -1, 0], samples[:, -1, 1], color='red')
    plt.axis('scaled')
    plt.xlim(-2.5, 2.5)
    plt.ylim(-0.6, 0.6)
    plt.title('All samples')
    plt.show()

    # Q5(c). visualize the trajectory
    traj = langevin_dynamics_sample(scorenet, 2, sigmas, sample_iters, sample_lr, return_traj=True).numpy()
    plt.figure(figsize=(20, 5))
    plt.plot(traj[0, :, 0], traj[0, :, 1], color='blue')
    plt.plot(traj[1, :, 0], traj[1, :, 1], color='green')

    plt.scatter(samples[:, -1, 0], samples[:, -1, 1], color='red')
    plt.axis('scaled')
    plt.xlim(-2.5, 2.5)
    plt.ylim(-0.6, 0.6)
    plt.title('Trajectories')
    plt.show()


if __name__ == '__main__':
    main()