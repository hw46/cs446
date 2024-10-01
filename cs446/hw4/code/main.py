import hw4
import hw4_utils
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def main():

    # initialize parameters
    lr = 0.01
    latent_dim = 6
    lam = 5e-5
    n_iter = 8000
    loss_fn = nn.MSELoss()

    # initialize model
    vae = hw4.VAE(lam=lam, lrate=lr, latent_dim=latent_dim, loss_fn=loss_fn)

    # generate data
    X = hw4_utils.generate_data()

    # fit the model to the data
    loss_rec, loss_kl, loss_total, Xhat, gen_samples = hw4.fit(vae, X, n_iter)
    
    
    # Plot empirical risk
    plt.figure(figsize=(10, 5))
    plt.plot(loss_total, label='Total Loss (Empirical Risk)')
    plt.title('Empirical Risk RbVAE vs Iteration Count')
    plt.xlabel('Iteration')
    plt.ylabel('Empirical Risk')
    plt.legend()
    plt.show()

    # Plot data points and their encoded-decoded approximations
    plt.figure(figsize=(10, 5))
    plt.scatter(X[:, 0], X[:, 1], label='Original Data')
    plt.scatter(Xhat[:, 0], Xhat[:, 1], label='Decoded Approximations')
    plt.title('Data Points and Decoded Approximations')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.show()

    # Plot data points, decoded approximations, and generated points
    plt.figure(figsize=(10, 5))
    plt.scatter(X[:, 0], X[:, 1], label='Original Data')
    plt.scatter(Xhat[:, 0], Xhat[:, 1], label='Decoded Approximations')
    plt.scatter(gen_samples[:, 0], gen_samples[:, 1], label='Generated Points')
    plt.title('Data Points, Decoded Approximations, and Generated Points')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.show()

    # Save the model to a checkpoint file
    torch.save(vae.cpu().state_dict(), "vae.pb")
    

if __name__ == "__main__":
    main()
