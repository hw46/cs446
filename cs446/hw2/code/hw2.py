import hw2_utils
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

def svm_solver(x_train, y_train, lr, num_iters,
               kernel=hw2_utils.poly(degree=1), c=None):
    '''
    Computes an SVM given a training set, training labels, the number of
    iterations to perform projected gradient descent, a kernel, and a trade-off
    parameter for soft-margin SVM.

    Arguments:
        x_train: 2d tensor with shape (n, d).
        y_train: 1d tensor with shape (n,), whose elememnts are +1 or -1.
        lr: The learning rate.
        num_iters: The number of gradient descent steps.
        kernel: The kernel function.
           The default kernel function is 1 + <x, y>.
        c: The trade-off parameter in soft-margin SVM.
           The default value is None, referring to the basic, hard-margin SVM.

    Returns:
        alpha: a 1d tensor with shape (n,), denoting an optimal dual solution.
               Initialize alpha to be 0.
               Return alpha.detach() could possibly help you save some time
               when you try to use alpha in other places.

    Note that if you use something like alpha = alpha.clamp(...) with
    torch.no_grad(), you will have alpha.requires_grad=False after this step.
    You will then need to use alpha.requires_grad_().
    Alternatively, use in-place operations such as clamp_().
    '''
    n = x_train.shape[0]
    alpha = torch.zeros(n, requires_grad=False)
    for _ in range(num_iters):
        grad_alpha = torch.zeros_like(alpha)
        for i in range(n):
            for j in range(n):
                k_ij = kernel(x_train[i], x_train[j])
                grad_alpha[i] += alpha[j] * y_train[j] * y_train[i] * k_ij
            grad_alpha[i] = grad_alpha[i] - 1

        alpha -= lr * grad_alpha

        with torch.no_grad():
            if c is None:
                alpha.clamp_(min=0)
            else:
                alpha.clamp_(min=0, max=c)

    return alpha

def svm_predictor(alpha, x_train, y_train, x_test,
                  kernel=hw2_utils.poly(degree=1)):
    '''
    Returns the kernel SVM's predictions for x_test using the SVM trained on
    x_train, y_train with computed dual variables alpha.

    Arguments:
        alpha: 1d tensor with shape (n,), denoting an optimal dual solution.
        x_train: 2d tensor with shape (n, d), denoting the training set.
        y_train: 1d tensor with shape (n,), whose elements are +1 or -1.
        x_test: 2d tensor with shape (m, d), denoting the test set.
        kernel: The kernel function.
           The default kernel function is 1 + <x, y>.

    Return:
        A 1d tensor with shape (m,), the outputs of SVM on the test set.
    '''
    alpha_y = alpha * y_train.unsqueeze(1)

    predictions = torch.zeros(x_test.size(0))
    for i in range(x_test.size(0)):
        k_values = torch.tensor([kernel(x_train[j], x_test[i]) for j in range(x_train.size(0))])
        decision_value = torch.sum(alpha_y[:, 0] * k_values)
        predictions[i] = decision_value

    return torch.sign(predictions)


x_train, y_train = hw2_utils.xor_data()
lr = 0.1
num_iters = 10000
alpha_poly = svm_solver(x_train, y_train, lr, num_iters, kernel=hw2_utils.poly(degree=2))


'''x_train, y_train = hw2_utils.xor_data()

# Training Parameters
lr = 0.1
num_iters = 10000

# Train SVM with Polynomial Kernel, Degree 2
alpha_poly = svm_solver(x_train, y_train, lr, num_iters, kernel=hw2_utils.poly(degree=2))

# Train SVM with RBF Kernel, sigma = 1, 2, 4
alpha_rbf_sigma1 = svm_solver(x_train, y_train, lr, num_iters, kernel=hw2_utils.rbf(sigma=1))
alpha_rbf_sigma2 = svm_solver(x_train, y_train, lr, num_iters, kernel=hw2_utils.rbf(sigma=2))
alpha_rbf_sigma4 = svm_solver(x_train, y_train, lr, num_iters, kernel=hw2_utils.rbf(sigma=4))

def make_predictor(alpha, x_train, y_train, kernel):
    def predictor(x_test):
        return svm_predictor(alpha, x_train, y_train, x_test, kernel)
    return predictor

# Create predictors
pred_poly = make_predictor(alpha_poly, x_train, y_train, hw2_utils.poly(degree=2))
pred_rbf_sigma1 = make_predictor(alpha_rbf_sigma1, x_train, y_train, hw2_utils.rbf(sigma=1))
pred_rbf_sigma2 = make_predictor(alpha_rbf_sigma2, x_train, y_train, hw2_utils.rbf(sigma=2))
pred_rbf_sigma4 = make_predictor(alpha_rbf_sigma4, x_train, y_train, hw2_utils.rbf(sigma=4))

# Plot contours
hw2_utils.svm_contour(pred_poly, xmin=-5, xmax=5, ymin=-5, ymax=5)
hw2_utils.svm_contour(pred_rbf_sigma1, xmin=-5, xmax=5, ymin=-5, ymax=5)
hw2_utils.svm_contour(pred_rbf_sigma2, xmin=-5, xmax=5, ymin=-5, ymax=5)
hw2_utils.svm_contour(pred_rbf_sigma4, xmin=-5, xmax=5, ymin=-5, ymax=5)'''



# XOR Data
x_train, y_train = hw2_utils.xor_data()

# Training Parameters
lr = 0.1
num_iters = 10000

# Train SVMs
kernels = {
    "poly2": hw2_utils.poly(degree=2),
    "rbf1": hw2_utils.rbf(sigma=1),
    "rbf2": hw2_utils.rbf(sigma=2),
    "rbf4": hw2_utils.rbf(sigma=4),
}

alphas = {}
for key, kernel in kernels.items():
    alphas[key] = svm_solver(x_train, y_train, lr, num_iters, kernel=kernel)
    
def make_predictor(alpha, x_train, y_train, kernel):
    def predictor(x_test):
        return svm_predictor(alpha, x_train, y_train, x_test, kernel)
    return predictor

predictors = {key: make_predictor(alpha, x_train, y_train, kernel) for key, alpha in alphas.items() for key, kernel in kernels.items() if key in alphas}

for key, predictor in predictors.items():
    plt.figure(figsize=(8, 6))
    hw2_utils.svm_contour(predictor, xmin=-5, xmax=5, ymin=-5, ymax=5)
    plt.title(f"Contour Plot for {key} Kernel SVM")
    plt.show()