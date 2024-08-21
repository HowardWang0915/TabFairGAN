import torch
class CondIndLossFunc(torch.nn.Module):
    def __init__(self, S_start_index, Y_start_index, X_start_index, len_S, len_Y, len_X):
        super(CondIndLossFunc, self).__init__()
        self._S_start_index = S_start_index
        self._Y_start_index = Y_start_index
        self._X_start_index = X_start_index
        self._len_S = len_S
        self._len_Y = len_Y
        self._len_X = len_X
    # def forward(self, x, crit_fake_pred, lam):
    #     G = x[:, self._S_start_index:self._S_start_index + 2]
    #     # print(x[0,64])
    #     I = x[:, self._Y_start_index:self._Y_start_index + 2]
    #     X = x[:, self._X_start_index:self._X_start_index + 2]
    #     mean_Y_given_S_X = (G[:, self._underpriv_index] * I[:, self._desire_index] * X[:, self._X_desire_index]).sum() / (G[:, self._underpriv_index] * X[:, self._X_desire_index]).sum()
    #     mean_Y_given_X = (I[:, self._desire_index] * X[:, self._X_desire_index]).sum() / (X[:, self._X_desire_index]).sum()
    #     conditional_indep_loss = lam * torch.abs(mean_Y_given_S_X - mean_Y_given_X)
    #     disp = conditional_indep_loss - torch.mean(crit_fake_pred)
    #
    #     return disp
    def forward(self, x, crit_fake_pred, lam):
        total_loss = 0

        for i in range(self._len_S):
            G = x[:, self._S_start_index + i]

            for j in range(self._len_Y):
                I = x[:, self._Y_start_index + j]

                for k in range(self._len_X):
                    X_attr = x[:, self._X_start_index + k]

                    # Compute mean_Y_given_S_X and mean_Y_given_X
                    mean_Y_given_S_X = (G * I * X_attr).sum() / (G * X_attr).sum()
                    mean_Y_given_X = (I * X_attr).sum() / X_attr.sum()

                    # Compute the conditional independence loss
                    conditional_indep_loss = lam * torch.abs(mean_Y_given_S_X - mean_Y_given_X)
                    total_loss += conditional_indep_loss

        # Optional: Include additional terms in the loss
        disp = total_loss - torch.mean(crit_fake_pred)

        return disp

def get_gradient(crit, real, fake, epsilon):
    mixed_data = real * epsilon + fake * (1 - epsilon)

    mixed_scores = crit(mixed_data)

    gradient = torch.autograd.grad(
        inputs=mixed_data,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    return gradient


def gradient_penalty(gradient):
    gradient = gradient.view(len(gradient), -1)
    gradient_norm = gradient.norm(2, dim=1)

    penalty = torch.mean((gradient_norm - 1) ** 2)
    return penalty


def get_gen_loss(crit_fake_pred):
    gen_loss = -1. * torch.mean(crit_fake_pred)

    return gen_loss


def get_crit_loss(crit_fake_pred, crit_real_pred, gp, c_lambda):
    crit_loss = torch.mean(crit_fake_pred) - torch.mean(crit_real_pred) + c_lambda * gp

    return crit_loss
