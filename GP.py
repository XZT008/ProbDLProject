import gpytorch
import torch
import numpy as np
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from botorch import fit_fully_bayesian_model_nuts
from pyro.infer.mcmc import NUTS, MCMC
import pyro
from botorch.models.gpytorch import GPyTorchModel
from gpytorch.priors import GammaPrior, UniformPrior, NormalPrior
from gpytorch.constraints import Positive


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.ls_constraint = None
        self.D = train_x.shape[1]

        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(ard_num_dims=train_x.shape[1], lengthscale_constraint=self.ls_constraint),
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GP_Wrapper:
    def __init__(self, train_x, train_y, device="cpu"):
        self.device = device
        self.X = train_x
        self.y = train_y.squeeze()

        self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
        self.gp_model = ExactGPModel(self.X, self.y, self.likelihood).to(self.device)

    def train_model(self, epochs=500, lr=0.1, optim="ADAM"):
        self.gp_model.train()
        self.likelihood.train()

        if optim == "ADAM":
            optimizer = torch.optim.Adam(self.gp_model.parameters(), lr=lr)
        elif optim == "RMSPROP":
            optimizer = torch.optim.RMSprop(self.gp_model.parameters(), lr=lr)
        else:
            raise NotImplementedError
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.gp_model).to(self.device)

        for epoch in range(epochs):
            optimizer.zero_grad()
            output = self.gp_model(self.X)
            loss = -mll(output, self.y)
            loss.backward()
            optimizer.step()

    def pred(self, test_x, num_samples=8):
        self.gp_model.eval()
        f_pred = self.gp_model(test_x)
        means = f_pred.mean
        vars = f_pred.variance
        dist = torch.distributions.MultivariateNormal(
            means.squeeze(),
            torch.diag(vars.squeeze())
        )
        samples = dist.sample((num_samples,)).permute(1, 0)
        return means


class SAASBO_Wrapper:
    def __init__(self, train_x, train_y, device="cpu"):
        self.device = device
        self.X = train_x
        self.y = train_y
        self.gp = SaasFullyBayesianSingleTaskGP(
                    train_X=self.X,
                    train_Y=self.y,
                )

    def train_model(self):
        fit_fully_bayesian_model_nuts(
            self.gp,
            warmup_steps=256,
            num_samples=128,
            thinning=16,
            disable_progbar=False,
        )

    def pred(self, test_X):
        with torch.no_grad():
            posterior = self.gp.posterior(test_X)
        mixture_mean = posterior.mixture_mean
        return mixture_mean


class ADDGPModel(gpytorch.models.ExactGP):

    def __init__(self, train_x, train_y, likelihood):
        super(ADDGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.dim = train_x.shape[1]

        self.kernel_list = [gpytorch.kernels.MaternKernel(active_dims=i) for i in range(self.dim)]
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.AdditiveKernel(*self.kernel_list))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class ADDGP_Wrapper:
    def __init__(self, train_x, train_y):
        self.X = train_x
        self.y = train_y.squeeze()
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()

        self.gp_model = ADDGPModel(self.X, self.y, self.likelihood)

    def train_model(self, epochs=500, lr=0.1):
        self.gp_model.train()
        self.likelihood.train()
        optimizer = torch.optim.Adam(self.gp_model.parameters(), lr=lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.gp_model)

        for epoch in range(epochs):
            optimizer.zero_grad()
            output = self.gp_model(self.X)
            # Calc loss and backprop gradients
            loss = -mll(output, self.y)
            loss.backward()
            optimizer.step()

    def pred(self, test_x, num_samples=8):
        self.gp_model.eval()
        f_pred = self.gp_model(test_x)
        means = f_pred.mean
        return means

class ExactGPModelPyro(gpytorch.models.ExactGP, GPyTorchModel):
    _num_outputs = 1

    def __init__(self, train_x, train_y, likelihood, if_ard=False, if_softplus=True):
        super(ExactGPModelPyro, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.ls_constraint = None
        if not if_softplus:
            self.ls_constraint = Positive(transform=torch.exp, inv_transform=torch.log)
        if if_ard:
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel(ard_num_dims=train_x.shape[1], lengthscale_constraint=self.ls_constraint),
            )
        else:
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel(lengthscale_constraint=self.ls_constraint))
        self.fitted = False
        self.num_samples = None

    def _check_if_fitted(self):
        return self.fitted

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GP_Wrapper_pyro:
    def __init__(self, train_x, train_y, if_ard=True, if_softplus=True):
        self.X = train_x
        self.y = train_y.squeeze()
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.Positive())
        self.gp_model = ExactGPModelPyro(self.X, self.y, self.likelihood, if_ard, if_softplus)
        # this model is used for loading samples from gp_model for easy batched posterior
        self.gp_model_acqf = SaasFullyBayesianSingleTaskGP(self.X, self.y.unsqueeze(-1))

    def train_model(self, warmup_steps=256, num_samples=128, thinning=16):
        self.gp_model.mean_module.register_prior("mean_prior", NormalPrior(0.0, 1.0), "constant")
        self.gp_model.covar_module.base_kernel.register_prior("lengthscale_prior", UniformPrior(0.001, 30.0),
                                                              "lengthscale")
        self.gp_model.covar_module.register_prior("outputscale_prior", GammaPrior(2.0, 0.15), "outputscale")
        self.likelihood.register_prior("noise_prior", GammaPrior(0.9, 10.0), "noise")

        def pyro_model(x, y):
            with gpytorch.settings.fast_computations(False, False, False):
                sampled_model = self.gp_model.pyro_sample_from_prior()
                output = sampled_model.likelihood(sampled_model(x))
                pyro.sample("obs", output, obs=y)
            return y

        nuts_kernel = NUTS(pyro_model)
        mcmc_run = MCMC(nuts_kernel, num_samples=num_samples, warmup_steps=warmup_steps, disable_progbar=False)

        mcmc_run.run(self.X, self.y)
        mcmc_samples = mcmc_run.get_samples()
        for k, v in mcmc_samples.items():
            mcmc_samples[k] = v[::thinning]
        self.gp_model.pyro_load_from_samples(mcmc_samples)
        (
            self.gp_model_acqf.mean_module,
            self.gp_model_acqf.covar_module,
            self.gp_model_acqf.likelihood,
        ) = self.gp_model.mean_module, self.gp_model.covar_module, self.gp_model.likelihood

        self.gp_model.fitted = True
        self.gp_model.num_samples = int(num_samples / thinning)
        self.gp_model.eval()

    def pred(self, test_X):
        with torch.no_grad():
            posterior = self.gp_model.posterior(test_X)
        means = posterior.mean.squeeze().mean(dim=0)
        return means
