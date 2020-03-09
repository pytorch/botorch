import torch
import math

device = torch.device("cpu")
dtype = torch.float
torch.manual_seed(3);

sigma = math.sqrt(0.2)
train_X = torch.linspace(0, 1, 20, dtype=dtype, device=device).view(-1, 1)
train_Y_noiseless = torch.sin(train_X * (2 * math.pi))
train_Y = train_Y_noiseless + sigma * torch.randn_like(train_Y_noiseless)
train_Yvar = torch.full_like(train_Y, 0.2)

from botorch.cross_validation import gen_loo_cv_folds

cv_folds = gen_loo_cv_folds(train_X=train_X, train_Y=train_Y, train_Yvar=train_Yvar)

cv_folds.train_X.shape, cv_folds.train_Y.shape

cv_folds.test_X.shape, cv_folds.test_Y.shape

from botorch.cross_validation import batch_cross_validation
from botorch.models import FixedNoiseGP
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood

# instantiate and fit model
cv_results = batch_cross_validation(
    model_cls=FixedNoiseGP,
    mll_cls=ExactMarginalLogLikelihood,
    cv_folds=cv_folds,
)

from matplotlib import pyplot as plt
%matplotlib inline

posterior = cv_results.posterior
mean = posterior.mean
cv_error = ((cv_folds.test_Y.squeeze() - mean.squeeze()) ** 2).mean()
print(f"Cross-validation error: {cv_error : 4.2}")

# get lower and upper confidence bounds
lower, upper = posterior.mvn.confidence_region()

# scatterplot of predicted versus test
_, axes = plt.subplots(1, 1, figsize=(6, 4))
plt.plot([-1.5, 1.5], [-1.5, 1.5], 'k', label="true objective", linewidth=2)

axes.set_xlabel("Actual")
axes.set_ylabel("Predicted")

axes.errorbar(
    x=cv_folds.test_Y.numpy().flatten(), 
    y=mean.numpy().flatten(), 
    xerr=1.96*sigma,
    yerr=((upper-lower)/2).numpy().flatten(),
    fmt='*'
);

model = cv_results.model
with torch.no_grad():
    # evaluate the models at a series of points for plotting 
    plot_x = torch.linspace(0, 1, 101).view(1, -1, 1).repeat(cv_folds.train_X.shape[0], 1, 1)
    posterior = model.posterior(plot_x)
    mean = posterior.mean
    
    # get lower and upper confidence bounds
    lower, upper = posterior.mvn.confidence_region()
    plot_x.squeeze_()

_, axes = plt.subplots(1, 1, figsize=(6, 4))

# plot the 12th CV fold
num = 12 

# plot the training data in black
axes.plot(
    cv_folds.train_X[num - 1].detach().numpy(), 
    cv_folds.train_Y[num - 1].detach().numpy(), 
    'k*'
)

# plot the test data in red
axes.plot(
    cv_folds.test_X[num - 1].detach().numpy(), 
    cv_folds.test_Y[num - 1].detach().numpy(), 
    'r*'
)

# plot posterior means as blue line
axes.plot(plot_x[num - 1].numpy(), mean[num-1].numpy(), 'b')

# shade between the lower and upper confidence bounds
axes.fill_between(
    plot_x[num - 1].numpy(), 
    lower[num - 1].numpy(), 
    upper[num - 1].numpy(), 
    alpha=0.5
);


