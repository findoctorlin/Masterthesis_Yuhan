import torch
import gpytorch
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.kernels import RBFKernel, ScaleKernel, MaternKernel
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution, DeltaVariationalDistribution, MeanFieldVariationalDistribution
from gpytorch.distributions import MultivariateNormal
from gpytorch.models import ApproximateGP, GP
from gpytorch.mlls import VariationalELBO, AddedLossTerm
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.models.deep_gps import DeepGPLayer, DeepGP
from gpytorch.mlls import DeepApproximateMLL


class DeepGPHiddenLayer(DeepGPLayer):    #returns samples of q(f)
    '''
    input_dims: dimensions of input data
    output_dims: number of GP for each hidden layer
    kernel_type: GP kernel type, e.g. RBF, Matern, RQ
    '''
    def __init__(self, input_dims, output_dims, num_inducing=128, mean_type='constant', kernel_type='rbf'):
        # output layer
        if output_dims is None:
            inducing_points = torch.randn(num_inducing, input_dims)#random initialization from N(0,1)
            batch_shape = torch.Size([])#default
        # hidden layer
        else: 
            inducing_points = torch.randn(output_dims, num_inducing, input_dims) #output_dims(2): check FIGURE 1 in ETH paper, first hidden layer has 2 GPs
            batch_shape = torch.Size([output_dims])
        # multivariate normal distribution with a full covariance matrix
        variational_distribution = CholeskyVariationalDistribution(    #q(u):marginal variational distribution
            num_inducing_points = num_inducing,
            batch_shape = batch_shape #define the shape of batch,Z.B. torch.Size([4,3])
        )

        variational_strategy = VariationalStrategy(    #q(f)=integral(p(f|u)q(u))du, defines how to compute q(f)
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True #Whether or not the inducing point locations 𝐙 should be learned (i.e. are they parameters of the model).
        )

        super(ToyDeepGPHiddenLayer, self).__init__(variational_strategy, input_dims, output_dims)    #inherit _init_() from father “DeepGPLayer”

        self.variational_distribution = variational_distribution

        if mean_type == 'constant':
            self.mean_module = ConstantMean(batch_shape=batch_shape)
        else:
            self.mean_module = LinearMean(input_dims)

        if kernel_type == 'rbf':
            self.covar_module = ScaleKernel(
                RBFKernel(batch_shape=batch_shape, ard_num_dims=input_dims),
                batch_shape=batch_shape, ard_num_dims=None
            )
        elif kernel_type == 'matern0.5':
            self.covar_module = ScaleKernel(
                MaternKernel(nu=0.5, batch_shape=batch_shape, ard_num_dims=input_dims),
                batch_shape=batch_shape, ard_num_dims=None
            )
        elif kernel_type == 'matern1.5':
            self.covar_module = ScaleKernel(
                MaternKernel(nu=1.5, batch_shape=batch_shape, ard_num_dims=input_dims),
                batch_shape=batch_shape, ard_num_dims=None
            )
        else:
            print("Not supported Kernel")
            raise


    def forward(self, x):    #returns posterior
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x) #Constructs a multivariate normal random variable, based on mean and covariance

    def __call__(self, x, *other_inputs, **kwargs):
        """
        Overriding __call__ isn't strictly necessary, but it lets us add concatenation based skip connections
        easily. For example, hidden_layer2(hidden_layer1_outputs, inputs) will pass the concatenation of the first
        hidden layer's outputs and the input data to hidden_layer2.
        """
        if len(other_inputs):
            if isinstance(x, gpytorch.distributions.MultitaskMultivariateNormal):
                x = x.rsample()

            processed_inputs = [
                inp.unsqueeze(0).expand(gpytorch.settings.num_likelihood_samples.value(), *inp.shape)
                for inp in other_inputs
            ]

            x = torch.cat([x] + processed_inputs, dim=-1)

        return super().__call__(x, are_samples=bool(len(other_inputs)))


# in this example we build a 2-layers DGP
class DeepGP(DeepGP):#this class should contain "DeepGPLayer" modules, which we defined already above
    '''
    DGP Module which passes forward through the various DGP Layers
    Inspired by the archtiecture of Deep Neural Network, the idea is to stack multiple GPs horizontally and vertically.

    Args:
        train_x_shape: Train data shape
        output_dims: list with dimensionality for each hidden layer
        num_inducing: Size of the variational distribution (size of variational mean)
        kernel_type: GP kernel type, e.g. RBF, Matern, RQ

    Exp:
    # 2-layer DGP with 5 GPs in each layer
    >>> import torch
    >>> model = DeepGP(train_x_shape=X_train.shape, output_dims=[5, 5])
    '''
    def __init__(self, train_x_shape, output_dims, num_inducing=128, kernel_type='rbf'):
        # L hidden layers of a L+1 layer GP
        output_dims.append(None)  # The last layer has None output_dims

        # As in Salimbeni et al. 2017 finds that using a linear mean for the hidden layer improves performance
        means = (len(output_dims) - 1) * ['linear'] + ['constant']  # Only the last layer with constant mean

        hidden_layers = torch.nn.ModuleList([DeepGPHiddenLayer(
            input_dims=train_x_shape[-1],
            output_dims=output_dims[0],
            num_inducing=num_inducing,
            mean_type=means[0],
            kernel_type=kernel_type
            )])

        for layer in range(1, len(output_dims)):
            hidden_layers.append(DeepGPHiddenLayer(
                input_dims=hidden_layers[-1].output_dims,
                output_dims=output_dims[layer],
                num_inducing=num_inducing,
                mean_type=means[layer],
                kernel_type=kernel_type
                ))  

        super().__init__()

        self.hidden_layers = hidden_layers
        # self.last_layer = last_layer
        self.likelihood = GaussianLikelihood()
        self.inducing_value = self.hidden_layer.variational_distribution() # test for inducing point

    def forward(self, inputs):    #responsible for forwarding through the various layers
        hidden_rep1 = self.hidden_layer(inputs)
        output = self.last_layer(hidden_rep1)
        return output

    def get_inducing(self): #新加的
        return self.inducing_value

    def predict(self, test_loader): #returns the mean, variance, and log marginal likelihood of the predictions
        with torch.no_grad():
            mus = []
            variances = []
            lls = [] #log likelihood score
            for x_batch, y_batch in test_loader:
                preds = self.likelihood(self(x_batch)) #mapping from latent function value f(X) to observed labels y
                # prediction.append(preds)
                mus.append(preds.mean) #length of mus is 102
                # print("preds.mean.shape:", preds.mean.shape)
                variances.append(preds.variance)
                # print("preds.variance.shape:", preds.variance.shape)
                lls.append(model.likelihood.log_marginal(y_batch, model(x_batch)))

        return torch.cat(mus, dim=-1), torch.cat(variances, dim=-1), torch.cat(lls, dim=-1), preds # in tutorial didn't return preds