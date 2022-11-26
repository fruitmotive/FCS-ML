from collections import defaultdict
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeRegressor

#########################################################################################################

def score(clf, x, y):
    return roc_auc_score(y == 1, clf.predict_proba(x)[:, 1])

#########################################################################################################

class Boosting:

    def __init__(
            self,
            base_model_params: dict = None,
            n_estimators: int = 10,
            learning_rate: float = 0.1,
            subsample: float = 0.3,
            early_stopping_rounds: int = None,
            plot: bool = False,
    ):
        self.base_model_class = DecisionTreeRegressor
        self.base_model_params: dict = {} if base_model_params is None else base_model_params

        self.n_estimators: int = n_estimators

        self.models: list = []
        self.gammas: list = []

        self.learning_rate: float = learning_rate
        self.subsample: float = subsample

        self.early_stopping_rounds: int = early_stopping_rounds

        self.plot: bool = plot

        self.history = defaultdict(list)

        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self.loss_fn = lambda y, z: -np.log(self.sigmoid(y * z)).mean()
        self.loss_derivative = lambda y, z: -y * self.sigmoid(-y * z)
        self.loss_derivative2 = lambda y, z: y ** 2 * self.sigmoid(-y * z) * (1 - self.sigmoid(-y * z))

#########################################################################################################

    def fit_new_base_model(self, x, y, predictions):

        rands = np.random.permutation(y.shape[0])[:int(self.subsample * y.shape[0])]
        s = - self.loss_derivative(y[rands], predictions[rands])

        new_base_model = self.base_model_class(**self.base_model_params).fit(x[rands], s)
        new_predictions = new_base_model.predict(x)

        optimal_gamma = self.find_optimal_gamma(y, predictions, new_predictions)

        self.gammas.append(optimal_gamma)
        self.models.append(new_base_model)

        return optimal_gamma * new_predictions

#########################################################################################################

    def fit(self, x_train, y_train, x_valid, y_valid):

        train_predictions = np.zeros(y_train.shape[0])
        valid_predictions = np.zeros(y_valid.shape[0])
        
        n_min, val_loss_min = 0, np.inf

        for i in range(1, self.n_estimators + 1):
            train_predictions += self.learning_rate * self.fit_new_base_model(x_train, y_train, train_predictions)
            valid_predictions += self.learning_rate * self.gammas[-1] * self.models[-1].predict(x_valid)

            self.history['train_score'].append(self.loss_fn(y_train, train_predictions))
            val_loss = self.loss_fn(y_valid, valid_predictions)
            self.history['valid_score'].append(val_loss)

            if self.early_stopping_rounds is not None:
                if val_loss < val_loss_min:
                    n_min = i
                    val_loss_min = val_loss

                if i - n_min >= self.early_stopping_rounds:
                    self.n_estimators = i
                    break

        if self.plot:
            plt.plot(np.arange(1, self.n_estimators + 1), self.history['train_score'], label='train_score')
            plt.plot(np.arange(1, self.n_estimators + 1), self.history['valid_score'], label='valid_score')

            plt.xlabel('number of models')
            plt.ylabel("score")
            plt.legend()
            plt.show()

#########################################################################################################

    def predict_proba(self, x):

        result = np.zeros((x.shape[0], 2))
        aux = np.zeros(x.shape[0])

        for gamma, model in zip(self.gammas, self.models):
            aux += self.learning_rate * gamma * model.predict(x)
        
        aux = self.sigmoid(aux)
        result[:, 0] = 1 - aux
        result[:, 1] = aux

        return result

#########################################################################################################

    def find_optimal_gamma(self, y, old_predictions, new_predictions) -> float:

        gammas = np.linspace(start=0, stop=1, num=100)
        losses = [self.loss_fn(y, old_predictions + gamma * new_predictions) for gamma in gammas]

        return gammas[np.argmin(losses)]

#########################################################################################################

    def score(self, x, y):

        return score(self, x, y)

#########################################################################################################

    @property
    def feature_importances_(self):

        pass
