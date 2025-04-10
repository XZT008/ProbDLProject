from data import *
from GP import GP_Wrapper, GP_Wrapper_pyro, SAASBO_Wrapper, ADDGP_Wrapper


if __name__ == "__main__":
    N_tr, N_te = 200, 100
    D = 50
    SEED = 0
    func = FuncStybTang_V1(D, maximize=True)
    dst = BayesOptDataset(func, N_tr+N_te, 'lhs', SEED)
    X, Y = dst.get_data()
    X_tr, X_te = X[:N_tr], X[N_tr:]
    y_tr, y_te = Y[:N_tr], Y[N_tr:]


    # Normal GP, with MLE training
    """
    simple_GP = GP_Wrapper(X_tr, y_tr)
    simple_GP.train_model()
    pred_y = simple_GP.pred(X_te)
    MSE = ((pred_y.squeeze() - y_te.squeeze())**2).mean()
    """

    # Additive GP, MLE
    """
    Add_GP = ADDGP_Wrapper(X_tr, y_tr)
    Add_GP.train_model()
    pred_y = Add_GP.pred(X_te)
    MSE = ((pred_y.squeeze() - y_te.squeeze()) ** 2).mean()
    """

    # SaasBO
    """
    saasGP = SAASBO_Wrapper(X_tr, y_tr)
    saasGP.train_model()
    pred_y = saasGP.pred(X_te)
    MSE = ((pred_y.squeeze() - y_te.squeeze()) ** 2).mean()
    """

    # GP, with NUTS
    nuts_gp = GP_Wrapper_pyro(X_tr, y_tr)
    nuts_gp.train_model()
    pred_y = nuts_gp.pred(X_te)
    MSE = ((pred_y.squeeze() - y_te.squeeze()) ** 2).mean()
    print()


