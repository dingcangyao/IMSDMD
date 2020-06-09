import numpy as np


def dimensionalityReduction(X_dmd, n_components):
    # Phi = np.zeros((n_components, X_dmd.shape[0]))
    # d = 32
    # for i in range(X_dmd.shape[0]):
    #     col_idx = np.random.permutation(range(n_components))
    #     Phi[col_idx[0:d], i] = 1
    # X_dmd_trans = Phi.dot(X_dmd)

    # Phi = np.random.randint(0,2, (n_components, X_dmd.shape[0]))
    # Phi[Phi == 0] = -1
    # X_dmd_trans = Phi.dot(X_dmd)


    # Phi = np.random.RandomState(1500).randn(n_components, X_dmd.shape[0])
    # X_dmd_trans = Phi.dot(X_dmd)

    t = 2
    r = 5
    Phi = np.random.RandomState(0).randn(n_components, X_dmd.shape[0])
    u, s, v = np.linalg.svd(Phi)
    avg = np.mean(s)
    j = len(s[s > avg])
    Phi[:, 0:j] *= t
    u, s, v = np.linalg.svd(Phi)
    s[ :r] = 1
    s = np.diag(s)
    diag = np.zeros((u.shape[1], v.shape[0]))
    diag[0: s.shape[0], 0:s.shape[1]] = s
    Phi = (u.dot(diag)).dot(v.T)

    X_dmd_trans = Phi.dot(X_dmd)

    #
    # pca = PCA(n_components=n_components, whiten=True)
    # X_dmd_trans = pca.fit_transform(X_dmd.T)
    return X_dmd_trans.T
