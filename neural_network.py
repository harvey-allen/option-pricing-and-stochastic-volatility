import numpy as np
import json
from math import gamma
from scipy.interpolate import CubicSpline


def elu(x):
    val = x
    val[x < 0] = np.exp(x[x < 0]) - 1
    return val


class NeuralNetwork:
    def __init__(self, fileName):
        self.weights = []
        self.biases = []
        self.scaleMeanIn = []
        self.scaleStdIn = []
        self.scaleMeanOut = []
        self.scaleStdOut = []
        with open(fileName) as json_file:
            tmp = json.load(json_file)
        nLayers = int((len(tmp) - 4) / 2)
        for i in range(nLayers):
            self.weights.append(np.transpose(tmp[2 * i]))
            self.biases.append(np.array(tmp[2 * i + 1]).reshape(-1, 1))
        self.scaleMeanIn = np.array(tmp[-4]).reshape(-1, 1)
        self.scaleStdIn = np.sqrt(np.array(tmp[-3]).reshape(-1, 1))
        self.scaleMeanOut = np.array(tmp[-2]).reshape(-1, 1)
        self.scaleStdOut = np.sqrt(np.array(tmp[-1]).reshape(-1, 1))

    def Eval(self, x):
        nLayers = len(self.weights)
        val = (x - self.scaleMeanIn) / self.scaleStdIn
        for i in range(0, nLayers - 1):
            val = elu(np.dot(self.weights[i], val) + self.biases[i])
        val = np.dot(self.weights[nLayers - 1], val) + self.biases[nLayers - 1]
        return self.scaleStdOut * val + self.scaleMeanOut


class NeuralNetworkPricer:
    def __init__(self, contracts_folder, weights_folder, model_name):
        self.nn = []
        self.idx_in = []
        self.idx_out = []
        self.lb = []
        self.ub = []
        self.label = model_name
        self.T = np.loadtxt("").reshape(-1, 1)
        self.k = np.loadtxt("").reshape(-1, 1)
        Txi = np.array(
            [
                0.0025,
                0.0050,
                0.0075,
                0.0100,
                0.0125,
                0.0150,
                0.0175,
                0.0200,
                0.0400,
                0.0600,
                0.0800,
                0.1000,
                0.1200,
                0.1400,
                0.1600,
                0.2800,
                0.4000,
                0.5200,
                0.6400,
                0.7600,
                0.8800,
                1.0000,
                1.2500,
                1.5000,
                1.7500,
                2.0000,
                3.0000,
            ]
        )
        json_files = [
            ""
        ]
        idxOutStart = 0
        for i in range(len(json_files)):
            self.nn.append(
                NeuralNetwork(weights_folder + "\\" + model_name + json_files[i])
            )
            self.idx_in.append(np.arange(0, self.nn[i].scaleMeanIn.shape[0]))
            idxOutEnd = idxOutStart + self.nn[i].scaleMeanOut.shape[0]
            self.idx_out.append(np.arange(idxOutStart, idxOutEnd))
            idxOutStart = idxOutEnd
        self.lb = np.concatenate(
            (np.array([0, 0.1, -1]), pow(0.05, 2) * np.ones(28))
        ).reshape(-1, 1)
        self.ub = np.concatenate((np.array([0.5, 1.25, 0]), np.ones(28))).reshape(
            -1, 1
        )
        self.Txi = np.concatenate((np.array([0]), Txi))

    def EvalInGrid(self, x: list):
        if any(x < self.lb) or any(x > self.ub):
            raise Exception(
                "NeuralNetworkPricer: EvalInGrid: Parameter bounds are violated."
            )
        nNetworks = len(self.nn)
        nPts = self.k.shape[0]
        iv = np.zeros(nPts).reshape(-1, 1)
        for i in range(0, nNetworks):
            iv[self.idx_out[i]] = self.nn[i].Eval(x[self.idx_in[i]])

        return iv

    def AreContractsInDomain(self, kq, Tq):
        if not kq.shape == Tq.shape:
            raise Exception(
                "NeuralNetworkPricer: AreContractsInDomain: Shape of input vectors are not the same."
            )
        uniqT = np.unique(Tq)
        uniqTGrid = np.unique(self.T)
        minTGrid = np.min(uniqTGrid)
        maxTGrid = np.max(uniqTGrid)
        idxValid = np.ones((len(kq), 1), dtype=bool)
        for i in range(0, len(uniqT)):
            idxT = Tq == uniqT[i]
            if uniqT[i] > maxTGrid or uniqT[i] < minTGrid:
                idxValid[idxT] = False
            else:
                if uniqT[i] == maxTGrid:
                    idxAbove = len(uniqTGrid) - 1
                else:
                    idxAbove = np.argmax(uniqTGrid > uniqT[i])
                idxBelow = idxAbove - 1
                idxGridBelow = self.T == uniqTGrid[idxBelow]
                idxGridAbove = self.T == uniqTGrid[idxAbove]
                idxValid[idxT] = (
                    kq[idxT]
                    >= np.max(
                        [np.min(self.k[idxGridBelow]), np.min(self.k[idxGridAbove])]
                    )
                ) & (
                    kq[idxT]
                    <= np.min(
                        [np.max(self.k[idxGridBelow]), np.max(self.k[idxGridAbove])]
                    )
                )
        return np.ravel(idxValid)

    def Eval(self, x, kq, Tq):
        ivGrid = self.EvalInGrid(x)
        if not all(self.AreContractsInDomain(kq, Tq)):
            raise Exception(
                "NeuralNetworkPricer: Eval: Some contracts violate the neural network domain."
            )
        ivGrid = self.EvalInGrid(x)
        nPts = kq.shape[0]
        iv = np.zeros((nPts, 1))
        uniqT = np.unique(Tq)
        uniqTGrid = np.unique(self.T)
        maxTGrid = max(uniqTGrid)
        for i in range(0, len(uniqT)):
            idxT = Tq == uniqT[i]
            if uniqT[i] == maxTGrid:
                idxAbove = len(uniqTGrid) - 1
            else:
                idxAbove = np.argmax(uniqTGrid > uniqT[i])
            idxBelow = idxAbove - 1
            T_above = uniqTGrid[idxAbove]
            T_below = uniqTGrid[idxBelow]
            idxGridBelow = self.T == uniqTGrid[idxBelow]
            idxGridAbove = self.T == uniqTGrid[idxAbove]
            iv_below_grid = ivGrid[idxGridBelow]
            iv_above_grid = ivGrid[idxGridAbove]
            k_below_grid = self.k[idxGridBelow]
            k_above_grid = self.k[idxGridAbove]
            idxSort_below = np.argsort(k_below_grid)
            idxSort_above = np.argsort(k_above_grid)
            spline_lower = CubicSpline(
                k_below_grid[idxSort_below],
                iv_below_grid[idxSort_below],
                bc_type="natural",
            )
            spline_upper = CubicSpline(
                k_above_grid[idxSort_above],
                iv_above_grid[idxSort_above],
                bc_type="natural",
            )
            iv_below = spline_lower(kq[idxT])
            iv_above = spline_upper(kq[idxT])
            frac = (uniqT[i] - T_below) / (T_above - T_below)
            iv[idxT] = np.sqrt(
                (
                    (1 - frac) * T_below * pow(iv_below, 2)
                    + frac * T_above * pow(iv_above, 2)
                )
                / uniqT[i]
            )

        return iv


def CheckNonNeqReqTheta(v0, H, t, theta):
    val = theta + (v0 / gamma(1 / 2 - H)) * pow(t, -H - 1 / 2)
    val[-1] = theta[-1]
    valid = all(val >= 0)

    return [valid, val]


def GetThetaFromXi(v0, H, t, xi):
    t_ext = np.concatenate((np.zeros((1, 1)), t))
    n = len(xi)
    theta = np.zeros((n, 1))
    for i in range(0, n):
        if i == 0:
            wii = (1 / gamma(H + 3 / 2)) * pow(t[i], H + 1 / 2)
            wik_theta_sum = 0
        else:
            wik_theta_sum = 0
            for j in range(1, i + 1):
                wik_theta_sum = (
                    wik_theta_sum
                    + (1 / gamma(H + 3 / 2))
                    * (
                        pow(t[i] - t_ext[j - 1], H + 1 / 2)
                        - pow(t[i] - t_ext[j], H + 1 / 2)
                    )
                    * theta[j - 1]
                )
            wii = (1 / gamma(H + 3 / 2)) * pow(t[i] - t[i - 1], H + 1 / 2)
        theta[i] = (xi[i] - v0 - wik_theta_sum) / wii

    return theta
