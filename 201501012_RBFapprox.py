import numpy as np
import matplotlib.pyplot as plt
from sys import argv
import xlrd


def featureNormalize(X):
    X_norm = np.divide((X - np.mean(X, axis=0)), np.std(X, axis=0))

    return X_norm


def activation(a, mu, spread):
    # here it is RBF function
    a = np.exp(-np.sum(np.multiply((a - mu), (a - mu)), axis=1) / (2 * spread * spread))
    return a


def rbf(a, mu, spread):
    # here it is RBF function
    a = np.exp(-np.sum(np.multiply((a - mu), (a - mu)), axis=1) / (2 * spread * spread))
    return a


def feed_forward(xtrain, weights1, bias1, weights2, bias2):
    a1 = np.dot(xtrain, np.transpose(weights1)) + bias1
    h1 = activation(a1)
    a2 = np.dot(h1, np.transpose(weights2)) + bias2
    prd = (a2)
    return prd


def accuracy_rmse(xtrain, weights1, bias1, weights2, bias2, y):
    prd = (feed_forward(xtrain, weights1, bias1, weights2, bias2))
    return float(100 * np.sum(prd == y)) / y.shape[0]


def k_means(xtrain, K):
    clusters = xtrain[np.random.choice(xtrain.shape[0], size=K, replace=False)]
    err = 999;
    if visualize:
        import matplotlib.pyplot as plt
        plt.ion()
        plt.show()
        plt.plot(xtrain[:, 0], xtrain[:, 1], 'o')
        plt.plot(clusters[:, 0], clusters[:, 1], 'x')
        plt.draw()

    olderr = err + 100
    while err >= 0.05:
        dis = np.zeros((K, xtrain.shape[0]))
        for i in range(K):
            dis[i] = np.sum(np.multiply((xtrain - clusters[i]), ((xtrain - clusters[i]))), axis=1)
        assignment = np.argmin(dis, axis=0)

        oldones = clusters
        for i in range(K):
            tot_in_c = float(np.sum(assignment == i))
            if tot_in_c > 0:
                clusters[i] = np.sum(xtrain[assignment == i], axis=0) / tot_in_c
        err = 0
        for i in range(K):
            err = err + np.sum(np.multiply((xtrain[assignment == i] - clusters[i]), ((xtrain[assignment == i] - clusters[i]))))
        if visualize:
            print err
            plt.pause(0.5)
            for i in range(K):
                vap = xtrain[assignment == i]
                plt.plot(vap[:, 0], vap[:, 1], 'o')
            plt.plot(clusters[:, 0], clusters[:, 1], 'x')
            plt.draw()
        if np.absolute(olderr - err) <= 0.01:
            break;
        olderr = err

    return clusters


def train_network(xtrain, ytrain, K):
    clusters = k_means(xtrain, K)
    #clusters= xtrain[np.random.choice(xtrain.shape[0], size=K,replace=False)]
    #new_dim = np.zeros((xtrain.shape[0], K))
    cls_dis = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            cls_dis[i, j] = np.power(np.sum(np.multiply(clusters[i] - clusters[j], clusters[i] - clusters[j])), 0.5)
    max_d = np.max(cls_dis)
    spread = max_d / float(np.power(K, 0.5))
    new_dim = np.zeros((xtrain.shape[0], K))
    for i in range(K):
        new_dim[:, i] = rbf(xtrain, clusters[i], spread)

    saved_dim = new_dim
    new_dim = np.concatenate((new_dim, np.ones((new_dim.shape[0], 1))), axis=1)
    answer = np.dot(np.dot(np.linalg.pinv(np.dot(np.transpose(new_dim), new_dim)), np.transpose(new_dim)), ytrain)
    weights1 = answer[0:answer.shape[0] - 1, :]
    bias1 = answer[-1, :]
    weights1 = weights1.T
    bias1 = bias1.T
    a1 = np.dot(saved_dim, np.transpose(weights1)) + bias1
    prd = (a1)

    # new_dim = np.zeros((xtest.shape[0], K))
    # for i in range(K):
    #     new_dim[:, i] = rbf(xtest, clusters[i], spread)
    # a1 = np.dot(new_dim, np.transpose(weights1)) + bias1
    # tprd = (a1)

    print "\n\nRMSE error ::\n"
    print "on training data ::"
    print np.power((np.sum(np.multiply(prd - ytrain, prd - ytrain)) / xtrain.shape[0]), 0.5)
    print "\n"
    mean = np.sum(prd - ytrain) / xtrain.shape[0]
    print "mean"
    print mean
    print "deviation"
    print np.power((np.sum(np.multiply(prd - ytrain - mean, prd - ytrain - mean)) / xtrain.shape[0]), 0.5)
    plt.scatter(ytrain, prd)
    plt.title('pseudo inv Training Actual function vs true function for ' + str(K) + ' RBF centers')
    plt.legend(('True function', 'Approximated function function',), loc='upper left')

    plt.show()

    plt.plot(range(1,ytrain[0:100].shape[0]+1,1),ytrain[0:100])
    plt.title('original')
    plt.hold(True)
    plt.legend(('True function', 'Approximated function function',), loc='upper left')

    plt.plot(range(1, prd[0:100].shape[0]+1,1), prd[0:100])
    plt.title('approximated')
    plt.legend(('True function', 'Approximated function function',), loc='upper left')

    plt.show()


training_file = argv[1]
testing_file = argv[2]

# training_file = 'fin_29.xlsx'
# testing_file = 'fin_test_s29.xlsx'

# training_file = 'SI_40.xlsx'
# testing_file = 'SI_test_s40.xlsx'

data = xlrd.open_workbook(training_file).sheet_by_index(0)
train = []
for row in range (data.nrows):
    _row=[]
    for col in range (data.ncols):
        _row.append(data.cell_value(row, col))
    train.append(_row)

data = xlrd.open_workbook(testing_file).sheet_by_index(0)
test = []
for row in range (data.nrows):
    _row=[]
    for col in range (data.ncols):
        _row.append(data.cell_value(row, col))
    test.append(_row)


visualize = False
feat = len(train[0]) - 1;
output = 1
K = 40

train = np.array(train)
test = np.array(test)

xtrain = train[:, 0:feat]
ytrain = train[:, feat]
xtest = test[:, 0:feat]

total_train = xtrain.shape[0]
total_val = int(total_train * 0.1)

# 10-fold validation
xval = xtrain[total_train - total_val:total_train, :]
xtrain = xtrain[0:total_train - total_val, :]
yval = ytrain[total_train - total_val:total_train]
ytrain = ytrain[0:total_train - total_val]

weights1 = (np.random.rand(output, K) - .5) / 1000
bias1 = (np.random.rand(1, output) - .5) / 1000

#feature normalization
xtrain = featureNormalize(xtrain)
xval = featureNormalize(xval)
xtest = featureNormalize(xtest)

ytrain = np.reshape(ytrain, (ytrain.shape[0], 1))
yval = np.reshape(yval, (yval.shape[0], 1))

train_network(xtrain, ytrain, K)