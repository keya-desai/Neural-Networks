from numpy import asarray, delete, vstack, insert, array, mat, loadtxt, zeros, exp, multiply, sum, square, sqrt, hstack, \
    unique, ones, argmax, power, argmin
from numpy.random import random, choice, randint
from numpy.linalg import pinv
from sys import argv
from math import floor
import xlrd


def target_to_coded_class_labels(t, n_classes):
    m = -1 * mat(ones((n_classes, t.shape[1])))
    for i in range(t.shape[1]):
        m[int(t[0, i] - 1), i] = 1
    return m


def coded_class_labels_to_target(m):
    return argmax(m, axis=0) + 1


def confusion_matrix(target, output):
    confusion = mat(zeros([len(unique(list(target))), len(unique(list(target)))]))
    for i in range(target.shape[1]):
        confusion[int(target[0, i]) - 1, int(output[0, i]) - 1] += 1
    return confusion


def accuracy(confusion_matrix):
    overall_accuracy = (confusion_matrix.trace() / confusion_matrix.sum()) * 100
    average_accuracy = 0
    geometric_mean_accuracy = 1
    for i in range(confusion_matrix.shape[0]):
        average_accuracy += (confusion_matrix[i, i] / confusion_matrix.sum(axis=1)[i, 0])
        geometric_mean_accuracy *= (100.0 * (confusion_matrix[i, i] / confusion_matrix.sum(axis=1)[i, 0]))
    average_accuracy *= (100.0 / confusion_matrix.shape[0])
    geometric_mean_accuracy = geometric_mean_accuracy ** (1.0 / confusion_matrix.shape[0])
    return (overall_accuracy[0, 0], average_accuracy, geometric_mean_accuracy)


def error(target, output, loss_function):
    if loss_function == 'least_squares':
        return 0.5 * sum(square(target - output))
    if loss_function == 'fourth_power':
        return 0.25 * sum(square(square(target - output)))
    if loss_function == 'modified_least_squares':
        output[output > 1] = 1
        output[output < -1] = -1
        return 0.5 * sum(square(target - output))
    if loss_function == 'modified_fourth_power':
        output[output > 1] = 1
        output[output < -1] = -1
        return 0.25 * sum(square(square(target - output)))


def stratified_folds(data, k):
    data = data.copy()
    data[:, -1] = data[:, -1] - 1
    folds = []
    classwise_data = []
    classwise_data_distribution = []
    n_classes = len(unique(array(data[:, -1]).flatten()))
    n_data = data.shape[0]
    size_1_fold = floor(float(n_data) / k)
    for i in range(n_classes):
        classdata = data[array(data[:, -1] == i).flatten(), :].copy()
        classdata = insert(classdata, classdata.shape[1], 0, axis=1)
        classdata[:, -1] = mat(randint(low=0, high=k, size=classdata.shape[0])).T
        classwise_data.append(classdata)
        classwise_data_distribution.append(classwise_data[i].shape[0])
    for i in range(k):
        for j in range(n_classes):
            if j == 0:
                fold = classwise_data[j][array(classwise_data[j][:, -1] == i).flatten(), :]
            else:
                fold = vstack([fold, classwise_data[j][array(classwise_data[j][:, -1] == i).flatten(), :]])
        fold = delete(fold, -1, 1)
        folds.append(fold.copy())
    return folds


training_file = argv[1]
testing_file = argv[2]
classes = argv[3]
n_epoch = argv[4]
n_fold = argv[5]
learning_rate = argv[6]


classes = int(classes)
n_epoch = int(n_epoch)
n_fold = int(n_fold)
learning_rate = float(learning_rate)

data = xlrd.open_workbook(training_file).sheet_by_index(0)
data_train = []
for row in range (data.nrows):
    _row=[]
    for col in range (data.ncols):
        _row.append(data.cell_value(row, col))
    data_train.append(_row)

data = xlrd.open_workbook(testing_file).sheet_by_index(0)
data_test = []
for row in range (data.nrows):
    _row=[]
    for col in range (data.ncols):
        _row.append(data.cell_value(row, col))
    data_test.append(_row)
data_test = mat(data_test);

n_train = len(data_train)
n_test = data_test.shape[0]

for i in range(n_train):
    o=data_train[i][-classes:]
    data_train[i] = data_train[i][: -classes]
    for j in range(len(o)):
        if o[j]==1:
            data_train[i].append(j+1);
            break;
data_train = mat(data_train)


# Initialize the algorithm parameters
inp = data_train.shape[1] - 1
hid = inp + 11
out = len(unique(list(data_train[:, -1])))
cluster_centers = False
mu = 1e-2
n_epoch = 2000
n_fold = 3
learning_rate_W_zy = 1e-4
learning_rate_center = 1e-4
learning_rate_spread = 1e-4
loss_function = 'least_squares'
print 'Input = ' + str(inp)
print 'Hidden = ' + str(hid)
print 'Output = ' + str(out)
print 'Learning rate = ' + str(learning_rate_center)
print 'Epoch = ' + str(n_epoch)
print 'Folds = ' + str(n_fold)
# Initialize the algorithm parameters

# Initialize weights
data_fold = stratified_folds(data_train, n_fold)

W_zy = 0.001 * (mat(random(size=(hid, out))) * 2.0 - 1.0)

# Network : X -> Z -> Y
# Train the network

for k in range(n_fold):

    # Prepare training data and CV data
    data = vstack(data_fold[:k] + data_fold[k + 1:])
    cv = data_fold[k]
    n_train = data.shape[0]
    # Extract input matrix from training data
    x = data[:, 0:inp].T
    # Extract target outputs from training data
    target = data[:, -1].T
    t = target_to_coded_class_labels(target, out)  # Targets

    center = x[:, choice(x.shape[1], hid, replace=True)].copy()
    if cluster_centers:
        one = ones((1, hid))
        count = 2000
        while count > 0:
            X = x[:, randint(n_train)]
            center_to_update = argmin(power(sum(square(center - (X * one)), axis=0), 0.5))
            center[:, center_to_update] = (1 - mu) * center[:, center_to_update] + mu * X
            count -= 1
    d_max = 0.0
    for i in range(hid):
        for j in range(i):
            d = power(sum(power(center[:, i] - center[:, j], 2)), 0.5)
            if d_max < d:
                d_max = d
    spread = (d_max / sqrt(hid)) * mat(ones((1, hid))).T

    err = 0
    for n in range(n_epoch):
        X_in = x
        X_out = X_in

        ones_n_train = mat(ones(n_train)).T
        ones_hid = mat(ones(hid)).T
        Z_in = ((ones_hid * sum(power(X_out, 2), axis=0) + sum(power(center, 2), axis=0).T * ones_n_train.T - 2 * (
        center.T * X_out)) / (2 * power(spread, 2)))
        Z_out = exp(-Z_in)

        Y_in = W_zy.T * Z_out
        Y_out = Y_in

        DW_zy = - Z_out * (t - Y_out).T
        DW_center = - (((X_out * multiply((W_zy * (t - Y_out)), Z_out).T) - (multiply(center, sum(multiply((W_zy * (t - Y_out)), Z_out), axis=1).T))) / power(spread.T, 2))
        DW_spread = - sum(multiply(multiply(W_zy * (t - Y_out), Z_out), Z_in), axis=1) / spread

        W_zy = W_zy - learning_rate_W_zy * DW_zy
        center = center - learning_rate_center * DW_center
        spread = spread - learning_rate_spread * DW_spread

        err_this_epoch = error(t, Y_out, loss_function)
        err = err + err_this_epoch

        print str(float(n) / n_epoch * 100) + ' % : ' + str(sqrt(err_this_epoch / n_train)) + '\r',
    print ''

    X_in = cv[:, 0:inp].T
    X_out = X_in

    ones_n_train = mat(ones(X_in.shape[1])).T
    ones_hid = mat(ones(hid)).T
    Z_in = ((ones_hid * sum(power(X_out, 2), axis=0) + sum(power(center, 2), axis=0).T * ones_n_train.T - 2 * (
    center.T * X_out)) / (2 * power(spread, 2)))
    Z_out = exp(-Z_in)

    Y_in = W_zy.T * Z_out
    Y_out = Y_in

    t = target_to_coded_class_labels(cv[:, -1].T, out)

    print 'Error on testing fold : ' + str(sqrt(sum(square(t - Y_out)) / n_train))

# Validation on training data phase
x = data_train[:, 0:inp].T
target = data_train[:, -1].T - 1
t = target_to_coded_class_labels(target, out)

X_in = x
X_out = X_in

ones_n_train = mat(ones(X_in.shape[1])).T
ones_hid = mat(ones(hid)).T
Z_in = ((ones_hid * sum(power(X_out, 2), axis=0) + sum(power(center, 2), axis=0).T * ones_n_train.T - 2 * (
center.T * X_out)) / (2 * power(spread, 2)))
Z_out = exp(- Z_in)

Y_in = W_zy.T * Z_out
Y_out = Y_in

Y_out[Y_out > 0] = 1
Y_out[Y_out < 0] = -1

Y_out = coded_class_labels_to_target(Y_out)
t = coded_class_labels_to_target(t)

print 'Training data classification  : \n'

for j in list(asarray(Y_out)[0]):
    print j

confusion = confusion_matrix(t, Y_out)
(overall_accuracy, average_accuracy, geometric_mean_accuracy) = accuracy(confusion)

print '\n\n========== Trainig phase results ====================\n'
print 'No. of samples : ' + str(data_train.shape[0])
print 'Confusion matrix : \n' + str(repr(confusion))
print 'Overall accuracy : ' + str(overall_accuracy) + '%'
print 'Average accuracy : ' + str(average_accuracy) + '%'
print 'Geometric mean accuracy : ' + str(geometric_mean_accuracy) + '%'

# Testing phase
x = data_test[:, 0:inp].T

X_in = x
X_out = X_in

ones_n_train = mat(ones(X_in.shape[1])).T
ones_hid = mat(ones(hid)).T
Z_in = ((ones_hid * sum(power(X_out, 2), axis=0) + sum(power(center, 2), axis=0).T * ones_n_train.T - 2 * (
center.T * X_out)) / (2 * power(spread, 2)))
Z_out = exp(- Z_in)

Y_in = W_zy.T * Z_out
Y_out = Y_in

Y_out[Y_out > 0] = 1
Y_out[Y_out < 0] = -1

Y_out = coded_class_labels_to_target(Y_out)

print '\nTesting data classification  : \n'
for j in list(asarray(Y_out)[0]):
    print j

print '\n\n========== Testing phase results ====================\n'
print 'No. of samples : ' + str(data_train.shape[0])
