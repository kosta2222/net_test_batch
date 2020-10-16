import numpy as np
#This function returns the mini-batches given the inputs and targets:

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert inputs.shape[0] == targets.shape[0]
    if shuffle:
        indices = np.arange(inputs.shape[0])
        np.random.shuffle(indices)
    for start_idx in range(0, inputs.shape[0] - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]
# and this tells you how to use that for training:

for n in range(n_epochs):
    for batch in iterate_minibatches(X, Y, batch_size, shuffle=True):
        x_batch, y_batch = batch
        l_train, acc_train = f_train(x_batch, y_batch)

    l_val, acc_val = f_val(Xt, Yt)
    logging.info('epoch ' + str(n) + ' ,train_loss ' + str(l_train) + ' ,acc ' + str(acc_train) + ' ,val_loss ' + str(l_val) + ' ,acc ' + str(acc_val))
# Obviously you need to define the f_train, f_val and other functions yourself given the optimisation library (e.g. Lasagne, Keras) you are using.