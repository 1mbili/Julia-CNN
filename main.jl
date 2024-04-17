using MLDatasets

# load training set
train_x, train_y = MNIST(split=:train)[:]

# load test set
test_x,  test_y  = MNIST(split=:test)[:]
