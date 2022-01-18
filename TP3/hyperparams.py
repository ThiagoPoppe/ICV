import keras

from keras.optimizers import SGD

from networks.Networks import LeNet, MyFirstCNN, MySecondCNN, MyFirstNN, MySecondNN
from networks.DataLoader import Cifar10

if __name__ == '__main__':
    # Abrindo arquivo para salvar os melhores hiper parâmetros
    f = open('hyperparams.txt', 'w')

    # Lendo os dados
    (train_images, train_labels), (test_images, test_labels) = Cifar10.read_data()

    # Definindo os possíveis learning rates, epochs e batch_sizes
    lr_list = [0.01, 0.001, 0.0001]
    epochs_list = [10, 20, 30]
    batch_list = [64, 128, 256]

    # *** LeNet ****
    best_acc = 0.0
    best_loss = 0.0
    best_params = []

    for lr in lr_list:
        for epoch in epochs_list:
            for batch_size in batch_list:
                model = LeNet.build(opt=SGD(lr=lr))
                model.fit(train_images, train_labels,
                          epochs=epoch,
                          batch_size=batch_size,
                          validation_data=(test_images, test_labels),
                          verbose=1)
                test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=0)
                if test_acc > best_acc:
                    best_loss = test_loss
                    best_acc = test_acc
                    best_params = [lr, epoch, batch_size]

    print('Best loss (LeNet):', best_loss, file=f)
    print('Best acc (LeNet):', best_acc, file=f)
    print('Best hyperparams (LeNet):', best_params, file=f)
    print('\n', file=f)

    # *** MyFirstNN ****

    best_acc = 0.0
    best_loss = 0.0
    best_params = []

    for lr in lr_list:
        for epoch in epochs_list:
            for batch_size in batch_list:
                model = MyFirstNN.build(opt=SGD(lr=lr))
                model.fit(train_images, train_labels,
                          epochs=epoch,
                          batch_size=batch_size,
                          validation_data=(test_images, test_labels),
                          verbose=1)
                test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=0)
                if test_acc > best_acc:
                    best_loss = test_loss
                    best_acc = test_acc
                    best_params = [lr, epoch, batch_size]

    print('Best loss (MyFirstNN):', best_loss, file=f)
    print('Best acc (LeMyFirstNN):', best_acc, file=f)
    print('Best hyperparams (MyFirstNN):', best_params, file=f)
    print('\n', file=f)

    # *** MySecondNN ****

    best_acc = 0.0
    best_loss = 0.0
    best_params = []

    for lr in lr_list:
        for epoch in epochs_list:
            for batch_size in batch_list:
                model = MySecondNN.build(opt=SGD(lr=lr))
                model.fit(train_images, train_labels,
                          epochs=epoch,
                          batch_size=batch_size,
                          validation_data=(test_images, test_labels),
                          verbose=1)
                test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=0)
                if test_acc > best_acc:
                    best_loss = test_loss
                    best_acc = test_acc
                    best_params = [lr, epoch, batch_size]

    print('Best loss (MySecondNN):', best_loss, file=f)
    print('Best acc (MySecondNN):', best_acc, file=f)
    print('Best hyperparams (MySecondNN):', best_params, file=f)
    print('\n', file=f)

    # *** MyFirstCNN ****

    best_acc = 0.0
    best_loss = 0.0
    best_params = []

    for lr in lr_list:
        for epoch in epochs_list:
            for batch_size in batch_list:
                model = MyFirstCNN.build(opt=SGD(lr=lr))
                model.fit(train_images, train_labels,
                          epochs=epoch,
                          batch_size=batch_size,
                          validation_data=(test_images, test_labels),
                          verbose=1)
                test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=0)
                if test_acc > best_acc:
                    best_loss = test_loss
                    best_acc = test_acc
                    best_params = [lr, epoch, batch_size]

    print('Best loss (MyFirstCNN):', best_loss, file=f)
    print('Best acc (MyFirstCNN):', best_acc, file=f)
    print('Best hyperparams (MyFirstCNN):', best_params, file=f)
    print('\n', file=f)

    # *** MySecondCNN ****

    best_acc = 0.0
    best_loss = 0.0
    best_params = []

    for lr in lr_list:
        for epoch in epochs_list:
            for batch_size in batch_list:
                model = MySecondCNN.build(opt=SGD(lr=lr))
                model.fit(train_images, train_labels,
                          epochs=epoch,
                          batch_size=batch_size,
                          validation_data=(test_images, test_labels),
                          verbose=1)
                test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=0)
                if test_acc > best_acc:
                    best_loss = test_loss
                    best_acc = test_acc
                    best_params = [lr, epoch, batch_size]

    print('Best loss (MySecondCNN):', best_loss, file=f)
    print('Best acc (MySecondCNN):', best_acc, file=f)
    print('Best hyperparams (MySecondCNN):', best_params, file=f)
    print('\n', file=f)

    f.close()