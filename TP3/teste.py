import keras

from keras.optimizers import SGD

from networks.Networks import LeNet, MyFirstCNN, MySecondCNN, MyFirstNN, MySecondNN
from networks.DataLoader import Cifar10

if __name__ == '__main__':
    # Lendo os dados
    (train_images, train_labels), (test_images, test_labels) = Cifar10.read_data()

    # logdir = "tuning/MySecondCNN"
    # tensorboard = keras.callbacks.TensorBoard(log_dir=logdir)

    model = MySecondCNN.build(opt=SGD(lr=0.01))
    model.fit(train_images, train_labels,
                epochs=22,
                batch_size=64,
                validation_data=(test_images, test_labels),
                verbose=2)
                # callbacks=[tensorboard])
    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=0)
    
    print('loss:', test_loss)
    print('acc:', test_acc)