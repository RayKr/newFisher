import matplotlib.pyplot as plt
import numpy as np


def show_acc(file_path):
    epoch, acc, adv = [], [], []
    with open(file_path, 'r') as f:
        for line in f.readlines():
            line = line.strip('\n').split(',')
            epoch.append(int(line[0].split('=')[1]))
            acc.append(float(line[1].split('=')[1].replace('%', '')))
            adv.append(float(line[2].split('=')[1].replace('%', '')))
        # print(epoch, acc, adv)

    plt.plot(epoch, acc, color='r')
    plt.plot(epoch, adv, color='b')
    plt.axis([0, 200, 0, 100])
    plt.grid(b=None, which='major', axis='y')
    plt.title('rfgsm eps=0.15')
    plt.xlabel('epoch')
    plt.ylabel('Accuracy (%)')
    plt.show()


def show_log(file_path):
    iters, loss, acc = [], [], []
    count = 0
    with open(file_path, 'r') as f:
        for line in f.readlines():
            line = line.strip('\n').split(' ')
            count += 1
            if count % 128 == 0:
                iters.append(int(line[2]))
                loss.append(float(line[4]))
                acc.append(float(line[7].replace('%', '')))

    fig = plt.figure()
    ax1 = fig.add_subplot()
    ax1.plot(iters, loss, 'b')
    ax1.set_yticks(np.arange(0, 2.4, 0.2))
    ax1.set_ylabel('loss')
    plt.xlabel('iter.')

    ax2 = ax1.twinx()  # this is the important function
    ax2.plot(iters, acc, 'r')
    ax2.set_yticks(np.arange(0, 100, 10))
    ax2.set_ylabel('Accuracy (%)')

    plt.title('rfgsm eps=0.15')
    plt.show()


# show_acc('../train/net/pre_rfgsm/acc.txt')
show_log('../train/net/pre_rfgsm/log.txt')
