import matplotlib.pyplot as plt
import numpy as np


def show_acc(file_path, title, max_epoch=200):
    epoch, acc, adv = [], [], []
    with open(file_path, 'r') as f:
        for line in f.readlines():
            line = line.strip('\n').split(',')
            epoch.append(int(line[0].split('=')[1]))
            acc.append(float(line[1].split('=')[1].replace('%', '')))
            adv.append(float(line[2].split('=')[1].replace('%', '')))
        # print(epoch, acc, adv)

    l1, = plt.plot(epoch, acc, color='r')
    l2, = plt.plot(epoch, adv, color='b')
    plt.legend(handles=[l1, l2], labels=['Clean Accuracy', 'Adv Accuracy'])
    plt.axis([0, max_epoch, 0, 100])
    plt.grid(b=None, which='major', axis='y')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')

    # 添加箭头
    # plt.annotate('mix clean(5000) + adv(2000)',
    #              ha='center', va='bottom',
    #              xytext=(60, 30),
    #              xy=(103, 68),
    #              arrowprops={'facecolor': 'black', 'shrink': 0.05, 'width': 1, 'headwidth': 5})
    # plt.annotate('mix clean(20000) + adv(2000)',
    #              ha='center', va='bottom',
    #              xytext=(150, 20),
    #              xy=(172, 75),
    #              arrowprops={'facecolor': 'black', 'shrink': 0.05, 'width': 1, 'headwidth': 5})

    plt.show()


def show_log(file_path, title):
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
    l1, = ax1.plot(iters, loss, 'b')
    ax1.set_yticks(np.arange(0, 3., 0.3))
    ax1.set_ylabel('loss')
    plt.xlabel('iter.')

    ax2 = ax1.twinx()  # this is the important function
    l2, = ax2.plot(iters, acc, 'r')
    ax2.set_yticks(np.arange(0, 100, 10))
    ax2.set_ylabel('Accuracy (%)')

    plt.legend(handles=[l1, l2], labels=['loss', 'Accuracy'])
    plt.title(title)
    plt.show()


show_acc('../train/net/swin_t/acc.txt', 'Swin Transformer Train', max_epoch=100)
show_log('../train/net/swin_t/log.txt', 'Swin Transformer Train')
