lr = 1e-2


def scheduler(epoch, threshold=10):
    if epoch <= threshold:
        return lr
    else:
        return lr / (epoch - threshold)


for i in range(50):
    print(i + 1, scheduler(i))
