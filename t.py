lr = 1e-2


def scheduler(epoch, threshold=5):
    if epoch <= threshold:
        return lr
    else:
        return lr / (epoch - threshold)


for i in range(20):
    print(i + 1, scheduler(i))
