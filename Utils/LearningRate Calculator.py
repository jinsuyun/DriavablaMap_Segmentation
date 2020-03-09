lr = 1e-2


def scheduler(epoch):
    threshold = 15
    repeat = 2
    if epoch <= threshold:
        return lr
    else:
        diff = epoch - threshold
        return min(lr / (diff / repeat), lr)


for i in range(50):
    print(i + 1, scheduler(i))
