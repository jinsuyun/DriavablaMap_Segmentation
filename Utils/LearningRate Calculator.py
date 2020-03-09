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
    print('{:02d}'.format(i + 1), scheduler(i))
