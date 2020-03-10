import matplotlib.pyplot as plt

lr = 2e-3


def scheduler(epoch):
    threshold = 30
    lr2 = 1e-3
    if epoch <= threshold:
        return lr
    else:
        return lr2


epoch = []
result = []
for i in range(50):
    print('{:02d}'.format(i + 1), scheduler(i))
    epoch.append(i + 1)
    result.append(scheduler(i))

plt.style.use('seaborn')
plt.plot(epoch, result)
plt.show()

# def scheduler(epoch):
#     threshold = 10
#     repeat = 5
#     if epoch <= threshold:
#         return lr
#     else:
#         diff = epoch - threshold
#         return min(lr / (diff / repeat), lr)
