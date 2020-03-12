import matplotlib.pyplot as plt

lr = 3e-4


def scheduler(epoch):
    warmup = 5
    warmup_lr = 1e-5
    threshold = 30
    lr2 = 1e-4
    if epoch < warmup:
        return warmup_lr
    elif epoch < threshold:
        return lr
    else:
        return lr2


epoch = []
result = []
for i in range(60):
    print('{:02d}'.format(i + 1), scheduler(i))
    epoch.append(i + 1)
    result.append(scheduler(i))

plt.style.use('seaborn')
plt.plot(epoch, result)
plt.show()

# def scheduler(epoch):
#     threshold = 10
#     repeat = 5
#     min_lr = 5e-4
#     if epoch <= threshold:
#         return lr
#     else:
#         diff = epoch - threshold
#         new_lr = max(min(lr / (diff / repeat), lr), min_lr)
#         return new_lr
