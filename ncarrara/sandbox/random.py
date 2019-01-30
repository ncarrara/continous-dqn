import numpy as np


def generate_radom_point_hyperplan(coeff, bias, min_x, max_x):
    print("-----------")
    print("-----------")
    print("-----------")
    x = np.zeros(len(coeff))
    # print("bias", bias)
    for i in range(0, len(coeff) - 1):
        # print("------------")
        min_xi = (bias - np.dot(coeff[i + 1:], np.full(len(coeff) - (i + 1), max_x))) / coeff[i]
        max_xi = (bias - np.dot(coeff[i + 1:], np.full(len(coeff) - (i + 1), min_xi))) / coeff[i]
        min_xi = np.max([min_xi,min_x])
        max_xi = np.min([max_xi,max_x])
        # print("[min_x,max_xi]", min_xi,max_xi)
        xi = min_xi + np.random.random_sample() * (max_xi - min_xi)
        bias = bias - xi * coeff[i]
        # print("bias", bias)
        x[i] = xi
        # print("xi", xi)
    x[-1] = bias / coeff[-1]
    # print(x)
    return x


def generate_random_linear_combinaison(budget, beta_max, Na, stick_to_budget=False):
    if stick_to_budget:
        raise NotImplementedError("TODO")
    p = np.random.random(Na)
    p /= np.sum(p)

    b = np.zeros(Na)

    remaining_budget = budget
    indexes = np.asarray(range(0, Na))
    np.random.shuffle(indexes)

    for i in indexes:
        bi = np.random.random() * beta_max
        Ebi = bi * p[i]
        if remaining_budget - Ebi < 0:
            bi = remaining_budget / p[i]
            Ebi = remaining_budget
        remaining_budget -= Ebi
        b[i] = bi
        if remaining_budget <= 0:
            break
    # print(remaining_budget)

    return p, b


# Na = 5
# beta_max = 1
# budget = 0.9

# for _ in range(100):
#     p, b = generate_random_linear_combinaison(budget, beta_max, Na, False)
#     print("{} <= {}".format(np.dot(p, b), budget))

import matplotlib.pyplot as plt

coeff = np.array([0.3, 0.7])
bias = 0.9
f = lambda x: ((bias - x * coeff[0]) / coeff[1])
x = np.linspace(-1000., 1000., 1000)

plt.plot(x, [f(xi) for xi in x])

max_x = 10
min_x = -10

for _ in range(10000):
    x = generate_radom_point_hyperplan(coeff, bias, min_x, max_x)
    plt.scatter(x[0], x[1])
plt.xlim([min_x- np.abs(0.1*min_x), max_x + np.abs(0.1 * max_x)])
plt.ylim([min_x- np.abs(0.1*min_x), max_x + np.abs(0.1 * max_x)])
plt.grid()
plt.show()

# p0 = 0.3
# p1 = 1 - p0
# beta_max = 1
# budget = 0.9
#
# for _ in range(1000000):
#     b0, b1 = np.random.random_sample(2)
#     b0 *= beta_max
#     b1 *= beta_max
#     if np.abs(p0 * b0 + p1 * b1 - budget) < 0.0001:
#         plt.scatter(b0, b1, c="blue")
# # plt.show()
#
# p0 = 0.3
# p1 = 1 - p0
# beta_max = 6
# budget = 0.9
#
# for _ in range(1000000):
#     b0, b1 = np.random.random_sample(2)
#     b0 *= beta_max
#     b1 *= beta_max
#     if np.abs(p0 * b0 + p1 * b1 - budget) < 0.0001:
#         plt.scatter(b0, b1, c="red")
# plt.show()
