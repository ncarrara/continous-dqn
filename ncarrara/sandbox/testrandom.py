import numpy as np


def generate_random_point_hyperplan(coeff, bias, min_x, max_x):
    x = np.zeros(len(coeff))
    indexes = np.asarray(range(0, len(coeff)))
    np.random.shuffle(indexes)  # need shuffle to not favorise a dimension
    remain_indexes = np.copy(indexes)
    for i_index, index in enumerate(indexes):
        remain_indexes = remain_indexes[1:]
        current_coeff = np.take(coeff, remain_indexes)
        fullmin = np.full(len(remain_indexes), min_x)
        fullmax = np.full(len(remain_indexes), max_x)
        dotmax = np.dot(current_coeff, fullmax)
        dotmin = np.dot(current_coeff, fullmin)
        min_xi = (bias - dotmax) / coeff[index]
        max_xi = (bias - dotmin) / coeff[index]
        min_xi = np.max([min_xi, min_x])
        max_xi = np.min([max_xi, max_x])
        xi = min_xi + np.random.random_sample() * (max_xi - min_xi)
        bias = bias - xi * coeff[index]
        x[index] = xi
        if len(remain_indexes) == 1:
            break
    last_index = remain_indexes[0]
    x[last_index] = bias / coeff[last_index]
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

# coeff = np.array([0.3, 0.7])
# bias = 0.9
# f = lambda x: ((bias - x * coeff[0]) / coeff[1])
# x = np.linspace(-1000., 1000., 1000)
#
# plt.plot(x, [f(xi) for xi in x])
#
# max_x = 10
# min_x = -10
#
# for _ in range(10000):
#     x = generate_radom_point_hyperplan(coeff, bias, min_x, max_x)
#     plt.scatter(x[0], x[1])
# plt.xlim([min_x- np.abs(0.1*min_x), max_x + np.abs(0.1 * max_x)])
# plt.ylim([min_x- np.abs(0.1*min_x), max_x + np.abs(0.1 * max_x)])
# plt.grid()
# plt.show()


from mpl_toolkits.mplot3d import Axes3D

for bias in np.linspace(0., 1, 20):
    bias = 1 - bias
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    coeff = np.array([1 / 3, 1 / 3, 1 / 3])

    max_x = 1
    min_x = 0

    for _ in range(5000):
        x = generate_random_point_hyperplan(coeff, bias, min_x, max_x)
        np.testing.assert_almost_equal(np.dot(coeff, x) - bias, 0.)
        ax.scatter(x[0], x[1], x[2])

    ax.set_xlim(min_x - np.abs(min_x * 0.5), max_x + np.abs(max_x * 0.5))
    ax.set_ylim(min_x - np.abs(min_x * 0.5), max_x + np.abs(max_x * 0.5))
    ax.set_zlim(min_x - np.abs(min_x * 0.5), max_x + np.abs(max_x * 0.5))

    # for ii in []:
    ax.view_init(elev=10., azim=45)
    plt.savefig("img/{:.2f}.png".format(bias))

    plt.close()
#
# import imageio
# images = []
# for filename in ["{}.png".format(ii) for ii in range(0,360,10)]:
#     images.append(imageio.imread(filename))
# imageio.mimsave('plot.gif', images)
