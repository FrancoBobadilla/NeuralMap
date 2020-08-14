from matplotlib.patches import Polygon, RegularPolygon, Circle, Wedge, Patch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm, colorbar, colors, pyplot as plt
from numpy import ones, nanmax, nanmin, arange, sin, pi, ndarray, isnan, unravel_index, sqrt


def tiles(cart_coord, hexagonal, data, cmap=cm.RdYlGn_r, size=10, borders=False, norm=True, labels=None,
               intensity=None, title=None):
    if intensity is None:
        intensity = ones((cart_coord.shape[0], cart_coord.shape[1]))

    data_c = data.copy()
    data_max = nanmax(data_c)
    data_min = nanmin(data_c)

    if data_max == data_min:
        data_max = 1.
        data_min = 0.

    if norm:
        data_c = data_c.astype(float)
        data_c -= data_min
        data_c /= data_max - data_min

    xrange = arange(cart_coord.shape[0])
    yrange = arange(cart_coord.shape[1])

    f = plt.figure(figsize=(size, size))
    ax = f.add_subplot(111)
    ax.set_aspect('equal')

    ax_cb = make_axes_locatable(ax).new_horizontal(size="5%", pad=0.25)

    if title is not None:
        ax.title.set_text(title)

    if hexagonal:
        numVertices = 6
        radius = sin(pi / 3) * 2 / 3
        orientation = 0
        ax.set_xticks(xrange + 0.25)
        ax.set_yticks(yrange * sin(pi / 3))

    else:
        numVertices = 4
        radius = sin(pi / 4)
        orientation = pi / 4
        ax.set_xticks(xrange)
        ax.set_yticks(yrange)

    a = radius / 2

    ax.set_xticklabels(xrange)
    ax.set_yticklabels(yrange)

    for i in range(data_c.shape[0]):
        for j in range(data_c.shape[1]):
            if type(data_c[i, j]) is ndarray:
                wx = cart_coord[(i, j, 0)]
                wy = cart_coord[(i, j, 1)]
                if hexagonal:
                    if not isnan(data_c[i, j, 0]):
                        ax.add_patch(Polygon(
                            ([(wx + .25, wy - a / 2), (wx + .25, wy + a / 2), (wx + .5, wy + a), (wx + .5, wy - a)]),
                            facecolor=cmap(data_c[i, j, 0]), edgecolor=cmap(data_c[i, j, 0]), alpha=intensity[i, j]))
                    if not isnan(data_c[i, j, 1]):
                        ax.add_patch(
                            Polygon(([(wx + .25, wy + a / 2), (wx, wy + a), (wx, wy + 2 * a), (wx + .5, wy + a)]),
                                    facecolor=cmap(data_c[i, j, 1]), edgecolor=cmap(data_c[i, j, 1]),
                                    alpha=intensity[i, j]))
                    if not isnan(data_c[i, j, 2]):
                        ax.add_patch(
                            Polygon(([(wx, wy + a), (wx - .25, wy + a / 2), (wx - .5, wy + a), (wx, wy + 2 * a)]),
                                    facecolor=cmap(data_c[i, j, 2]), edgecolor=cmap(data_c[i, j, 2]),
                                    alpha=intensity[i, j]))
                    if not isnan(data_c[i, j, 3]):
                        ax.add_patch(Polygon(
                            ([(wx - .25, wy + a / 2), (wx - .25, wy - a / 2), (wx - .5, wy - a), (wx - .5, wy + a)]),
                            facecolor=cmap(data_c[i, j, 3]), edgecolor=cmap(data_c[i, j, 3]), alpha=intensity[i, j]))
                    if not isnan(data_c[i, j, 4]):
                        ax.add_patch(
                            Polygon(([(wx - .25, wy - a / 2), (wx, wy - a), (wx, wy - 2 * a), (wx - .5, wy - a)]),
                                    facecolor=cmap(data_c[i, j, 4]), edgecolor=cmap(data_c[i, j, 4]),
                                    alpha=intensity[i, j]))
                    if not isnan(data_c[i, j, 5]):
                        ax.add_patch(
                            Polygon(([(wx, wy - a), (wx + .25, wy - a / 2), (wx + .5, wy - a), (wx, wy - 2 * a)]),
                                    facecolor=cmap(data_c[i, j, 5]), edgecolor=cmap(data_c[i, j, 5]),
                                    alpha=intensity[i, j]))
                else:
                    if not isnan(data_c[i, j, 0]):
                        ax.add_patch(Polygon(
                            ([(wx + .25, wy - .25), (wx + .5, wy - .5), (wx + .5, wy + .5), (wx + .25, wy + .25)]),
                            facecolor=cmap(data_c[i, j, 0]), edgecolor=cmap(data_c[i, j, 0]), alpha=intensity[i, j]))
                    if not isnan(data_c[i, j, 1]):
                        ax.add_patch(Polygon(
                            ([(wx + .5, wy + .5), (wx + .25, wy + .25), (wx - .25, wy + .25), (wx - .5, wy + .5)]),
                            facecolor=cmap(data_c[i, j, 1]), edgecolor=cmap(data_c[i, j, 1]), alpha=intensity[i, j]))
                    if not isnan(data_c[i, j, 2]):
                        ax.add_patch(Polygon(
                            ([(wx - .25, wy + .25), (wx - .5, wy + .5), (wx - .5, wy - .5), (wx - .25, wy - .25)]),
                            facecolor=cmap(data_c[i, j, 2]), edgecolor=cmap(data_c[i, j, 2]), alpha=intensity[i, j]))
                    if not isnan(data_c[i, j, 3]):
                        ax.add_patch(Polygon(
                            ([(wx - .5, wy - .5), (wx - .25, wy - .25), (wx + .25, wy - .25), (wx + .5, wy - .5)]),
                            facecolor=cmap(data_c[i, j, 3]), edgecolor=cmap(data_c[i, j, 3]), alpha=intensity[i, j]))

                if not isnan(data_c[i, j, numVertices]):
                    ax.add_patch(RegularPolygon((wx, wy), numVertices=numVertices, radius=a, orientation=orientation,
                                                facecolor=cmap(data_c[i, j, numVertices]),
                                                edgecolor=cmap(data_c[i, j, numVertices]), alpha=intensity[i, j]))
                if borders:
                    ax.add_patch(
                        RegularPolygon((wx, wy), numVertices=numVertices, radius=radius, orientation=orientation,
                                       fill=0, edgecolor='black', alpha=intensity[i, j]))

            else:
                if not isnan(data_c[i, j]):
                    if borders:
                        edgecolor = 'black'
                    else:
                        edgecolor = cmap(data_c[i, j])
                    ax.add_patch(RegularPolygon(cart_coord[i, j], numVertices=numVertices, radius=radius,
                                                orientation=orientation, facecolor=cmap(data_c[i, j]),
                                                edgecolor=edgecolor, alpha=intensity[i, j]))

    ax.plot(cart_coord[..., 0].max(), cart_coord[..., 1].max(), ' ', alpha=0)
    ax.plot(cart_coord[..., 0].min(), cart_coord[..., 1].min(), ' ', alpha=0)

    if labels is None:
        colorbar.ColorbarBase(ax_cb, cmap=cmap, orientation='vertical',
                              norm=colors.Normalize(vmin=data_min, vmax=data_max))
        f.add_axes(ax_cb)
    else:
        ax.legend(bbox_to_anchor=(1, 1), handles=[Patch(color=cmap(k), label=label) for k, label in enumerate(labels)])

    plt.show()


def bubbles(diameters, cart_coord, data, connection_matrix=None, reverse_matrix=None, norm=True, labels=None,
                 intensity=None, title='plot', cmap=cm.RdYlGn_r, size=10, borders=False, show_empty_nodes=True):
    if intensity is None:
        intensity = ones((cart_coord.shape[0], cart_coord.shape[1]))

    d_max = diameters.max()

    data_c = data.copy()
    data_max = nanmax(data_c)
    data_min = nanmin(data_c)
    if data_max == data_min:
        data_max = 1.
        data_min = 0.

    if norm:
        data_c = data_c.astype(float)
        data_c = data_c.astype(float)
        data_c -= data_min
        data_c /= data_max - data_min

    f = plt.figure(figsize=(size, size))
    ax = f.add_subplot(111)
    ax.set_aspect('equal')

    ax_cb = make_axes_locatable(ax).new_horizontal(size="5%", pad=0.25)

    ax.title.set_text(title)

    if connection_matrix is not None:
        if reverse_matrix is None:
            reverse_matrix = connection_matrix.copy() * 0
        cm_min = nanmin(connection_matrix)
        width = cart_coord[..., 0].max()
        height = cart_coord[..., 1].max()
        for i in range(connection_matrix.shape[0]):
            for j in range(connection_matrix.shape[1]):
                if not isnan(connection_matrix[i, j]):
                    first_pos = cart_coord[unravel_index(i, (cart_coord.shape[0], cart_coord.shape[1]))].copy()
                    second_pos = cart_coord[unravel_index(j, (cart_coord.shape[0], cart_coord.shape[1]))].copy()
                    if reverse_matrix[i, j]:
                        reversed = False
                        if first_pos[0] - second_pos[0] > width / 2:
                            reversed = True
                            second_pos[0] += 1 + width
                        if second_pos[0] - first_pos[0] > width / 2:
                            reversed = True
                            second_pos[0] -= 1 + width
                        if first_pos[1] - second_pos[1] > height / 2:
                            reversed = True
                            second_pos[1] += 1 + height
                        if second_pos[1] - first_pos[1] > height / 2:
                            reversed = True
                            second_pos[1] -= 1 + height
                        if reversed:
                            second_pos = (second_pos + first_pos) / 2

                    plt.plot([first_pos[0], second_pos[0]], [first_pos[1], second_pos[1]], zorder=-d_max * 2,
                             color='black', alpha=cm_min / connection_matrix[i, j])

    for i in range(data_c.shape[0]):
        for j in range(data_c.shape[1]):
            if diameters[i, j] > 0:
                if type(data_c[i, j]) is ndarray:
                    init = 0
                    for k in range(data_c[i, j].shape[0]):
                        next = init + data_c[i, j, k] * 360 / diameters[i, j]
                        if data_c[i, j, k]:
                            ax.add_patch(Wedge(cart_coord[i, j], r=sqrt(diameters[i, j] / d_max) / 3, theta1=init,
                                               theta2=next, facecolor=cmap(k / data_c[i, j].shape[0]),
                                               edgecolor=cmap(k / data_c[i, j].shape[0]), zorder=-diameters[i, j],
                                               alpha=intensity[i, j]))
                        init = next
                    if borders:
                        ax.add_patch(
                            Circle(cart_coord[i, j], radius=sqrt(diameters[i, j] / d_max) / 3, facecolor='None',
                                   edgecolor='black', zorder=-diameters[i, j], alpha=intensity[i, j]))
                else:
                    if borders:
                        edgecolor = 'black'
                    else:
                        edgecolor = cmap(data_c[i, j])
                    ax.add_patch(Circle(cart_coord[i, j], radius=sqrt(diameters[i, j] / d_max) / 3,
                                        facecolor=cmap(data_c[i, j]), edgecolor=edgecolor, zorder=-diameters[i, j],
                                        alpha=intensity[i, j]))
            else:
                if show_empty_nodes:
                    ax.add_patch(RegularPolygon(cart_coord[i, j], numVertices=4, radius=sqrt(1 / d_max) / 3,
                                                facecolor='lightgrey', edgecolor='lightgrey', zorder=-d_max,
                                                alpha=intensity[i, j]))

    plt.plot(cart_coord[..., 0].max(), cart_coord[..., 1].max(), ' ', alpha=0)
    plt.plot(cart_coord[..., 0].min(), cart_coord[..., 1].min(), ' ', alpha=0)

    if labels is None:
        colorbar.ColorbarBase(ax_cb, cmap=cmap, orientation='vertical',
                              norm=colors.Normalize(vmin=data_min, vmax=data_max))
        f.add_axes(ax_cb)
    else:
        ax.legend(bbox_to_anchor=(1, 1), handles=[Patch(color=cmap(k), label=label) for k, label in enumerate(labels)])

    plt.show()
