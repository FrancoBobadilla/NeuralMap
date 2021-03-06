"""
Utility functions for making SOM-specific highly customizable plots.

Functions
--------------
.. autosummary::
   :toctree: generated/
   update
   tiles
   bubbles
   legend_or_bar

"""
from numpy import ones, nanmax, nanmin, arange, pi, ndarray, isnan, unravel_index, sqrt

from matplotlib import colorbar, colors, pyplot as plt
from matplotlib.patches import Polygon, RegularPolygon, Circle, Wedge, Patch
from mpl_toolkits.axes_grid1 import make_axes_locatable


def update(positions, hexagonal, data, dimensions=10, bmu=None, relative_positions=None,
           displacement=None, color_map=plt.cm.get_cmap('RdYlGn_r')):
    """
    Plot the update value of each node and their relative position displacement.

    Parameters
    ----------
    positions: ndarray
        Cartesian coordinates of the nodes in the 2D virtual space.
        Dimensions should be (x, y, 2).
    hexagonal: bool
        Whether the nodes have an hexagonal shape or squared.
    data: array_like
        Update value of each node.
        Dimensions should be (x, y).
    dimensions: array_like
        Horizontal and vertical measure of the map in the virtual space.
        Dimensions should be (2).
    bmu: tuple or array_like (optional, default None)
        Cartesian coordinate of the best matching unit in the 2D virtual space.
        Dimensions should be (2).
    relative_positions: ndarray (optional, default None)
        Cartesian coordinates of the relative positions of the nodes.
        Dimensions should be (x, y, 2).
    displacement: array_like (optional, default None)
        Displacement vector of each node over the relative positions space.
        Dimensions should be (x, y, 2).
    color_map: colormap (optional, default plt.cm.get_cmap('RdYlGn_r'))
        Color map to paint the nodes.

    """
    data_c = data.copy()
    data_max = nanmax(data_c)
    data_min = nanmin(data_c)

    if data_max == data_min:
        data_max = 1.
        data_min = 0.

    data_c -= data_min
    data_c /= data_max - data_min
    x_range = arange(positions.shape[0])
    y_range = arange(positions.shape[1])
    figure = plt.figure(figsize=(dimensions, dimensions))
    axes = figure.add_subplot(111)
    axes.set_aspect('equal')

    if hexagonal:
        num_vertices = 6
        radius = (3 ** 0.5) / 3
        orientation = 0
        axes.set_xticks(x_range)
        axes.set_yticks(y_range * (3 ** 0.5) / 2)

    else:
        num_vertices = 4
        radius = (2 ** 0.5) / 2
        orientation = pi / 4
        axes.set_xticks(x_range)
        axes.set_yticks(y_range)

    axes.set_xticklabels(x_range)
    axes.set_yticklabels(y_range)

    for i in range(data_c.shape[0]):
        for j in range(data_c.shape[1]):
            axes.add_patch(
                RegularPolygon(positions[i, j], numVertices=num_vertices, radius=radius,
                               orientation=orientation,
                               facecolor=color_map(data_c[i, j]),
                               edgecolor=color_map(data_c[i, j])))

    if bmu is not None:
        axes.add_patch(Circle(bmu, radius=radius / 3, facecolor='white', edgecolor='white'))
    if relative_positions is not None and displacement is not None:
        axes.quiver(relative_positions[..., 0], relative_positions[..., 1], displacement[..., 0],
                    displacement[..., 1], angles='xy', scale_units='xy', scale=1, zorder=10)
    legend_or_bar([positions[..., 0].max(), positions[..., 1].max()],
                  [positions[..., 0].min(), positions[..., 1].min()],
                  data_min, data_max, color_map, figure, axes)


def tiles(positions, hexagonal, data, color_map=plt.cm.get_cmap('RdYlGn_r'), size=10, text=None,
          borders=False, norm=True, labels=None, intensity=None, title=None):
    """
    Plot the map with nodes having colors according to their values (or set of values).
    Nodes have an hexagonal or squared shape that completely fill the space.

    Parameters
    ----------
    positions: ndarray
        Cartesian coordinates of the nodes in the 2D virtual space.
        Dimensions should be (x, y, 2).
    hexagonal: bool
        Whether the nodes have an hexagonal shape or squared.
    data: array_like
        Value of each node.
        Dimensions should be (x, y).
    color_map: colormap (optional, default plt.cm.get_cmap('RdYlGn_r'))
        Color map to paint the nodes.
        Ignored if entered a ``labels`` value.
    size: int (optional, default 10)
        Horizontal and vertical size of the final plot.
    text: array_like (optional, default None)
        Text to display for each node.
        Dimensions should be (x, y).
    borders: bool (optional, default False)
        Draw nodes borders.
    norm: bool (optional, default True)
        Normalize input ``data`` to the range [0, 1].
    labels: array_like (optional, default None)
        A 1D array containing each node label (could be their class or cluster) as an integer value.
        Length should be (x * y).
    intensity: array_like (optional, default None)
        Alpha value of each node. This value affects nodes colors. A lower value makes them lighter.
        Dimensions should be (x, y).
        Values should be in the range [0, 1].
    title: string (optional, default None)
        Plot some title.

    """
    if intensity is None:
        intensity = ones((positions.shape[0], positions.shape[1]))

    if labels is not None:
        color_map = plt.cm.get_cmap('hsv', len(labels) + 1)

    if text is None:
        text = [[''] * positions.shape[1]] * positions.shape[0]

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

    x_range = arange(positions.shape[0])
    y_range = arange(positions.shape[1])
    figure = plt.figure(figsize=(size, size))
    axes = figure.add_subplot(111)
    axes.set_aspect('equal')

    if title is not None:
        axes.title.set_text(title)

    if hexagonal:
        num_vertices = 6
        radius = (3 ** 0.5) / 3
        orientation = 0
        axes.set_xticks(x_range)
        axes.set_yticks(y_range * (3 ** 0.5) / 2)

    else:
        num_vertices = 4
        radius = (2 ** 0.5) / 2
        orientation = pi / 4
        axes.set_xticks(x_range)
        axes.set_yticks(y_range)

    half_radius = radius / 2
    axes.set_xticklabels(x_range)
    axes.set_yticklabels(y_range)

    for i in range(data_c.shape[0]):
        for j in range(data_c.shape[1]):
            if isinstance(data_c[i, j], ndarray):
                x_position = positions[(i, j, 0)]
                y_position = positions[(i, j, 1)]

                if hexagonal:
                    if not isnan(data_c[i, j, 0]):
                        axes.add_patch(Polygon(
                            ([(x_position + .25, y_position - half_radius / 2),
                              (x_position + .25, y_position + half_radius / 2),
                              (x_position + .5, y_position + half_radius),
                              (x_position + .5, y_position - half_radius)]),
                            facecolor=color_map(data_c[i, j, 0]),
                            edgecolor=color_map(data_c[i, j, 0]),
                            alpha=intensity[i, j]))

                    if not isnan(data_c[i, j, 1]):
                        axes.add_patch(
                            Polygon(([(x_position + .25, y_position + half_radius / 2),
                                      (x_position, y_position + half_radius),
                                      (x_position, y_position + 2 * half_radius),
                                      (x_position + .5, y_position + half_radius)]),
                                    facecolor=color_map(data_c[i, j, 1]),
                                    edgecolor=color_map(data_c[i, j, 1]),
                                    alpha=intensity[i, j]))

                    if not isnan(data_c[i, j, 2]):
                        axes.add_patch(
                            Polygon(([(x_position, y_position + half_radius),
                                      (x_position - .25, y_position + half_radius / 2),
                                      (x_position - .5, y_position + half_radius),
                                      (x_position, y_position + 2 * half_radius)]),
                                    facecolor=color_map(data_c[i, j, 2]),
                                    edgecolor=color_map(data_c[i, j, 2]),
                                    alpha=intensity[i, j]))

                    if not isnan(data_c[i, j, 3]):
                        axes.add_patch(Polygon(
                            ([(x_position - .25, y_position + half_radius / 2),
                              (x_position - .25, y_position - half_radius / 2),
                              (x_position - .5, y_position - half_radius),
                              (x_position - .5, y_position + half_radius)]),
                            facecolor=color_map(data_c[i, j, 3]),
                            edgecolor=color_map(data_c[i, j, 3]),
                            alpha=intensity[i, j]))

                    if not isnan(data_c[i, j, 4]):
                        axes.add_patch(
                            Polygon(([(x_position - .25, y_position - half_radius / 2),
                                      (x_position, y_position - half_radius),
                                      (x_position, y_position - 2 * half_radius),
                                      (x_position - .5, y_position - half_radius)]),
                                    facecolor=color_map(data_c[i, j, 4]),
                                    edgecolor=color_map(data_c[i, j, 4]),
                                    alpha=intensity[i, j]))

                    if not isnan(data_c[i, j, 5]):
                        axes.add_patch(
                            Polygon(([(x_position, y_position - half_radius),
                                      (x_position + .25, y_position - half_radius / 2),
                                      (x_position + .5, y_position - half_radius),
                                      (x_position, y_position - 2 * half_radius)]),
                                    facecolor=color_map(data_c[i, j, 5]),
                                    edgecolor=color_map(data_c[i, j, 5]),
                                    alpha=intensity[i, j]))

                else:
                    if not isnan(data_c[i, j, 0]):
                        axes.add_patch(Polygon(
                            ([(x_position + .25, y_position - .25),
                              (x_position + .5, y_position - .5),
                              (x_position + .5, y_position + .5),
                              (x_position + .25, y_position + .25)]),
                            facecolor=color_map(data_c[i, j, 0]),
                            edgecolor=color_map(data_c[i, j, 0]),
                            alpha=intensity[i, j]))

                    if not isnan(data_c[i, j, 1]):
                        axes.add_patch(Polygon(
                            ([(x_position + .5, y_position + .5),
                              (x_position + .25, y_position + .25),
                              (x_position - .25, y_position + .25),
                              (x_position - .5, y_position + .5)]),
                            facecolor=color_map(data_c[i, j, 1]),
                            edgecolor=color_map(data_c[i, j, 1]),
                            alpha=intensity[i, j]))

                    if not isnan(data_c[i, j, 2]):
                        axes.add_patch(Polygon(
                            ([(x_position - .25, y_position + .25),
                              (x_position - .5, y_position + .5),
                              (x_position - .5, y_position - .5),
                              (x_position - .25, y_position - .25)]),
                            facecolor=color_map(data_c[i, j, 2]),
                            edgecolor=color_map(data_c[i, j, 2]),
                            alpha=intensity[i, j]))

                    if not isnan(data_c[i, j, 3]):
                        axes.add_patch(Polygon(
                            ([(x_position - .5, y_position - .5),
                              (x_position - .25, y_position - .25),
                              (x_position + .25, y_position - .25),
                              (x_position + .5, y_position - .5)]),
                            facecolor=color_map(data_c[i, j, 3]),
                            edgecolor=color_map(data_c[i, j, 3]),
                            alpha=intensity[i, j]))

                if not isnan(data_c[i, j, num_vertices]):
                    axes.add_patch(
                        RegularPolygon((x_position, y_position), numVertices=num_vertices,
                                       radius=half_radius,
                                       orientation=orientation,
                                       facecolor=color_map(data_c[i, j, num_vertices]),
                                       edgecolor=color_map(data_c[i, j, num_vertices]),
                                       alpha=intensity[i, j]))

                if borders:
                    axes.add_patch(
                        RegularPolygon((x_position, y_position), numVertices=num_vertices,
                                       radius=radius, orientation=orientation,
                                       fill=0, edgecolor='black', alpha=intensity[i, j]))

                axes.text(*positions[i, j], text[i][j], ha="center", va="center")

            else:
                if not isnan(data_c[i, j]):
                    if borders:
                        edge_color = 'black'

                    else:
                        edge_color = color_map(data_c[i, j])

                    axes.add_patch(
                        RegularPolygon(positions[i, j], numVertices=num_vertices, radius=radius,
                                       orientation=orientation, facecolor=color_map(data_c[i, j]),
                                       edgecolor=edge_color, alpha=intensity[i, j]))

                    axes.text(*positions[i, j], text[i][j], ha="center", va="center")

    legend_or_bar([positions[..., 0].max(), positions[..., 1].max()],
                  [positions[..., 0].min(), positions[..., 1].min()],
                  data_min, data_max, color_map, figure, axes, labels=labels)


def bubbles(diameters, positions, data, color_map=None, size=10, text=None,
            borders=False, norm=True, labels=None, intensity=None, title=None, connections=None,
            reverse=None, display_empty_nodes=True):
    """
    Plot the map with nodes as circles with a position (center) and a diameter
    with a color according to the input data.

    Parameters
    ----------
    diameters: array_like
        Diameters of each node.
        Dimensions should be (x, y).
        Values should be greater than 0.
    positions: ndarray
        Cartesian coordinates of the nodes in the 2D virtual space.
        Dimensions should be (x, y, 2).
    data: array_like
        Value of each node.
        Dimensions should be (x, y).
    color_map: colormap (optional, default None)
        Color map to paint the nodes.
    size: int (optional, default 10)
        Horizontal and vertical size of the final plot.
    text: array_like (optional, default None)
        Text to display for each node.
        Dimensions should be (x, y).
    borders: bool (optional, default False)
        Draw nodes borders.
    norm: bool (optional, default True)
        Normalize the input data to the range [0, 1].
    labels: array_like (optional, default None)
        A 1D array containing each node label (could be their class or cluster) as a numeric value.
        Length should be (x * y).
    intensity: array_like (optional, default None)
        Alpha value of each node. This value affects nodes colors. A low value makes them lighter.
        Dimensions should be (x, y).
        Values should be in the range [0, 1].
    title: string (optional, default None)
        Plot some title.
    connections: ndarray (optional, default None)
        Matrix indicating intensity of connections between nodes.
        Use nan to represent that there is no connection between two nodes.
        Dimensions should be (x * y, x * y).
    reverse: array_like (optional, default None)
        Matrix of bool values indicating for each connection whether it should be drawn straight
        or be drawn "going around" the toroidal space.
        Dimensions should be (x * y, x * y).
    display_empty_nodes: bool (optional, default True)
        Draw nodes with a ``diameter`` value equals to 0 as grey diamonds.

    """
    if intensity is None:
        intensity = ones((positions.shape[0], positions.shape[1]))

    if color_map is None:
        if labels is not None:
            color_map = plt.cm.get_cmap('hsv', len(labels) + 1)

        else:
            color_map = plt.cm.get_cmap('RdYlGn_r')

    if text is None:
        text = [[''] * positions.shape[1]] * positions.shape[0]

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

    figure = plt.figure(figsize=(size, size))
    axes = figure.add_subplot(111)
    axes.set_aspect('equal')

    if title is not None:
        axes.title.set_text(title)

    if connections is not None:
        if reverse is None:
            reverse = connections.copy() * 0

        cm_min = nanmin(connections)
        width = positions[..., 0].max()
        height = positions[..., 1].max()

        for i in range(connections.shape[0]):
            for j in range(connections.shape[1]):
                if not isnan(connections[i, j]):
                    first_pos = positions[
                        unravel_index(i, (positions.shape[0], positions.shape[1]))].copy()
                    second_pos = positions[
                        unravel_index(j, (positions.shape[0], positions.shape[1]))].copy()

                    if reverse[i, j]:
                        edges_connection = False

                        if first_pos[0] - second_pos[0] > width / 2:
                            edges_connection = True
                            second_pos[0] += 1 + width

                        if second_pos[0] - first_pos[0] > width / 2:
                            edges_connection = True
                            second_pos[0] -= 1 + width

                        if first_pos[1] - second_pos[1] > height / 2:
                            edges_connection = True
                            second_pos[1] += 1 + height

                        if second_pos[1] - first_pos[1] > height / 2:
                            edges_connection = True
                            second_pos[1] -= 1 + height

                        if edges_connection:
                            second_pos = (second_pos + first_pos) / 2

                    plt.plot([first_pos[0], second_pos[0]], [first_pos[1], second_pos[1]],
                             zorder=-d_max * 2, color='black',
                             alpha=(0.2 + cm_min / connections[i, j]) / 1.2)

    for i in range(data_c.shape[0]):
        for j in range(data_c.shape[1]):
            if diameters[i, j] > 0:
                if isinstance(data_c[i, j], ndarray):
                    start = 0

                    for k in range(data_c[i, j].shape[0]):
                        end = start + data_c[i, j, k] * 360 / diameters[i, j]

                        if data_c[i, j, k]:
                            radius = 0.8 * sqrt(diameters[i, j] / d_max) / 3
                            axes.add_patch(Wedge(positions[i, j], r=radius, theta1=start,
                                                 theta2=end,
                                                 facecolor=color_map(k / data_c[i, j].shape[0]),
                                                 edgecolor=color_map(k / data_c[i, j].shape[0]),
                                                 zorder=-diameters[i, j],
                                                 alpha=intensity[i, j]))

                        start = end

                    if borders:
                        radius = 0.8 * sqrt(diameters[i, j] / d_max) / 3
                        axes.add_patch(Circle(positions[i, j], radius=radius, facecolor='None',
                                       edgecolor='black', zorder=-diameters[i, j],
                                       alpha=intensity[i, j]))

                    axes.text(*positions[i, j], text[i][j], ha="center", va="center")

                else:
                    if borders:
                        edge_color = 'black'

                    else:
                        edge_color = color_map(data_c[i, j])

                    radius = 0.8 * sqrt(diameters[i, j] / d_max) / 3
                    axes.add_patch(Circle(positions[i, j], radius=radius,
                                          facecolor=color_map(data_c[i, j]), edgecolor=edge_color,
                                          zorder=-diameters[i, j], alpha=intensity[i, j]))
                    axes.text(*positions[i, j], text[i][j], ha="center", va="center")

            else:
                if display_empty_nodes:
                    radius = 0.8 * sqrt(1 / d_max) / 3
                    axes.add_patch(RegularPolygon(positions[i, j], numVertices=4, radius=radius,
                                   facecolor='lightgrey', edgecolor='lightgrey', zorder=-d_max,
                                   alpha=intensity[i, j]))
                    axes.text(*positions[i, j], text[i][j], ha="center", va="center")

    legend_or_bar([positions[..., 0].max(), positions[..., 1].max()],
                  [positions[..., 0].min(), positions[..., 1].min()],
                  data_min, data_max, color_map, figure, axes, labels=labels)


def legend_or_bar(position_max, position_min, data_min, data_max, color_map, figure, axes,
                  labels=None):
    """
    Plot the color bar or legend if labels are provided at the right of the main plot.

    Parameters
    ----------
    position_max: list
        Cartesian coordinates of the upper right corner.
        Should be a list with two elements.
    position_min: list
        Cartesian coordinates of the bottom left corner.
        Should be a list with two elements.
    data_min: float
        Minimum value of the nodes.
        Ignored if ``labels`` are provided.
    data_max: float
        Maximum value of the nodes.
        Ignored if ``labels`` are provided.
    color_map: colormap
        Color map to paint the nodes.
    figure: figure
        Matplotlib figure.
    axes: axes
        Matplotlib axes.
    labels: array_like (optional, default None)
        A 1D array containing labels without repeating, as a numeric value.

    """
    plt.plot(position_max[0], position_max[1], ' ', alpha=0)
    plt.plot(position_min[0], position_min[1], ' ', alpha=0)
    ax_cb = make_axes_locatable(axes).new_horizontal(size="5%", pad=0.25)

    if labels is None:
        colorbar.ColorbarBase(ax_cb, cmap=color_map, orientation='vertical',
                              norm=colors.Normalize(vmin=data_min, vmax=data_max))
        figure.add_axes(ax_cb)

    else:
        axes.legend(bbox_to_anchor=(1, 1),
                    handles=[Patch(color=color_map(k), label=label) for k, label in
                             enumerate(labels)])
