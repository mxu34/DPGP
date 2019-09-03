

from argoverse.map_representation.map_api import ArgoverseMap
from argoverse.utils.mpl_plotting_utils import draw_lane_polygons
import matplotlib.pyplot as plt
import numpy as np


def map_Selection(map_range, seq_lane_props):
    x_min = map_range[0]
    x_max = map_range[1]
    y_min = map_range[2]
    y_max = map_range[3]
    # print(x_min, x_max, y_min, y_max)
    lane_centerlines = []
    # Get lane centerlines which lie within the range of trajectories
    for lane_id, lane_props in seq_lane_props.items():
        lane_cl = lane_props.centerline
        if (
                np.min(lane_cl[:, 0]) < x_max
                and np.min(lane_cl[:, 1]) < y_max
                and np.max(lane_cl[:, 0]) > x_min
                and np.max(lane_cl[:, 1]) > y_min
            ):
            lane_centerlines.append(lane_cl)
    return lane_centerlines


def draw_local_map(avm, city_name, x_min, x_max,y_min, y_max, ax):
    seq_lane_bbox = avm.city_halluc_bbox_table[city_name]
    seq_lane_props = avm.city_lane_centerlines_dict[city_name]

    # fig = plt.figure(figsize=(15,15))
    # ax = fig.add_subplot(111)
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    map_range = np.array([x_min, x_max, y_min, y_max])
    seq_lane_props = avm.city_lane_centerlines_dict[city_name]
    lane_centerlines = map_Selection(map_range, seq_lane_props)
    # print('PIT map visualization with highest traffic data density')
    for lane_cl in lane_centerlines:
        plt.plot(lane_cl[:, 0], lane_cl[:, 1], "--", color="grey", alpha=1, linewidth=1, zorder=0)

    # use built in function to plot road info
    local_lane_polygons = avm.find_local_lane_polygons([x_min, x_max, y_min, y_max], city_name)
    draw_lane_polygons(ax, local_lane_polygons, color='#02383C')