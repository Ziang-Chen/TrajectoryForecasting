# -------------------------------------------------
# Driving Intend Visualization Tools
# BY : Ziang Chen
#--------------------------------------------------

from config import conf
from argoverse.visualization.mpl_point_cloud_vis import draw_point_cloud_bev
from argoverse.map_representation.map_api import ArgoverseMap
from argoverse.data_loading.argoverse_tracking_loader import ArgoverseTrackingLoader
import copy
from tqdm import tqdm

conz=conf()

tracking_dataset_dir = conz.dataset+'sample'

am = ArgoverseMap()


log_index = 0
frame_index = 100
idx = 100
argoverse_loader = ArgoverseTrackingLoader(tracking_dataset_dir)
log_id = argoverse_loader.log_list[log_index]
argoverse_data = argoverse_loader[log_index]
city_name = argoverse_data.city_name
#lidar_pts = argoverse_data.get_lidar(idx)


import matplotlib
import matplotlib.pyplot as plt
from visualize_30hz_benchmark_data_on_map import DatasetOnMapVisualizer

# Map from a bird's-eye-view (BEV)
dataset_dir = tracking_dataset_dir
experiment_prefix = 'visualization_demo'

#if you are running for the first time, or using a new set of logs, this will need to be set False to accumelate the labels again
use_existing_files = True

city_to_egovehicle_se3 = argoverse_data.get_pose(idx)

import logging
logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)

domv = DatasetOnMapVisualizer(dataset_dir, experiment_prefix, use_existing_files=use_existing_files, log_id=argoverse_data.current_log)

with tqdm(range(len(argoverse_data.lidar_timestamp_list))) as idxs:
    for idx in idxs:
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111)
        lidar_pts = argoverse_data.get_lidar(idx)
        driveable_area_pts = copy.deepcopy(lidar_pts)
        driveable_area_pts = city_to_egovehicle_se3.transform_point_cloud(
            driveable_area_pts
        )  # put into city coords
        driveable_area_pts = am.remove_non_driveable_area_points(driveable_area_pts, city_name)
        driveable_area_pts = city_to_egovehicle_se3.inverse_transform_point_cloud(
            driveable_area_pts
        )

        xcenter,ycenter,_ = argoverse_data.get_pose(idx).translation
    
        xmin = xcenter - 150  # 150
        xmax = xcenter + 150  # 150
        ymin = ycenter - 150  # 150
        ymax = ycenter + 150  # 150
        ax.scatter(xcenter, ycenter, 200, color="g", marker=".", zorder=2)
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])
        local_lane_polygons = am.find_local_lane_polygons([xmin, xmax, ymin, ymax], city_name)
        local_das = am.find_local_driveable_areas([xmin, xmax, ymin, ymax], city_name)


        domv.render_bev_labels_mpl(
            city_name,
            ax,
            "city_axis",
            None,
            copy.deepcopy(local_lane_polygons),
            copy.deepcopy(local_das),
            log_id,
            argoverse_data.lidar_timestamp_list[idx],
            city_to_egovehicle_se3,
            am,
        )
    
        fig.suptitle(conz.title)
        plt.savefig(conz.result_base_dir+"intend_viz/"+"{:}.jpg".format(idx))