from reader import Reader
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
mpl.use('Agg')


def get_corners(dimensions, location, rotation_y):
    dimensions = np.clip(dimensions, a_min=1.5, a_max=5)
    R = np.array([[+np.cos(rotation_y), 0, +np.sin(rotation_y)],
                  [0, 1, 0],
                  [-np.sin(rotation_y), 0, +np.cos(rotation_y)]],
                 dtype=np.float32)
    h, w, l = dimensions
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
    corners_3D = np.dot(R, [x_corners, y_corners, z_corners])
    corners_3D += location.reshape((3, 1))
    return corners_3D


def draw_projection(corners, P2, ax, color):
    projection = np.dot(P2, np.vstack([corners, np.ones(8, dtype=np.int32)]))
    projection = (projection / projection[2])[:2]
    orders = [[0, 1, 2, 3, 0],
              [4, 5, 6, 7, 4],
              [2, 6], [3, 7],
              [1, 5], [0, 4]]
    for order in orders:
        ax.plot(projection[0, order], projection[1, order],
                color=color, linewidth=2)
    return


def draw_space(corners, ax, color):
    assert corners.shape == (3, 8)
    orders = [0, 1, 2, 3, 0, 4, 5, 6, 7, 4, 5, 1, 2, 6, 7, 3]
    lines = np.zeros((3, 16), dtype=np.float32)
    for index, point in enumerate(orders):
        lines[:, index] = corners[:, point]
    ax.plot(-lines[0], lines[2] - 8, -lines[1], c=color, linewidth=2)
    return


def draw_space_truth(corners, ax, color):
    assert corners.shape == (3, 8)
    orders = [0, 1, 2, 3, 0, 4, 5, 6, 7, 4, 5, 1, 2, 6, 7, 3]
    lines = np.zeros((3, 16), dtype=np.float32)
    for index, point in enumerate(orders):
        lines[:, index] = corners[:, point]
    ax.plot(-lines[0], lines[2] - 8, -lines[1], c=color, linewidth=3)
    return


def draw_point_cloud(point_cloud, ax, color):
    ax.scatter(-point_cloud[0], point_cloud[2] -
               8, -point_cloud[1], c=color, s=0.5)
    return


def read_lidar(lidar_path, lidar_to_camera):
    lidar = np.fromfile(lidar_path, dtype=np.float32).reshape((-1, 4))
    lidar[:, 3] = 1.0
    camera = np.dot(lidar_to_camera, lidar.T)
    return camera


def keep_in_image(point_cloud_camera, camera_to_image, width=1240, height=370):
    image_coor = np.dot(camera_to_image,
                        np.concatenate([point_cloud_camera,
                                        np.zeros(shape=(1,
                                                        point_cloud_camera.shape[-1]))],
                                       axis=0))
    image_coor = image_coor[:2] / image_coor[2]
    keep = np.logical_and(np.logical_and(image_coor[0, :] > 0,
                                         image_coor[0, :] < 800),
                          np.logical_and(image_coor[1, :] > 0,
                                         image_coor[1, :] < 350))
    return point_cloud_camera[:, keep]


# 3D detection outputs
LABEL_DIR = '../outputs/kittiBox/val_out'

# Left RGB images
IMAGE_DIR = '../data/KittiBox/training/image_2'

LIDAR_DIR = '../data/KittiBox/training/velodyne'
CALIB_DIR = '../data/KittiBox/training/calib'

label_reader = Reader(IMAGE_DIR, LIDAR_DIR, LABEL_DIR, CALIB_DIR)
show_indices = label_reader.indices

for index in show_indices:
    data_label = label_reader.data[index]

    P2 = data_label['camera_to_image']
    image = Image.open(data_label['image_path'])
    fig = plt.figure(figsize=(14, 8))

    ax = fig.gca(projection='3d')
    ax.grid(False)
    ax.set_axis_off()
    ax.set_xlim((-10, 10))
    ax.set_ylim((10, 30))
    ax.set_zlim((-12, 0))

    # Visualize point cloud and detections in 3D space
    for tracklet in data_label['tracklets']:
        dim, loc, rot = [tracklet['dimensions'], tracklet['location'],
                         tracklet['rotation_y']]
        corners = get_corners(dim, loc, rot)
        draw_space(corners, ax, 'orange')

    point_cloud = read_lidar(
        data_label['lidar_path'],
        data_label['lidar_to_camera'])
    draw_point_cloud(point_cloud, ax, 'gray')
    plt.savefig('./visualize/{}_scan'.format(index))
    plt.close()

    fig = plt.figure(figsize=(13, 4))
    ax = fig.gca()
    ax.grid(False)
    ax.set_axis_off()
    ax.set_xlim((1240, 0))
    ax.set_ylim((374, 0))
    ax.imshow(image)

    # Visualize projected 3D bounding boxes on image plane
    for tracklet in data_label['tracklets']:
        dim, loc, rot = [tracklet['dimensions'], tracklet['location'],
                         tracklet['rotation_y']]
        corners = get_corners(dim, loc, rot)
        draw_projection(corners, P2, ax, 'orange')
    plt.savefig('./visualize/{}_proj'.format(index))
    plt.close()
