import os
import numpy as np

class Reader(object):

    def __init__(self, image, lidar, label, calib):
        assert os.path.exists(image)
        assert os.path.exists(label)
        assert os.path.exists(calib)
        assert os.path.exists(lidar)
        self.indices = []
        self.data = {}
        for label_file in os.listdir(label):
            if not label_file[0] == '0':
                continue
            data = {}
            data['tracklets'] = []
            index = label_file.split('.')[0]
            data['image_path'] = os.path.join(image, index + '.png')
            data['lidar_path'] = os.path.join(lidar, index + '.bin')
            self.indices.append(index)
            calib_path = os.path.join(calib, index + '.txt')
            with open(calib_path) as calib_file:
                lines = calib_file.readlines()
                data['camera_to_image'] = np.reshape(lines[2].strip().split(' ')[1:], (3, 4)).astype(np.float32)
                data['lidar_to_camera'] = np.reshape(lines[5].strip().split(' ')[1:], (3, 4)).astype(np.float32)
                calib_file.close()
            label_path = os.path.join(label, index + '.txt')
            with open(label_path) as label_file:
                lines = label_file.readlines()
                for line in lines:
                    elements = line.split(' ')
                    if not elements[0] == 'Car' and not elements[0] =='car':
                        continue
                    bbox = np.array(elements[4: 8], dtype=np.float32)
                    dimensions = np.array(elements[8: 11], dtype=np.float32)
                    location = np.array(elements[11: 14], dtype=np.float32)
                    rotation_y = np.array(elements[14], dtype=np.float32)
                    data['tracklets'].append({'bbox': bbox,
                                              'dimensions': dimensions,
                                              'location': location,
                                              'rotation_y': rotation_y})
            self.data[index] = data
        return


