# MDE-AgriVLN - The Depth Reading Module

import numpy as np
from typing import List


def read_depth(place, id, t: List[int], frame_ratio, estimater):

   data = np.load(f"{estimater}/output/{place}_{id}/frame_{t[0]}'{t[1]}.npz", allow_pickle=True)
   depth_matrix = data["depth"] 

   frame_width = depth_matrix.shape[1]
   frame_height = depth_matrix.shape[0]

   sample_quantity = frame_ratio
   sample_interval = frame_width / sample_quantity[0]

   sample_matrix = []

   for y in range(int(sample_interval / 2), frame_height, int(sample_interval)):
      sample_map_line = []
      for x in range(int(sample_interval / 2), frame_width, int(sample_interval)):
         depth_value = depth_matrix[y, x]
         sample_map_line.append(round(float(depth_value), 2))
      sample_matrix.append(sample_map_line)

   return sample_matrix