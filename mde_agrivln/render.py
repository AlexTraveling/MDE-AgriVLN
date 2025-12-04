# MDE-AgriVLN - The Depth Map Rendering Function

import numpy as np
import matplotlib.pyplot as plt
import os


def render_depth_map(npz_path, save_path, cmap='turbo_r'):

   data = np.load(npz_path)

   possible_keys = [k for k in data.keys() if 'depth' in k.lower()]
   if not possible_keys:
      raise KeyError("No depth-related array found in the .npz file.")

   depth = data[possible_keys[0]]

   depth_vis = np.clip(depth, np.percentile(depth, 1), np.percentile(depth, 99))
   depth_vis = (depth_vis - depth_vis.min()) / (depth_vis.max() - depth_vis.min())

   dpi = 100
   h, w = depth_vis.shape
   plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)

   plt.imshow(depth_vis, cmap=cmap)
   plt.axis('off')
   plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

   os.makedirs(os.path.dirname(save_path), exist_ok=True)
   plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
   plt.close()
   
   return True
