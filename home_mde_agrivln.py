# MDE-AgriVLN - Home

import argparse
import time
import os
import json
import sys

from mde_agrivln.STL import STL
from mde_agrivln.decide import decide
from mde_agrivln.evaluate import evaluate


def check_label_format(label_path):
   valid_actions = {"[FORWARD]", "[LEFT ROTATE]", "[RIGHT ROTATE]", "[STOP]", "[WAIT]"}
   with open(label_path, "r") as f:
      labels = json.load(f)
   for i in range(len(labels)):
      entry = labels[i]
      if entry["action"] not in valid_actions:
         print(f"[ERROR] Action NO. {i} is invalid: {entry['action']}")
         return False
      if i < len(labels) - 1:
         end_time = round(labels[i]["time_range"][1], 3)
         next_start_time = round(labels[i + 1]["time_range"][0], 3)
         if end_time != next_start_time:
            print(f"[ERROR] Time steps {i} and {i+1} are not connected: {end_time} ≠ {next_start_time}")
            return False
   return True


if __name__ == '__main__':

   parser = argparse.ArgumentParser()
   parser.add_argument("-p", "--place", type=str, required=True, help="Place")
   parser.add_argument("-i", "--id_range", type=int, nargs='+', required=True, help="ID range")
   parser.add_argument("-r", "--representation", type=str, required=True, help="Representation: matrix, map or hybrid")
   parser.add_argument("-e", "--estimater", type=str, required=True, help="Monocular depth estimation model")
   parser.add_argument("-w", "--depth_matrix_width", type=int, required=False, help="Depth matrix width")
   parser.add_argument("-t", "--if_token", type=str, required=False, default='False', help="Token calculation")
   args = parser.parse_args()
   place = args.place
   id_range = args.id_range
   representation = args.representation
   estimater = args.estimater
   depth_matrix_width = args.depth_matrix_width
   if_token = args.if_token

   # check all the input information
   if len(id_range) == 1:
      id_range = id_range
   elif len(id_range) == 2:
      id_range = list(range(id_range[0], id_range[1] + 1))
   elif len(id_range) > 2:
      id_range = id_range
   else:
      print('[ERROR] Invalid ID range.')
      sys.exit(1)

   if representation == 'matrix' or representation == 'hybrid':
      depth_matrix_ratio = [16, 9]
   elif representation == 'map':
      depth_matrix_ratio = None

   if estimater not in ['depth_pro', 'depth_anything_v2', 'pixel-perfect-depth']:
      print('[ERROR] Invalid estimater.')
      sys.exit(1)

   if if_token not in ['True', 'False']:
      print('[ERROR] Invalid if_token.')
      sys.exit(1)

   # Running information
   method = 'MDE-AgriVLN'
   if if_token == 'True':
      exp = f'token-{method}-{representation}-{estimater}'
   else:
      exp = f'{method}-{representation}-{estimater}'
   LLM = 'deepseek-r1:32b'
   VLM = 'qwen2.5vl:32b'
   
   print(f'[INFO] Experiment: {exp}')
   print(f'[INFO] LLM: {LLM}')
   print(f'[INFO] VLM: {VLM}')
   print(f'[INFO] Representation: {representation}')
   print(f'[INFO] Estimater: {estimater}')
   if representation == 'matrix' or representation == 'hybrid':
      print(f'[INFO] Depth matrix ratio: {depth_matrix_ratio}')
   print(f'[INFO] Place: {place}')
   print(f'[INFO] ID range: {id_range}')
   
   for id in id_range:
      dir_path = f"runs/{exp}/{place}_{id}"
      os.makedirs(dir_path, exist_ok=True)
      label_path = f"dataset/{place}_{id}/label.json"
      if check_label_format(label_path) == False:
         print(f'[ERROR] {place}_{id} label is wrong.')
      else:
         print(f"[INFO] {place}_{id} label is correct.")
         print(f'--- {place}_{id} starts ---')

         # the subtask list module
         STL_state = False
         STL_run = 1
         STL_run_max = 3
         while STL_state == False:
            STL_state = STL(LLM, exp, place, id)
            STL_run += 1
            if STL_run > STL_run_max:
               print('[ERROR] Fail to generate STL.')
               break
            elif STL_state == False:
               print('[ERROR] Fail to generate STL. Ready to regenerate.')
            time.sleep(0.1)

         # the decision making module
         decide(VLM, exp, place, id, representation, depth_matrix_ratio, estimater, if_token)
         time.sleep(0.1)

         # the evaluation module
         evaluate(exp, place, id)
         time.sleep(1.0)

         print(f'--- {place}_{id} ends ---')