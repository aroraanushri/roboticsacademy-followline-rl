import json
import math
import sys
import time
from gz.transport13 import Node
from gz.msgs10.pose_v_pb2 import Pose_V

def on_pose(message):
    for pose in message.pose:
        if pose.name == "f1":
            q = pose.orientation
            yaw = math.atan2(
                2*(q.w*q.z + q.x*q.y),
                1-2*(q.y*q.y + q.z*q.z)
            )
            position = [
                pose.position.x, pose.position.y, pose.position.z
            ]
            print(json.dumps([position, yaw]), flush=True)
            return

node = Node()
if not node.subscribe(Pose_V, sys.argv[1], on_pose):
    raise RuntimeError("Cannot subscribe to Gazebo poses")
while True:
    time.sleep(1)
