"""Experimental ROS 2 / modern Gazebo adapter; requires a live simulator.

No ROS 1 or Gazebo Classic service assumptions. Native Gazebo WorldControl
requests avoid adding a custom ROS service bridge to the Academy installation.
"""
import shutil
import subprocess
import threading
import time

import rclpy
from rclpy.context import Context
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from rosgraph_msgs.msg import Clock
from cv_bridge import CvBridge


class RosBackend:
    def __init__(self, config):
        self.config = config
        if not shutil.which("gz"):
            raise RuntimeError("gz CLI missing. Run inside the container running Gazebo.")
        self.context = Context()
        rclpy.init(context=self.context)
        self.node = rclpy.create_node("followline_rl", context=self.context)
        self.executor = SingleThreadedExecutor(context=self.context)
        self.executor.add_node(self.node)
        self.bridge = CvBridge()
        self.condition = threading.Condition()
        self.clock = None
        self.epoch = 0
        self.image = None
        self.image_stamp = None
        self.closed = False
        self.has_commanded = False
        self.error = None
        self.pub = self.node.create_publisher(Twist, config.cmd_topic, 10)
        self.node.create_subscription(Image, config.camera_topic, self._image, QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE))
        self.node.create_subscription(Clock, "/clock", self._clock, QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.thread = threading.Thread(target=self._spin, daemon=True)
        self.thread.start()

    def _spin(self):
        try:
            self.executor.spin()
        except Exception as exc:
            with self.condition:
                self.error = exc
                self.condition.notify_all()

    def _image(self, msg):
        with self.condition:
            self.image = msg
            self.image_stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            self.condition.notify_all()

    def _clock(self, msg):
        stamp = msg.clock.sec + msg.clock.nanosec * 1e-9
        with self.condition:
            if self.clock is not None and stamp < self.clock - 1e-6:
                self.epoch += 1
                self.image = self.image_stamp = None
            self.clock = stamp
            self.condition.notify_all()

    def _wait(self, predicate, label):
        end = time.monotonic() + self.config.timeout
        with self.condition:
            while not predicate():
                if self.error:
                    raise RuntimeError("Backend worker failed") from self.error
                if self.closed or not self.context.ok():
                    raise RuntimeError("ROS backend has stopped")
                remaining = end-time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"Timed out waiting for {label}. clock={self.clock}, image_stamp={self.image_stamp}, epoch={self.epoch}")
                self.condition.wait(min(remaining, 0.1))

    def _control(self, request):
        from google.protobuf import text_format
        from gz.transport13 import Node
        from gz.msgs10.world_control_pb2 import WorldControl
        from gz.msgs10.boolean_pb2 import Boolean

        if not hasattr(self, "_gz_node"):
            self._gz_node = Node()
            self._gz_pose = None
            self._gz_pose_seq = 0
            import sys
            topic = f"/world/{self.config.world}/dynamic_pose/info"
            self._pose_process = subprocess.Popen(
                [sys.executable, "-u", "-m",
                 "followline_rl.gz_pose_worker", topic],
                stdout=subprocess.PIPE, text=True,
            )
            self._pose_reader = threading.Thread(
                target=self._read_pose_stream, daemon=True
            )
            self._pose_reader.start()

        message = WorldControl()
        text_format.Parse(request, message)
        service = f"/world/{self.config.world}/control"
        started = time.monotonic()
        success, response = self._gz_node.request(
            service, message, WorldControl, Boolean, 3000
        )
        if not success or not response.data:
            raise RuntimeError(
                f"Gazebo control failed: request={request!r}, "
                f"success={success}, response={response}, "
                f"elapsed={time.monotonic()-started:.3f}s"
            )

    def command(self, v, w):
        self.has_commanded = True
        msg = Twist()
        msg.linear.x, msg.angular.z = float(v), float(w)
        self.pub.publish(msg)

    def _frame_after(self, target):
        self._wait(lambda: self.image is not None and self.clock is not None
                   and self.image_stamp >= target
                   and self.image_stamp <= self.clock + 0.05,
                   "a fresh camera frame at the target simulation time")
        with self.condition:
            msg, stamp = self.image, self.image_stamp
        return self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8"), stamp

    def inspect(self):
        """Read-only: no commands, unpause or reset."""
        self._wait(lambda: self.image is not None and self.clock is not None,
                   "camera and /clock (start the exercise simulation first)")
        self._wait(lambda: self.pub.get_subscription_count() > 0, "command subscriber")
        services = subprocess.run(["gz", "service", "-l"], capture_output=True,
                                  text=True, timeout=self.config.timeout)
        expected = f"/world/{self.config.world}/control"
        if services.returncode or expected not in services.stdout.splitlines():
            raise RuntimeError(f"Missing {expected}. Run in Gazebo's container/transport partition.")
        with self.condition:
            frame = self.bridge.imgmsg_to_cv2(self.image, desired_encoding="bgr8")
            report = {"camera_topic": self.config.camera_topic, "cmd_topic": self.config.cmd_topic,
                      "clock": self.clock, "image_stamp": self.image_stamp,
                      "command_subscribers": self.pub.get_subscription_count(),
                      "command_publishers_including_probe": self.node.count_publishers(self.config.cmd_topic),
                      "world_control": expected, "ros_distro": __import__("os").environ.get("ROS_DISTRO")}
        return frame, report

    def reset(self):
        import math

        try:
            self._control("pause: false")
            self.inspect()
        except BaseException:
            try:
                self._control("pause: true")
            except Exception as error:
                self.node.get_logger().error(str(error))
            raise
        if self.node.count_publishers(self.config.cmd_topic) > 1:
            raise RuntimeError("Another velocity publisher is active.")

        xyz = (53.462000, -10.734100, 0.004068)
        roll, pitch, yaw = 0.000076, 0.010359, -1.570000
        cr, sr = math.cos(roll/2), math.sin(roll/2)
        cp, sp = math.cos(pitch/2), math.sin(pitch/2)
        cy, sy = math.cos(yaw/2), math.sin(yaw/2)
        q = (sr*cp*cy-cr*sp*sy, cr*sp*cy+sr*cp*sy,
             cr*cp*sy-sr*sp*cy, cr*cp*cy+sr*sp*sy)

        self.command(0, 0)
        try:
            # Let zero velocity reach the robot and allow it to stop.
            self._control("pause: false")
            with self.condition:
                target = self.clock + 0.5
            self._frame_after(target)
            self._control("pause: true")

            request = (
                'name: "f1" position: {'
                f'x: {xyz[0]} y: {xyz[1]} z: {xyz[2]}'
                '} orientation: {'
                f'x: {q[0]} y: {q[1]} z: {q[2]} w: {q[3]}'
                '}'
            )
            from google.protobuf import text_format
            from gz.msgs10.pose_pb2 import Pose
            from gz.msgs10.boolean_pb2 import Boolean

            message = Pose()
            text_format.Parse(request, message)
            success, response = self._gz_node.request(
                f"/world/{self.config.world}/set_pose",
                message, Pose, Boolean, 5000
            )
            if not success or not response.data:
                raise RuntimeError(
                    f"Pose reset failed: success={success}, response={response}"
                )
            with self.condition:
                pose_seq = self._gz_pose_seq

            # Apply the pose and wait for an image captured afterwards.
            self._control("pause: false")
            with self.condition:
                target = self.clock + 0.5
            frame = self._frame_after(target)
        finally:
            self._control("pause: true")

        self._wait(
            lambda: self._gz_pose is not None
                    and self._gz_pose_seq > pose_seq,
            "a fresh f1 pose after reset"
        )
        with self.condition:
            position, actual_yaw = self._gz_pose

        distance = math.dist(position, xyz)
        yaw_error = abs(math.atan2(
            math.sin(actual_yaw-yaw), math.cos(actual_yaw-yaw)
        ))
        if distance > 0.15 or yaw_error > 0.15:
            raise RuntimeError(
                f"Reset pose incorrect: position={position}, yaw={actual_yaw}"
            )
        print(
            f"Reset verified: position={position}, yaw={actual_yaw}",
            flush=True
        )
        return frame

    def advance(self, v, w, dt):
        self.command(v, w)
        # Allow the ROS-to-Gazebo bridge to deliver the command while paused.
        time.sleep(0.02)
        with self.condition:
            start, epoch = self.clock, self.epoch
        try:
            self._control("pause: false")
            frame = self._frame_after(start + dt)
            if self.epoch != epoch:
                raise RuntimeError("Simulation reset during a training step")
            return frame
        finally:
            self._control("pause: true")

    def stop(self):
        if not self.has_commanded:
            return
        self.command(0, 0)
        try:
            self._control("pause: true")
        except Exception as exc:
            self.node.get_logger().error(f"Could not pause simulation on stop: {exc}")

    def close(self):
        if self.closed:
            return
        # Callers own the simulation state; inspect() itself never pauses it.
        self.stop()
        time.sleep(0.05)
        self.closed = True
        if hasattr(self, "_pose_process"):
            self._pose_process.terminate()
            try:
                self._pose_process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._pose_process.kill()
                self._pose_process.wait()
            self._pose_reader.join(timeout=2)
        self.executor.shutdown(timeout_sec=2)
        self.thread.join(timeout=2)
        self.node.destroy_node()
        self.context.shutdown()

    def _read_pose_stream(self):
        import json
        for line in self._pose_process.stdout:
            try:
                position, yaw = json.loads(line)
            except (ValueError, TypeError):
                continue
            with self.condition:
                self._gz_pose = (tuple(position), float(yaw))
                self._gz_pose_seq += 1
                self.condition.notify_all()
        with self.condition:
            if not self.closed:
                self.error = RuntimeError("Gazebo pose monitor exited")
                self.condition.notify_all()
