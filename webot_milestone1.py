# from controller import Robot, Camera
# import numpy as np

# # your library
# from milestone1_library import PerceptionSystem

# # ===============================
# # Webots setup
# # ===============================
# robot = Robot()
# timestep = int(robot.getBasicTimeStep())

# camera = robot.getDevice("camera")
# camera.enable(timestep)

# width = camera.getWidth()
# height = camera.getHeight()

# print("Milestone 1 – Computer Vision Controller Started")

# # ===============================
# # Initialize perception system
# # ===============================
# perception = PerceptionSystem()

# # ===============================
# # Main loop
# # ===============================
# while robot.step(timestep) != -1:

#     # --------------------------------------------------
#     # 1. Get camera image from Webots
#     # --------------------------------------------------
#     image = camera.getImage()

#     # Webots image: BGRA → RGB
#     img = np.frombuffer(image, dtype=np.uint8).reshape((height, width, 4))
#     frame_rgb = img[:, :, :3]   # drop alpha channel

#     # --------------------------------------------------
#     # 2. Run perception pipeline (ALL YOUR CODE)
#     # --------------------------------------------------
#     result = perception.process_frame(frame_rgb)

#     if not result or "features" not in result:
#         continue

#     blobs = result["blobs"]
#     features = result["features"]
#     matches = result["matches"]

#     # --------------------------------------------------
#     # 3. Interpret result (Milestone 1 objectives)
#     # --------------------------------------------------

#     print("\n--- Frame analysis ---")
#     print(f"Detected moving objects: {len(blobs)}")

#     for i, feature in enumerate(features):
#         cx, cy = feature.center

#         hist = feature.color
#         white_score = np.sum(hist[5:8, 5:8, 5:8])

#         is_goal = white_score > 0.15

#         if is_goal:
#             print(f"[GOAL - WHITE] Object {i} at ({cx:.1f}, {cy:.1f})")
#         else:
#             print(f"[DYNAMIC OBSTACLE] Object {i} at ({cx:.1f}, {cy:.1f})")


#     # ===============================
#     # 4. Tracking result
#     # ===============================
#     if matches:
#         print("\nTracking (old → new):")
#         for old_idx, new_idx in enumerate(matches):
#             print(f"Blob {old_idx} → Blob {new_idx}")


# ===========================================================
# version 2
# ==========================================================

# from controller import Robot, Camera
# import numpy as np
# from milestone1_library import PerceptionSystem

# robot = Robot()
# timestep = int(robot.getBasicTimeStep())

# camera = robot.getDevice("camera")
# camera.enable(timestep)

# width = camera.getWidth()
# height = camera.getHeight()

# perception = PerceptionSystem()

# print("Milestone 1 – Computer Vision Controller Started")

# while robot.step(timestep) != -1:

#     image = camera.getImage()
#     img = np.frombuffer(image, dtype=np.uint8).reshape((height, width, 4))
#     frame_rgb = img[:, :, :3]

#     result = perception.process_frame(frame_rgb)
#     if result is None:
#         continue

#     features = result["features"]

#     print("\n--- Frame analysis ---")

#     for i, f in enumerate(features):
#         cx, cy = f.center

#         # GOAL detection (WHITE)
#         white_score = np.sum(f.color[5:8, 5:8, 5:8])
#         is_goal = white_score > 0.15

#         if is_goal:
#             label = "GOAL (WHITE)"
#         elif f.moving:
#             label = "DYNAMIC OBSTACLE"
#         else:
#             label = "STATIC OBSTACLE"

#         print(f"[{label}] Object {i} at ({cx:.1f}, {cy:.1f})")



# ==========
# version 3
# ==========

# from controller import Robot, Camera
# import numpy as np
# from milestone1_library import PerceptionSystem

# robot = Robot()
# timestep = int(robot.getBasicTimeStep())

# camera = robot.getDevice("camera")
# camera.enable(timestep)

# width = camera.getWidth()
# height = camera.getHeight()

# perception = PerceptionSystem()

# print("Milestone 1 – Computer Vision Controller Started")

# while robot.step(timestep) != -1:
#     image = camera.getImage()
#     img = np.frombuffer(image, dtype=np.uint8).reshape((height, width, 4))
#     frame_rgb = img[:, :, :3]

#     result = perception.process_frame(frame_rgb)
#     if result is None:
#         continue

#     print("\n--- Frame analysis ---")

#     for i, f in enumerate(result["features"]):
#         cx, cy = f.center

#         if f.is_goal and f.moving:
#             label = "GOAL (DYNAMIC)"
#         elif f.is_goal:
#             label = "GOAL (STATIC)"
#         elif f.moving:
#             label = "DYNAMIC OBJECT"
#         else:
#             label = "STATIC OBJECT"

#         print(f"[{label}] Object {i} at ({cx:.1f}, {cy:.1f})")

# successful version
from controller import Robot, Camera
import numpy as np
import imageio
from milestone1_library import PerceptionSystem

robot = Robot()
timestep = int(robot.getBasicTimeStep())

camera = robot.getDevice("camera")
camera.enable(timestep)

width = camera.getWidth()
height = camera.getHeight()

perception = PerceptionSystem()

# LOAD GOAL IMAGE
goal_img = imageio.imread("C:\Perception_of_robot\goal_reference.png")[:, :, :3]
perception.load_goal_image(goal_img)

print("Milestone 1 – Computer Vision Controller Started")

while robot.step(timestep) != -1:
    image = camera.getImage()
    img = np.frombuffer(image, dtype=np.uint8).reshape((height, width, 4))
    frame_rgb = img[:, :, :3]

    result = perception.process_frame(frame_rgb)
    if result is None:
        continue

    print("\n--- Frame analysis ---")

    for i, f in enumerate(result["features"]):
        cx, cy = f.center

        if f.is_goal and f.moving:
            label = "GOAL (DYNAMIC)"
        elif f.is_goal:
            label = "GOAL (STATIC)"
        elif f.moving:
            label = "DYNAMIC OBJECT"
        else:
            label = "STATIC OBJECT"

        print(
            f"[{label}] Object {i} "
            f"score={f.goal_score:.3f} "
            f"at ({cx:.1f}, {cy:.1f})"
        )
