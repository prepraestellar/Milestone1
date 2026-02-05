# import numpy as np
# import math

# # =========================================================
# # 1. BASIC IMAGE OPS
# # =========================================================

# def rgb2gray(rgb):
#     return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])

# def convolution(kernel, array):
#     pad = kernel.shape[0] // 2
#     padded = np.pad(array, pad, mode='constant')
#     out = np.zeros_like(array, dtype=np.float32)

#     for i in range(array.shape[0]):
#         for j in range(array.shape[1]):
#             region = padded[i:i+kernel.shape[0], j:j+kernel.shape[1]]
#             out[i, j] = np.sum(region * kernel)
#     return out


# # =========================================================
# # 2. EDGE DETECTION
# # =========================================================

# def edge_detection(gray):
#     sobel_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
#     sobel_y = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])

#     gx = convolution(sobel_x, gray)
#     gy = convolution(sobel_y, gray)

#     mag = np.sqrt(gx**2 + gy**2)
#     mag = (mag / (np.max(mag) + 1e-8)) * 255
#     return mag, gx, gy


# # =========================================================
# # 3. BLOB CREATION
# # =========================================================

# def blob_ize(binary):
#     visited = np.zeros_like(binary, dtype=np.uint8)
#     blobs = []

#     rows, cols = binary.shape

#     for i in range(rows):
#         for j in range(cols):
#             if binary[i, j] == 0 or visited[i, j]:
#                 continue

#             stack = [(i, j)]
#             blob = set()

#             while stack:
#                 r, c = stack.pop()
#                 if r < 0 or r >= rows or c < 0 or c >= cols:
#                     continue
#                 if visited[r, c] or binary[r, c] == 0:
#                     continue

#                 visited[r, c] = 1
#                 blob.add((r, c))
#                 stack.extend([(r+1,c),(r-1,c),(r,c+1),(r,c-1)])

#             if len(blob) > 600:
#                 blobs.append(blob)

#     return blobs


# # =========================================================
# # 4. FEATURE EXTRACTION
# # =========================================================

# def center_of_mass(blob):
#     xs = [c for _, c in blob]
#     ys = [r for r, _ in blob]
#     return np.mean(xs), np.mean(ys)

# def color_histogram(blob, frame):
#     hist = np.zeros((8,8,8), dtype=np.float32)
#     for r,c in blob:
#         R,G,B = frame[r,c]
#         hist[R//32, G//32, B//32] += 1
#     hist /= np.sum(hist) + 1e-8
#     return hist

# def hog_histogram(blob, gx, gy, bins=8):
#     hist = np.zeros(bins, dtype=np.float32)
#     for r,c in blob:
#         mag = math.sqrt(gx[r,c]**2 + gy[r,c]**2)
#         angle = (math.degrees(math.atan2(gy[r,c], gx[r,c])) + 180) % 180
#         hist[int(angle // (180/bins))] += mag
#     hist /= np.sum(hist) + 1e-8
#     return hist


# # =========================================================
# # 5. DISTANCES & SCORING
# # =========================================================

# def euclidean(p1, p2):
#     return np.linalg.norm(np.array(p1) - np.array(p2))

# def bhattacharyya(h1, h2):
#     return -np.log(np.sum(np.sqrt(h1 * h2)) + 1e-8)

# class BlobFeature:
#     def __init__(self, blob, frame, gx, gy):
#         self.center = center_of_mass(blob)
#         self.color = color_histogram(blob, frame)
#         self.hog = hog_histogram(blob, gx, gy)

# def blob_score(b1, b2, w=(0.4, 0.4, 0.2)):
#     return (
#         w[0] * euclidean(b1.center, b2.center) +
#         w[1] * bhattacharyya(b1.color, b2.color) +
#         w[2] * bhattacharyya(b1.hog, b2.hog)
#     )




# # =========================================================
# # 6. MAIN PERCEPTION CLASS (WHAT CONTROLLER USES)
# # =========================================================

# class PerceptionSystem:
#     def __init__(self):
#         self.prev_gray = None
#         self.prev_features = None

#         self.blur_kernel = np.array([
#             [1,2,1],
#             [2,4,2],
#             [1,2,1]
#         ]) / 16.0

#     def process_frame(self, frame_rgb):
#         gray = rgb2gray(frame_rgb)
#         gray = convolution(self.blur_kernel, gray)

#         if self.prev_gray is None:
#             self.prev_gray = gray
#             return {
#                 "blobs": [],
#                 "features": [],
#                 "matches": []
#          }

#         # Background subtraction
#         # diff = np.abs(gray - self.prev_gray)

#         # if self.motion_acc is None:
#         #     self.motion_acc = diff
#         # else:
#         #     self.motion_acc = 0.8 * self.motion_acc + diff

#         # moving_mask = self.motion_acc > 8

#         diff = gray - self.prev_gray
#         moving_mask = np.abs(diff > 12)

#         # Edge detection
#         edges, gx, gy = edge_detection(gray)
#         edges = np.where(edges > 25, 255, 0)

#         # Blob creation
#         blobs = blob_ize(edges)

#         # Keep only moving blobs
#         surviving = set(zip(*np.where(moving_mask)))
#         moving_blobs = [
#             blob for blob in blobs
#             if any(p in surviving for p in blob)
#         ]

#         # Feature extraction
#         features = [
#             BlobFeature(blob, frame_rgb, gx, gy)
#             for blob in moving_blobs
#         ]

#         # Tracking
#         matches = []
#         if self.prev_features is not None:
#             for old in self.prev_features:
#                 scores = [
#                     blob_score(old, new)
#                     for new in features
#                 ]
#                 if scores:
#                     matches.append(np.argmin(scores))

#         self.prev_gray = gray
#         self.prev_features = features

#         return {
#             "blobs": moving_blobs,
#             "features": features,
#             "matches": matches
#         }




# =========================================
# can detect static vs dynamic obstacles correctly
# but still cant track 2 balls properly (dynamic yellow ball and the stationary goal ball still appear as 1 sentence = goal with dynamic value)


# import numpy as np
# import math

# # ===============================
# # Utility functions
# # ===============================

# def center_of_mass(blob):
#     xs = [p[1] for p in blob]
#     ys = [p[0] for p in blob]
#     return (np.mean(xs), np.mean(ys))


# def color_histogram(blob, frame):
#     hist = np.zeros((8, 8, 8), dtype=np.float32)

#     for (r, c) in blob:
#         R, G, B = frame[r, c]
#         hist[R//32, G//32, B//32] += 1

#     hist /= np.sum(hist) + 1e-8
#     return hist


# def hog_histogram(blob, grad_x, grad_y, num_bins=8):
#     hist = np.zeros(num_bins, dtype=np.float32)

#     for (r, c) in blob:
#         gx = grad_x[r, c]
#         gy = grad_y[r, c]

#         mag = math.sqrt(gx*gx + gy*gy)
#         angle = (math.degrees(math.atan2(gy, gx)) + 180) % 180
#         bin_idx = int(angle // (180 / num_bins))

#         hist[bin_idx] += mag

#     hist /= np.sum(hist) + 1e-8
#     return hist


# def convolution(kernel, array):
#     k = kernel.shape[0]
#     pad = k // 2
#     padded = np.pad(array, pad, mode="constant")
#     out = np.zeros_like(array)

#     for i in range(array.shape[0]):
#         for j in range(array.shape[1]):
#             region = padded[i:i+k, j:j+k]
#             out[i, j] = np.sum(region * kernel)

#     return out


# def compute_gradients(gray):
#     sobel_x = np.array([[-1, 0, 1],
#                         [-2, 0, 2],
#                         [-1, 0, 1]])
#     sobel_y = np.array([[1, 2, 1],
#                         [0, 0, 0],
#                         [-1, -2, -1]])
#     gx = convolution(sobel_x, gray)
#     gy = convolution(sobel_y, gray)
#     return gx, gy


# # ===============================
# # Blob utilities
# # ===============================

# def blob_ize(binary):
#     visited = np.zeros_like(binary, dtype=bool)
#     blobs = []

#     for r in range(binary.shape[0]):
#         for c in range(binary.shape[1]):
#             if binary[r, c] == 0 or visited[r, c]:
#                 continue

#             stack = [(r, c)]
#             blob = set()

#             while stack:
#                 x, y = stack.pop()
#                 if (x < 0 or x >= binary.shape[0] or
#                     y < 0 or y >= binary.shape[1]):
#                     continue
#                 if visited[x, y] or binary[x, y] == 0:
#                     continue

#                 visited[x, y] = True
#                 blob.add((x, y))

#                 stack += [(x+1,y), (x-1,y), (x,y+1), (x,y-1)]

#             if len(blob) > 300:
#                 blobs.append(blob)

#     return blobs


# # ===============================
# # Feature object
# # ===============================

# class BlobFeature:
#     def __init__(self, blob, frame, gx, gy, motion_mask):
#         self.blob = blob
#         self.center = center_of_mass(blob)
#         self.color = color_histogram(blob, frame)
#         self.hog = hog_histogram(blob, gx, gy)

#         # ⭐ THIS IS THE FIX ⭐
#         self.moving = any(motion_mask[r, c] for (r, c) in blob)


# # ===============================
# # Main perception system
# # ===============================

# class PerceptionSystem:
#     def __init__(self):
#         self.prev_gray = None

#         self.blur_kernel = np.array([[1,2,1],
#                                      [2,4,2],
#                                      [1,2,1]]) / 16.0

#     def process_frame(self, frame_rgb):
#         gray = np.dot(frame_rgb[...,:3], [0.299, 0.587, 0.114])

#         # 1. Motion detection
#         if self.prev_gray is None:
#             self.prev_gray = gray
#             return None

#         diff = gray - self.prev_gray
#         diff = convolution(self.blur_kernel, diff)
#         motion_mask = diff > 25

#         # 2. Edge detection
#         gx, gy = compute_gradients(gray)
#         mag = np.sqrt(gx**2 + gy**2)
#         edges = mag > 40

#         # 3. Blob detection
#         blobs = blob_ize(edges)

#         # 4. Feature extraction (PER BLOB)
#         features = [
#             BlobFeature(blob, frame_rgb, gx, gy, motion_mask)
#             for blob in blobs
#         ]

#         self.prev_gray = gray

#         return {
#             "blobs": blobs,
#             "features": features
#         }





# =============================================
# previous version before the rrode dilateion fix
# =============================================


# import numpy as np
# import math

# # ===============================
# # Utility functions
# # ===============================

# def center_of_mass(blob):
#     xs = [p[1] for p in blob]
#     ys = [p[0] for p in blob]
#     return (np.mean(xs), np.mean(ys))


# def color_histogram(blob, frame):
#     hist = np.zeros((8, 8, 8), dtype=np.float32)
#     for (r, c) in blob:
#         R, G, B = frame[r, c]
#         hist[R//32, G//32, B//32] += 1
#     hist /= np.sum(hist) + 1e-8
#     return hist


# def hog_histogram(blob, gx, gy, bins=8):
#     hist = np.zeros(bins, dtype=np.float32)
#     for (r, c) in blob:
#         mag = math.sqrt(gx[r, c]**2 + gy[r, c]**2)
#         angle = (math.degrees(math.atan2(gy[r, c], gx[r, c])) + 180) % 180
#         hist[int(angle // (180 / bins))] += mag
#     hist /= np.sum(hist) + 1e-8
#     return hist


# def convolution(kernel, array):
#     k = kernel.shape[0]
#     pad = k // 2
#     padded = np.pad(array, pad, mode="constant")
#     out = np.zeros_like(array)
#     for i in range(array.shape[0]):
#         for j in range(array.shape[1]):
#             region = padded[i:i+k, j:j+k]
#             out[i, j] = np.sum(region * kernel)
#     return out


# def compute_gradients(gray):
#     sobel_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
#     sobel_y = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
#     gx = convolution(sobel_x, gray)
#     gy = convolution(sobel_y, gray)
#     return gx, gy


# # ===============================
# # Blob detection
# # ===============================

# def blob_ize(binary):
#     visited = np.zeros_like(binary, dtype=bool)
#     blobs = []

#     for r in range(binary.shape[0]):
#         for c in range(binary.shape[1]):
#             if binary[r, c] == 0 or visited[r, c]:
#                 continue

#             stack = [(r, c)]
#             blob = set()

#             while stack:
#                 x, y = stack.pop()
#                 if x < 0 or y < 0 or x >= binary.shape[0] or y >= binary.shape[1]:
#                     continue
#                 if visited[x, y] or binary[x, y] == 0:
#                     continue

#                 visited[x, y] = True
#                 blob.add((x, y))
#                 stack += [(x+1,y), (x-1,y), (x,y+1), (x,y-1)]

#             if len(blob) > 300:
#                 blobs.append(blob)

#     return blobs


# def compute_blob_motion(blob, motion_mask):
#     moving_pixels = sum(1 for (r, c) in blob if motion_mask[r, c])
#     return moving_pixels / (len(blob) + 1e-8)


# # ===============================
# # Feature object
# # ===============================

# class BlobFeature:
#     def __init__(self, blob, frame, gx, gy, motion_mask):
#         self.blob = blob
#         self.center = center_of_mass(blob)
#         self.color = color_histogram(blob, frame)
#         self.hog = hog_histogram(blob, gx, gy)

#         self.moving_ratio = compute_blob_motion(blob, motion_mask)
#         self.moving = self.moving_ratio > 0.05

#         white_score = np.sum(self.color[5:8, 5:8, 5:8])
#         self.is_goal = white_score > 0.15

#         # red_score = np.sum(self.color[6:8, 0:2, 0:2])
#         # self.is_goal = red_score > 0.15

# # ===============================
# # Perception system
# # ===============================

# class PerceptionSystem:
#     def __init__(self):
#         self.prev_gray = None
#         self.blur_kernel = np.array([[1,2,1],
#                                      [2,4,2],
#                                      [1,2,1]]) / 16.0

#     def process_frame(self, frame_rgb):
#         gray = np.dot(frame_rgb[...,:3], [0.299, 0.587, 0.114])

#         if self.prev_gray is None:
#             self.prev_gray = gray
#             return None

#         diff = convolution(self.blur_kernel, np.abs(gray - self.prev_gray))
#         motion_mask = diff > 6 #25

#         gx, gy = compute_gradients(gray)
#         edges = np.sqrt(gx**2 + gy**2) > 80
#         blobs = blob_ize(edges)

#         features = [
#             BlobFeature(blob, frame_rgb, gx, gy, motion_mask)
#             for blob in blobs
#         ]

#         self.prev_gray = gray

#         return {
#             "features": features
#         }



# ==================================
# successful version: after erode, dilate and change the len blob > 50 -> can print out more than 1 object now

# ==================================
# import numpy as np
# import math

# # ===============================
# # Utility functions
# # ===============================

# def center_of_mass(blob):
#     xs = [p[1] for p in blob]
#     ys = [p[0] for p in blob]
#     return (np.mean(xs), np.mean(ys))


# def color_histogram(blob, frame):
#     hist = np.zeros((8, 8, 8), dtype=np.float32)
#     for (r, c) in blob:
#         R, G, B = frame[r, c]
#         hist[R // 32, G // 32, B // 32] += 1
#     hist /= np.sum(hist) + 1e-8
#     return hist.flatten()


# def hog_histogram(blob, gx, gy, bins=8):
#     hist = np.zeros(bins, dtype=np.float32)
#     for (r, c) in blob:
#         mag = math.sqrt(gx[r, c]**2 + gy[r, c]**2)
#         angle = (math.degrees(math.atan2(gy[r, c], gx[r, c])) + 180) % 180
#         hist[int(angle // (180 / bins))] += mag
#     hist /= np.sum(hist) + 1e-8
#     return hist


# def bhattacharyya_distance(h1, h2):
#     return np.sqrt(1 - np.sum(np.sqrt(h1 * h2)))


# def convolution(kernel, array):
#     k = kernel.shape[0]
#     pad = k // 2
#     padded = np.pad(array, pad, mode="constant")
#     out = np.zeros_like(array)
#     for i in range(array.shape[0]):
#         for j in range(array.shape[1]):
#             out[i, j] = np.sum(padded[i:i+k, j:j+k] * kernel)
#     return out


# def compute_gradients(gray):
#     sobel_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
#     sobel_y = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
#     gx = convolution(sobel_x, gray)
#     gy = convolution(sobel_y, gray)
#     return gx, gy


# # ===============================
# # Morphology
# # ===============================

# def binary_erosion(binary):
#     padded = np.pad(binary, 1, mode='constant', constant_values=False)
#     return (
#         padded[1:-1, 1:-1] &
#         padded[:-2, 1:-1] &
#         padded[2:, 1:-1] &
#         padded[1:-1, :-2] &
#         padded[1:-1, 2:]
#     )


# def binary_dilation(binary):
#     padded = np.pad(binary, 1, mode='constant', constant_values=False)
#     return (
#         padded[1:-1, 1:-1] |
#         padded[:-2, 1:-1] |
#         padded[2:, 1:-1] |
#         padded[1:-1, :-2] |
#         padded[1:-1, 2:]
#     )


# # ===============================
# # Blob detection
# # ===============================

# def blob_ize(binary):
#     visited = np.zeros_like(binary, dtype=bool)
#     blobs = []

#     for r in range(binary.shape[0]):
#         for c in range(binary.shape[1]):
#             if not binary[r, c] or visited[r, c]:
#                 continue

#             stack = [(r, c)]
#             blob = set()

#             while stack:
#                 x, y = stack.pop()
#                 if x < 0 or y < 0 or x >= binary.shape[0] or y >= binary.shape[1]:
#                     continue
#                 if visited[x, y] or not binary[x, y]:
#                     continue

#                 visited[x, y] = True
#                 blob.add((x, y))
#                 stack.extend([(x+1,y), (x-1,y), (x,y+1), (x,y-1)])

#             if len(blob) > 50:
#                 blobs.append(blob)

#     return blobs


# def compute_blob_motion(blob, motion_mask):
#     return sum(motion_mask[r, c] for (r, c) in blob) / (len(blob) + 1e-8)


# # ===============================
# # Blob feature
# # ===============================

# class BlobFeature:
#     def __init__(self, blob, frame, gx, gy, motion_mask):
#         self.blob = blob
#         self.center = center_of_mass(blob)
#         self.color = color_histogram(blob, frame)
#         self.hog = hog_histogram(blob, gx, gy)

#         self.moving_ratio = compute_blob_motion(blob, motion_mask)
#         self.moving = self.moving_ratio > 0.05

#         self.goal_score = None
#         self.is_goal = False


# # ===============================
# # Perception System
# # ===============================

# class PerceptionSystem:
#     def __init__(self):
#         self.prev_gray = None
#         self.goal_color = None
#         self.goal_hog = None

#         self.blur_kernel = np.array([
#             [1,2,1],
#             [2,4,2],
#             [1,2,1]
#         ]) / 16.0

#     def load_goal_image(self, goal_img):
#         gray = np.dot(goal_img[..., :3], [0.299, 0.587, 0.114])
#         gx, gy = compute_gradients(gray)

#         h, w = gray.shape
#         blob = {(r, c) for r in range(h) for c in range(w)}

#         self.goal_color = color_histogram(blob, goal_img)
#         self.goal_hog = hog_histogram(blob, gx, gy)

#         print("Goal image loaded")

#     def process_frame(self, frame_rgb):
#         gray = np.dot(frame_rgb[..., :3], [0.299, 0.587, 0.114])

#         if self.prev_gray is None:
#             self.prev_gray = gray
#             return None

#         diff = convolution(self.blur_kernel, np.abs(gray - self.prev_gray))
#         motion_mask = diff > 6

#         gx, gy = compute_gradients(gray)
#         mag = np.sqrt(gx**2 + gy**2)

#         edges = mag > 60
#         edges = binary_erosion(edges)
#         edges = binary_dilation(edges)

#         blobs = blob_ize(edges)

#         features = [
#             BlobFeature(blob, frame_rgb, gx, gy, motion_mask)
#             for blob in blobs
#         ]

#         if self.goal_color is not None:
#             for f in features:
#                 Dc = bhattacharyya_distance(f.color, self.goal_color)
#                 Ds = bhattacharyya_distance(f.hog, self.goal_hog)
#                 f.goal_score = 0.5*Dc + 0.5*Ds
#                 f.is_goal = f.goal_score < 0.6   # tune if needed

#         self.prev_gray = gray
#         return {"features": features}

import numpy as np
import math

# ===============================
# Utility functions
# ===============================

def center_of_mass(blob):
    xs = [p[1] for p in blob]
    ys = [p[0] for p in blob]
    return (np.mean(xs), np.mean(ys))


def bounding_box(blob):
    rows = [p[0] for p in blob]
    cols = [p[1] for p in blob]
    return min(rows), min(cols), max(rows), max(cols)


def color_histogram(blob, frame):
    hist = np.zeros((8, 8, 8), dtype=np.float32)
    for (r, c) in blob:
        R, G, B = frame[r, c]
        hist[R // 32, G // 32, B // 32] += 1
    hist /= np.sum(hist) + 1e-8
    return hist.flatten()


def hog_histogram(blob, gx, gy, bins=8):
    hist = np.zeros(bins, dtype=np.float32)
    for (r, c) in blob:
        mag = math.sqrt(gx[r, c]**2 + gy[r, c]**2)
        angle = (math.degrees(math.atan2(gy[r, c], gx[r, c])) + 180) % 180
        hist[int(angle // (180 / bins))] += mag
    hist /= np.sum(hist) + 1e-8
    return hist


def bhattacharyya_distance(h1, h2):
    return np.sqrt(1 - np.sum(np.sqrt(h1 * h2)))


def convolution(kernel, array):
    k = kernel.shape[0]
    pad = k // 2
    padded = np.pad(array, pad, mode="constant")
    out = np.zeros_like(array)
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            out[i, j] = np.sum(padded[i:i+k, j:j+k] * kernel)
    return out


def compute_gradients(gray):
    sobel_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    sobel_y = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    gx = convolution(sobel_x, gray)
    gy = convolution(sobel_y, gray)
    return gx, gy


# ===============================
# Morphology
# ===============================

def binary_erosion(binary):
    padded = np.pad(binary, 1, mode='constant', constant_values=False)
    return (
        padded[1:-1, 1:-1] &
        padded[:-2, 1:-1] &
        padded[2:, 1:-1] &
        padded[1:-1, :-2] &
        padded[1:-1, 2:]
    )


def binary_dilation(binary):
    padded = np.pad(binary, 1, mode='constant', constant_values=False)
    return (
        padded[1:-1, 1:-1] |
        padded[:-2, 1:-1] |
        padded[2:, 1:-1] |
        padded[1:-1, :-2] |
        padded[1:-1, 2:]
    )


# ===============================
# Blob detection
# ===============================

def blob_ize(binary):
    visited = np.zeros_like(binary, dtype=bool)
    blobs = []

    for r in range(binary.shape[0]):
        for c in range(binary.shape[1]):
            if not binary[r, c] or visited[r, c]:
                continue

            stack = [(r, c)]
            blob = set()

            while stack:
                x, y = stack.pop()
                if x < 0 or y < 0 or x >= binary.shape[0] or y >= binary.shape[1]:
                    continue
                if visited[x, y] or not binary[x, y]:
                    continue

                visited[x, y] = True
                blob.add((x, y))
                stack.extend([(x+1,y), (x-1,y), (x,y+1), (x,y-1)])

            if len(blob) > 50:
                blobs.append(blob)

    return blobs


def compute_blob_motion(blob, motion_mask):
    return sum(motion_mask[r, c] for (r, c) in blob) / (len(blob) + 1e-8)


# ===============================
# Blob feature
# ===============================

class BlobFeature:
    def __init__(self, blob, frame, gx, gy, motion_mask):
        self.blob = blob
        self.center = center_of_mass(blob)
        self.bbox = bounding_box(blob)  # 🔹 ADD ONLY THIS

        self.color = color_histogram(blob, frame)
        self.hog = hog_histogram(blob, gx, gy)

        self.moving_ratio = compute_blob_motion(blob, motion_mask)
        self.moving = self.moving_ratio > 0.05

        self.goal_score = None
        self.is_goal = False


# ===============================
# Perception System
# ===============================

class PerceptionSystem:
    def __init__(self):
        self.prev_gray = None
        self.goal_color = None
        self.goal_hog = None

        self.blur_kernel = np.array([
            [1,2,1],
            [2,4,2],
            [1,2,1]
        ]) / 16.0

    def load_goal_image(self, goal_img):
        gray = np.dot(goal_img[..., :3], [0.299, 0.587, 0.114])
        gx, gy = compute_gradients(gray)

        h, w = gray.shape
        blob = {(r, c) for r in range(h) for c in range(w)}

        self.goal_color = color_histogram(blob, goal_img)
        self.goal_hog = hog_histogram(blob, gx, gy)

        print("Goal image loaded")

    def process_frame(self, frame_rgb):
        gray = np.dot(frame_rgb[..., :3], [0.299, 0.587, 0.114])

        if self.prev_gray is None:
            self.prev_gray = gray
            return None

        diff = convolution(self.blur_kernel, np.abs(gray - self.prev_gray))
        motion_mask = diff > 6

        gx, gy = compute_gradients(gray)
        mag = np.sqrt(gx**2 + gy**2)

        edges = mag > 80 #60
        edges = binary_erosion(edges)
        edges = binary_dilation(edges)

        blobs = blob_ize(edges)

        features = [
            BlobFeature(blob, frame_rgb, gx, gy, motion_mask)
            for blob in blobs
        ]

        if self.goal_color is not None:
            for f in features:
                Dc = bhattacharyya_distance(f.color, self.goal_color)
                Ds = bhattacharyya_distance(f.hog, self.goal_hog)
                f.goal_score = 0.6 * Dc + 0.4 * Ds
                f.is_goal = f.goal_score < 0.6

        self.prev_gray = gray
        return {"features": features}
