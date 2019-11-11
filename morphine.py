# Nurdogan Karaman
# 150150141

import moviepy.editor as mpy
import dlib
import cv2
import numpy as np
from os import listdir
from os.path import isfile, join


# CONVERTING THE 68 FACIAL LANDMARKS TO (X, Y)
def to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    return coords
# CONVERTS RECTANGLE TO (X, Y, W, H)
def to_bounds(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    return (x, y, w, h)
def find_where(where_pt):
    here = None
    for num in where_pt[0]:
        here = num
        break
    return here

# PUTS FACIAL LANDMARKS TO IMAGE
def draw_landmarks_and_rect(image, landmarks, rectangles):
    # CONVERTING DLIB'S RECTANGLE TO A OPENCV BOUNDING BOX
    if rectangles:
        (m, n, o, p) = to_bounds(rectangles[0])
        cv2.rectangle(image, (m, n), (m + o, n + p), (232, 37, 73), 2)

    # PUT LANDMARKS
    for (x, y) in landmarks:
        cv2.rectangle(image, (x + 3, y + 3), (x - 3, y - 3), (143, 15, 255), -1)
# DRAWS DELAUNAY TRIANGLES TO IMAGE
def draw_delaunay_triangles(image, triangles):
    for i in range(len(triangles)):
        tri = triangles[i]
        # DRAWING LINE
        cv2.line(image, tri[0], tri[1], (0, 255, 0))
        cv2.line(image, tri[0], tri[2], (0, 255, 0))
        cv2.line(image, tri[1], tri[2], (0, 255, 0))

    return image

# DELAUNAY TRIANGULATION
def simple_delaunay_triangles(image, points):
    # APPENDING NEW POINTS TO LANDMARKS
    right = image.shape[0] - 1
    bottom = image.shape[1] - 1
    edge_points = np.array(((0, 0), (0, int(right / 2)), (0, right), (int(bottom / 2), right),
                    (bottom, right), (bottom, int(right / 2)), (bottom, 0), (int(bottom / 2), 0)))
    points = np.append(points, edge_points, axis=0)

    # TRIANGULATION
    subdiv = cv2.Subdiv2D((0, 0, image.shape[1], image.shape[0]))
    for i in range(0,76):
        subdiv.insert((points[i][0], points[i][1]))

    triangles = subdiv.getTriangleList().astype(int)
    triangles = triangles.reshape((len(triangles), 3 ,2))
    triang = []
    for tri in triangles:
        # TRIANGLE CORNERS
        pt1 = tuple(tri[0])
        pt2 = tuple(tri[1])
        pt3 = tuple(tri[2])
        triang.append((pt1,pt2,pt3))
    triangles = triang

    return triangles
# DELAUNAY TRIANGULATION ACC TO REFERENCE
def referenced_delaunay_triangles(image, points, points_ref, triangles_ref):
    right = image.shape[0] - 1
    bottom = image.shape[1] - 1
    edge_points = np.array(((0, 0), (0, int(right / 2)), (0, right), (int(bottom / 2), right),
                    (bottom, right), (bottom, int(right / 2)), (bottom, 0), (int(bottom / 2), 0)))

    # CALCULATING OFFSETS OF TRIANGLES
    points_ref = np.append(points_ref, edge_points, axis=0)
    off_triangles = []
    for i in range(len(triangles_ref)):
        tri = triangles_ref[i]

        off_pt1 = np.where((points_ref == tri[0]).all(axis=1))
        off_pt1 = find_where(off_pt1)
        off_pt2 = np.where((points_ref == tri[1]).all(axis=1))
        off_pt2 = find_where(off_pt2)
        off_pt3 = np.where((points_ref == tri[2]).all(axis=1))
        off_pt3 = find_where(off_pt3)

        if off_pt1 is not None and off_pt2 is not None and off_pt3 is not None:
            off_triangle_corners = [off_pt1, off_pt2, off_pt3]
            off_triangles.append(off_triangle_corners)

    # CALCULATING TRIANGLES ACC TO REFERENCES
    points = np.append(points, edge_points, axis=0)
    triangles = []
    for tri in off_triangles:
        # TRIANGLE CORNERS
        pt1 = tuple(points[tri[0]])
        pt2 = tuple(points[tri[1]])
        pt3 = tuple(points[tri[2]])
        triangles.append((pt1, pt2, pt3))

    return triangles

# CROPS HUMAN IMAGE ACC TO DETECTED FACE
def crop_human(image, dimensions):
    detector = dlib.get_frontal_face_detector()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # DETECTING FACE
    rectangles = detector(gray)
    # CROPPING THE IMAGE
    rectangle = rectangles[0]

    # DIMENSIONS OF THE CROPPING IMAGE
    width = rectangle.width() + (rectangle.width() % 2) + 100
    height = (width / 3) * 4

    height_p = int((height - rectangle.height()) / 2)
    width_p = int((width - rectangle.width()) / 2)

    # BOUNDS OF CROPPING IMAGE
    t = int(rectangle.top() - height_p)
    b = t + int(height)
    l = int(rectangle.left() - width_p)
    r = l + int(width)

    # CROPPING IMAGE
    cropped = image[t:b, l:r]
    # RESIZING IMAGE
    cropped = cv2.resize(cropped, dimensions)

    return cropped
# BRINGS LANDMARKS OF HUMAN IMAGE
def human_landmarks_bringer(image, which_img):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(path + "shape_predictor_68_face_landmarks.dat")

    # DETECTING FACE AGAIN BECAUSE OF CROPPING
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rectangles = detector(gray)

    try:
        landmarks = predictor(gray, rectangles[0])
    except:
        exit("HUMAN {} cannot be cropped as wanted. Please try another image.".format(which_img+1))
    landmarks = to_np(landmarks)

    return landmarks, rectangles

# PRE-PROCESSING TEMPLATE POINTS
def pre_pro_temp_points(dst_cat_eyes, dst_cat_mouth, cat_points):
    # READ TEMPLATE POINTS
    points = np.load(path + "template_points.npy")
    points = points.transpose()
    points[0], points[1] = points[1], np.array(points[0])
    points = points.transpose()

    # # CALCULATING SOME WANTED VALUES FOR PART 2
    apple_of_left_eye = ((points[36] + points[39]) / 2).astype(int)
    apple_of_right_eye = ((points[42] + points[45]) / 2).astype(int)
    mid_of_eyes = ((apple_of_right_eye + apple_of_left_eye) / 2).astype(int)

    dst_hum_eyes = np.linalg.norm(apple_of_right_eye - apple_of_left_eye)
    dst_hum_mouth = np.linalg.norm(points[66] - mid_of_eyes)

    ratio_hrz = (dst_cat_eyes / dst_hum_eyes)
    ratio_ver = (dst_cat_mouth / dst_hum_mouth)

    # RELOCATED LANDMARKS ACC TO RATIO
    points = points.transpose()
    points[0] = (points[0] * ratio_hrz).astype(int)
    points[1] = (points[1] * ratio_ver).astype(int)

    dif_x = points[0][66] - cat_points[2][0]
    dif_y = points[1][66] - cat_points[2][1]

    points[0] -= dif_x
    points[1] -= dif_y
    points = points.transpose()

    return points
# CROPS CAT IMAGE ACC TO DETECTED FACE
def crop_cat(image, frame, dimensions, bring):
    landmarks = []
    for i in range(1, int(frame[0]) + 1):
        landmarks.append([int(frame[2 * i - 1]), int(frame[2 * i])])

    points = np.array(landmarks).transpose()
    column_max, column_min = points[0].max(), points[0].min()
    row_max, row_min = points[1].max(), points[1].min()

    # DIMENSIONS OF CAT FACE BORDERS
    rectangle_width = column_max - column_min
    rectangle_height = row_max - row_min

    # DIMENSIONS OF THE CROPPING IMAGE
    width = rectangle_width + (rectangle_width % 2) + 10
    height = (width / 3) * 4

    width_p = int(width - rectangle_width)
    height_p = int(height - rectangle_height)

    # BOUNDS OF CROPPING IMAGE
    t = int(row_min - height_p/2)
    b = t + int(height)
    l = int(column_min - width_p/2)
    r = l + int(width)


    # IF THE LEFT OR RIGHT MOST SIDE IS OUT OF IMAGE
    if column_min <= 5 or column_max >= len(image[0]):
        rectangle_width = column_max or len(image[0]) - column_min

        l = 1 or int(column_min - width_p)
        r = l + int(width) or len(image[0])

    # IF THE TOP OR BOTTOM MOST SIDE IS OUT OF IMAGE
    if row_min <= 5 or row_max > len(image):
        rectangle_height = row_max or len(image) - row_min

        t = 1 or int(row_min - height_p)
        b = t + int(height) or len(image)


    # CROPPING IMAGE
    cropped = image[t:b, l:r]

    if bring == "landmarks":
        # RELOCATING LANDMARKS
        points[0] = points[0] - l
        points[1] = points[1] - t
        points = points.transpose()
        return (r-l), points

    # RESIZING IMAGE
    try:
        cropped = cv2.resize(cropped, dimensions)
    except:
        exit("CAT Image cannot be cropped as wanted. Please try another image.")
    return cropped
# BRINGS LANDMARKS OF CAT IMAGE
def cat_landmarks_bringer(image, frame, dimensions):
    width, landmarks = crop_cat(image, frame, dimensions, bring="landmarks")
    ratio = dimensions[0] / width
    landmarks = (landmarks * ratio).astype(int)

    return landmarks
# FITS CAT'S LANDMARKS ACC TO RESIZING
def fit_template_to_cat(landmarks):
    # CALCULATING SOME WANTED VALUES FOR PART 2 OR 3
    mid_of_eyes = ((landmarks[1] + landmarks[0])/2).astype(int)
    dst_cat_eyes = np.linalg.norm((landmarks[1] - landmarks[0]).astype(int))
    dst_cat_mouth = np.linalg.norm(landmarks[2] - mid_of_eyes)

    landmarks = pre_pro_temp_points(dst_cat_eyes, dst_cat_mouth, landmarks)

    return landmarks

# WARPS TRIANGLE
def warp_triangle(img_src, tri_src, tri_dst, dimensions):
    # GETTING AND APPLYING TRANSFORM MATRIX
    warp_mat = cv2.getAffineTransform(np.float32(tri_src), np.float32(tri_dst))
    img_dst = cv2.warpAffine(img_src, warp_mat, (dimensions[0], dimensions[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    return img_dst
# MORPHS TWO TRIANGLES
def morph_triangles(img_src, tris_src, img_dst, tris_dst, image, tris_mid, alpha):
    # BOUNDING RECTANGLES OF TRIANGLES
    rect_src = cv2.boundingRect(np.float32([tris_src]))
    rect_dst = cv2.boundingRect(np.float32([tris_dst]))
    rect_mid = cv2.boundingRect(np.float32([tris_mid]))
    (x1, y1, w1, h1) = rect_src
    (x2, y2, w2, h2) = rect_dst
    (x3, y3, w3, h3) = rect_mid

    # DETERMINING COORDINATES OF RECTANGLES' POINTS
    points_src = np.array(np.array(tris_src)[:] - [x1, y1])
    points_dst = np.array(np.array(tris_dst)[:] - [x2, y2])
    points_mid = np.array(np.array(tris_mid)[:] - [x3, y3])

    # CROPPING IMAGES TO RECTANGLE
    crpd_img_src = img_src[y1:y1 + h1, x1:x1 + w1]
    crpd_img_dst = img_dst[y2:y2 + h2, x2:x2 + w2]
    # WARPING IMAGES TO MID TRIANGLE
    warped_img_src = warp_triangle(crpd_img_src, points_src, points_mid, dimensions=(w3, h3))
    warped_img_dst = warp_triangle(crpd_img_dst, points_dst, points_mid, dimensions=(w3, h3))
    # BLENDING TWO WARPED IMAGES
    img_mid = cv2.addWeighted(warped_img_src, 1.0 - alpha, warped_img_dst, alpha, 0)

    # CREATING MASK AND APPLYING TO MID IMAGE
    mask_img_mid = np.zeros((h3, w3, 3), np.float32)
    cv2.fillConvexPoly(mask_img_mid, np.int32(points_mid), (1.0, 1.0, 1.0), 16, 0)

    # PASTE THE MID TRIANGLE TO THE BIG PICTURE
    image[y3:y3+h3, x3:x3+w3] = image[y3:y3+h3, x3:x3+w3] * (1 - mask_img_mid) + img_mid * mask_img_mid
# MORPHS TWO IMAGES
def morph_images(img_src, pts_src, img_dst, pts_dst, morph_step):
    # CALCULATING TRIANGLES
    trgs_src = simple_delaunay_triangles(img_src, pts_src)
    trgs_dst = referenced_delaunay_triangles(img_dst, pts_dst, pts_src, trgs_src)

    img_list = []
    for i in range(0, morph_step):
        image = np.zeros_like(img_src)

        # CALCULATING MID POINT
        pts_mid = [];
        alpha = i / (morph_step - 1)
        for j in range(0, len(pts_src)):
            x = (1 - alpha) * pts_src[j][0] + alpha * pts_dst[j][0]
            y = (1 - alpha) * pts_src[j][1] + alpha * pts_dst[j][1]
            pts_mid.append((x, y))
        # MID TRIANGLES BTW SOURCE AND DEST TRIANGLE
        trgs_mid = referenced_delaunay_triangles(image, pts_mid, pts_src, trgs_src)

        # MORPHING EACH TRIANGLE IN TRIANGLES
        for j in range(len(trgs_src)):
            morph_triangles(img_src, trgs_src[j], img_dst, trgs_dst[j], image, trgs_mid[j], alpha)

        reverted = image[:, :, ::-1]
        img_list.append(reverted)

    return img_list

# CREATING VIDEO CLIP
def video_maker(img_list, file_name, fps):
    clip = mpy.ImageSequenceClip(img_list, fps = fps)
    clip.write_videofile(path + file_name +".mp4", codec ='mpeg4')


def part_1():
    img_dmns = (360, 480)
    horizon_images = []
    for i in range(0, 2):
        # READ HUMAN IMAGES
        image = cv2.imread(path_of_humans + names_of_human_images[i])
        # CROPPING HUMAN IMAGE
        image = crop_human(image, dimensions=img_dmns)
        # DETECTING HUMAN FACE
        landmarks, rectangles = human_landmarks_bringer(image, i)

        # PUTTING LANDMARKS
        draw_landmarks_and_rect(image, landmarks, rectangles)
        # APPENDING TO FINAL HORIZONTAL IMAGE
        horizon_images.append(image)

    # READ CAT IMAGE
    image = cv2.imread(path_of_cats + names_of_cat_images[0])
    frame = np.genfromtxt(path_of_cats + names_of_cat_images[1], delimiter=' ')
    # DETECTING CAT FACE
    landmarks_cat = cat_landmarks_bringer(image, frame, dimensions=img_dmns)
    # CROPPING CAT IMAGE
    image = crop_cat(image, frame, dimensions=img_dmns, bring="image")

    # PUTTING LANDMARKS
    draw_landmarks_and_rect(image, landmarks_cat, None)
    # APPENDING TO FINAL HORIZONTAL IMAGE
    horizon_images.append(image)

    #RESULT IMAGE
    image = np.hstack((horizon_images))
    cv2.imwrite(path+"three_im.jpg", image)

def part_2():
    # READ CAT IMAGE
    image = cv2.imread(path_of_cats + names_of_cat_images[0])
    frame = np.genfromtxt(path_of_cats + names_of_cat_images[1], delimiter=' ')
    # DETECTING CAT FACE
    landmarks = cat_landmarks_bringer(image, frame, img_dmns)
    # CROPPING CAT IMAGE
    image = crop_cat(image, frame, dimensions=img_dmns, bring="image")

    # PUTTING LANDMARKS
    draw_landmarks_and_rect(image, landmarks, None)
    # FITTING TEMPLATE TO CAT
    landmarks = fit_template_to_cat(landmarks)
    # PUTTING LANDMARKS
    draw_landmarks_and_rect(image, landmarks, None)

    # RESULT IMAGE
    cv2.imwrite(path + "landmarked_cat.jpg", image)

def part_3():
    horizon_images = []
    for i in range(0, 2):
        # READ HUMAN IMAGES
        image = cv2.imread(path_of_humans + names_of_human_images[i])
        # CROPPING HUMAN IMAGE
        image = crop_human(image, dimensions=img_dmns)
        # DETECTING HUMAN FACE
        landmarks, rectangles = human_landmarks_bringer(image, i+1)

        # CALCULATING DELAUNAY TRIANGULATION
        triangles = simple_delaunay_triangles(image, landmarks)
        # DRAWING TRIANGLES
        image = draw_delaunay_triangles(image, triangles)
        # APPENDING TO FINAL HORIZONTAL IMAGE
        horizon_images.append(image)

    # READ CAT IMAGE
    image = cv2.imread(path_of_cats + names_of_cat_images[0])
    frame = np.genfromtxt(path_of_cats + names_of_cat_images[1], delimiter=' ')
    # DETECTING CAT FACE
    landmarks = cat_landmarks_bringer(image, frame, dimensions=img_dmns)
    # CROPPING CAT IMAGE
    image = crop_cat(image, frame, dimensions=img_dmns, bring="image")

    # FITTING TEMPLATE TO CAT
    landmarks = fit_template_to_cat(landmarks)
    # CALCULATING DELAUNAY TRIANGULATION
    triangles = simple_delaunay_triangles(image, landmarks)
    # DRAWING TRIANGLES
    image = draw_delaunay_triangles(image, triangles)

    # APPENDING TO FINAL HORIZONTAL IMAGE
    horizon_images.append(image)

    #RESULT IMAGE
    image = np.hstack((horizon_images))
    cv2.imwrite(path+"three_del.jpg", image)

def part_4():
    img_list = []
    # CALCULATION TRANSFORM MATRIX OF POINTS
    source_points = np.array([[200,20], [400,100], [200,300]]).astype(float)
    target_points = np.array([[400,300], [450,150], [600,200]])
    transform_points = (target_points - source_points)/20

    # CALCULATION TRANSFORM MATRIX OF COLOR
    source_color = np.array([30, 120, 240]).astype(float)
    target_color = np.array([240, 30, 120])
    transform_color = (target_color - source_color)/20


    for i in range(0,20):
        image = np.zeros((600,600,3), np.uint8)
        c = list(source_color)
        # DRAWING TRIANGLES
        cv2.polylines(image, [source_points.astype(int)], isClosed=True, color=c, thickness=1)
        cv2.fillPoly(image, [source_points.astype(int)], color=c)

        source_points += transform_points
        source_color += transform_color

        reverted = image[:, :, ::-1]
        img_list.append(reverted)

    # MAKING VIDEO
    video_maker(img_list, file_name="triangle", fps=20)

def part_5():
    img_dmns = (360, 480)
    img_list = []
    lndmrks_list = []
    # READ HUMAN IMAGES
    for i in range(0,2):
        image = cv2.imread(path_of_humans + names_of_human_images[i])
        # CROPPING HUMAN IMAGE
        image = crop_human(image, dimensions=img_dmns)
        # APPENDING TO IMAGES
        img_list.append(image)
        # HUMAN FACE DETECTING AND APPENDING IT TO LANDMARKS
        landmarks, junk = human_landmarks_bringer(image, i)
        lndmrks_list.append(landmarks)

    # READ CAT IMAGE
    image = cv2.imread(path_of_cats + names_of_cat_images[0])
    frame = np.genfromtxt(path_of_cats + names_of_cat_images[1], delimiter=' ')
    # DETECTING CAT FACE
    landmarks = cat_landmarks_bringer(image, frame, dimensions=img_dmns)
    # CROPPING CAT IMAGE
    image = crop_cat(image, frame, dimensions=img_dmns, bring="image")

    # FITTING TEMPLATE TO CAT AND APPENDING IT TO LANDMARKS
    landmarks = fit_template_to_cat(landmarks)
    lndmrks_list.append(landmarks)
    # APPENDING TO IMAGES
    img_list.append(image)

    for i in range(3):
        for j in range(3):
            if i != j:
                # MORPHING IMAGES
                images = morph_images(img_list[i], lndmrks_list[i], img_list[j], lndmrks_list[j], morph_step=19)
                # MAKING VIDEO
                video_maker(images, file_name="morphine_{}{}".format(i, j), fps=2)

# # # # # # # # # # # # # # # # # # #
#                                   #
# _____________ MAIN ______________ #
#                                   #
# # # # # # # # # # # # # # # # # # #

# NEEDED PATHS
path = "C:/Users/nurdi/Desktop/morphine/"
path_of_humans = path + "humans/"
path_of_cats = path + "cats/"

# TAKING ALL FILES IN DIRECTORY
names_of_human_images = [f for f in listdir(path_of_humans) if isfile(join(path_of_humans, f))]
names_of_cat_images = [f for f in listdir(path_of_cats) if isfile(join(path_of_cats, f))]

# RESULT IMAGE DIMENSIONS
img_dmns = (330, 440)

# part_1()
# part_2()
# part_3()
# part_4()
# part_5()
