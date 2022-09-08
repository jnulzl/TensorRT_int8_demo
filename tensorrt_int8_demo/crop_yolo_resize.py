# OpenCV implementation of crop face && resize to (des_width, des_height) using affine transform

import numpy as np
import cv2


def crop_yolo_resized_with_affine_transform(img_path, roi_xyxy, des_width, des_height, isFixResize=False):
    src_rgb = cv2.imread(img_path)

    '''
    face roi 
    (x0,y0)------------(x1,y1)
       |                  |
       |                  |
       |                  |
       |                  |
       |                  |
       |                  |
       |                  |
       |                  |
    (-,-)------------(x2, y2)
    '''    
    src_points = [[roi_xyxy[0], roi_xyxy[1]], [roi_xyxy[2], roi_xyxy[1]], [roi_xyxy[2], roi_xyxy[3]]]
    src_points = np.array(src_points, dtype=np.float32)
    
    if isFixResize:
        src_width = roi_xyxy[2] - roi_xyxy[0]
        src_height = roi_xyxy[3] - roi_xyxy[1]
        
        scale_x = des_width / src_width
        scale_y = des_height / src_height
        scale = scale_x if scale_x < scale_y else scale_y
               
        src_new_width  = src_width * scale
        src_new_height = src_height * scale
        x = (des_width - src_new_width) / 2
        y = (des_height - src_new_height) / 2
        
        des_points = [[x, y], [x + src_new_width, y], [x + src_new_width, y + src_new_height]]        
        # print(des_points)
    else:
        des_points = [[0, 0], [des_width, 0], [des_width, des_height]]
    
    des_points = np.array(des_points, dtype=np.float32)

    M = cv2.getAffineTransform(src_points, des_points)
    # print(M)
    crop_and_yolo_resized_with_affine_transform = cv2.warpAffine(src_rgb, M, (des_width, des_height))

    return crop_and_yolo_resized_with_affine_transform
    

def test():

    '''
    Source image from 
    https://www.whitehouse.gov/wp-content/uploads/2021/04/P20210303AS-1901.jpg
    or
    https://en.wikipedia.org/wiki/Joe_Biden#/media/File:Joe_Biden_presidential_portrait.jpg
    '''
    img_path = "Joe_Biden_presidential_portrait.jpg"
    # xmin ymin xmax ymax
    # roi_xyxy = [372, 132, 837, 760]
    roi_xyxy = [0, 0, 1200, 1500]
    
    des_width = 512
    des_height = 512
    
    # des_width = 200
    # des_height = 800
    
    # des_width = 800
    # des_height = 200
    
    # des_width = 1024+512
    # des_height = 1024+512
    
    
    # des_width = 1024+256
    # des_height = 1024+512
    
    # des_width = 1024+512
    # des_height = 1024+256
    isFixResize = True
    crop_and_yolo_resized_with_affine_transform = crop_yolo_resized_with_affine_transform(img_path, roi_xyxy , des_width, des_height, isFixResize)
    cv2.imshow("crop_and_yolo_resized_with_affine_transform", crop_and_yolo_resized_with_affine_transform)
    cv2.imwrite("crop_and_yolo_resized_with_affine_transform.jpg", crop_and_yolo_resized_with_affine_transform)
    cv2.waitKey(0)
    

def yolo_face_quant_data(img_lis_file):
    isFixResize = True
    des_width = 640
    des_height = 640
    all_imgs = []
    with open(img_lis_file, 'r') as fpR:
        imgs = fpR.readlines()
        for index,img_path in enumerate(imgs):
            img_path = img_path.strip()
            img = cv2.imread(img_path)
            img_height, img_width = img.shape[:2]
            print(index, img_path, img_height, img_width)
            roi_xyxy = [0, 0, img_width, img_height]
            img_pre = crop_yolo_resized_with_affine_transform(img_path, roi_xyxy , 
                                              des_width, des_height, isFixResize)
            img_pre = img_pre[:,:,::-1] / 255.0
            all_imgs.append(img_pre.transpose((2, 0, 1)))
    all_imgs = np.array(all_imgs, dtype=np.float32)
    print("all_imgs : ", all_imgs.shape)
    all_imgs.tofile("yolo_face_preprocessed_bin_file_input_size%d_num%d.bin"%(des_width, all_imgs.shape[0]))
            

if __name__ == "__main__":
    img_lis_file = "img_list.txt"
    yolo_face_quant_data(img_lis_file)

