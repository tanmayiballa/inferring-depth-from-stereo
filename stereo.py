import sys
from PIL import Image
import numpy as np
from datetime import datetime

MAX_DISPARITY = 50 # Set this to the maximum disparity in the image pairs you'll use
WINDOW_SIZE = 7 # Size of the block for naive stereo matching
MAX_NEIGHBOURS = 4
ITERATIONS = 20
ALPHA = 10000
## 0: Left, 1: Right, 2: Top, 3: Bottom: Indexes for the neighbours of a pixel.


## Reference: https://www.geeksforgeeks.org/python-pil-image-merge-method/#
## https://www.geeksforgeeks.org/python-pil-image-new-method/#
def displaying_3d(input_image, width, height):
    gray_scale = Image.new("L",(width,height))
    gray_scale.paste(input_image,(25,0)) ## Horizontal offset for the input image and overlaying it on the gray_scale img.
    image_3d = Image.merge("RGB",(gray_scale,input_image,input_image))
    image_3d.save("output-3d.png")
    return

## Reference: https://note.nkmk.me/en/python-pillow-add-margin-expand-canvas/
## Padding the image with all zeros, so that the corner and edge pixels won't be ignored during MRF and Naive.
def image_padding(img, pad_value):
    curr_width,curr_height = img.size
    new_height = curr_height + 2*pad_value
    new_width = curr_width + 2*pad_value
    padded_img = Image.new(img.mode, (new_width,new_height), 0)
    padded_img.paste(img, (pad_value,pad_value))
    return padded_img

## Removing the padding after processing the image.
def remove_padding(arr, pad_value):
    height, width, = arr.shape
    return arr[pad_value:height-pad_value, pad_value:width-pad_value]

def check_limits(x,y,w,h):
    left_limX = x - (int(WINDOW_SIZE/2) + MAX_DISPARITY)
    right_limX = x + (int(WINDOW_SIZE/2) + MAX_DISPARITY)
    left_limY = y - (int(WINDOW_SIZE / 2 )+ MAX_DISPARITY)
    right_limY = y + (int(WINDOW_SIZE / 2 )+ MAX_DISPARITY)

    if(left_limX<0 or left_limY<0 or right_limX>=h or right_limY>=w):
        return False
    return True

## Reference: https://numpy.org/doc/stable/reference/generated/numpy.fromfunction.html
## Filter to compute pair-wise disparity cost for MRF. It computes a quadratic cost.
def smooth_filter():
    r, c = MAX_DISPARITY,MAX_DISPARITY
    return np.fromfunction(lambda x,y:(x-y)**2, (r,c))

## Computing the data cost using a window-based approach.
def compute_disparity(pixel_X, pixel_Y, imW, imH,img1,img2):
    base_disparity_map = np.zeros(MAX_DISPARITY)
    if check_limits(pixel_X,pixel_Y,imW,imH) == False:
        return base_disparity_map
    for d in range(MAX_DISPARITY):
        img1_crop = img1[pixel_X - int(WINDOW_SIZE/2): pixel_X + int(WINDOW_SIZE/2), pixel_Y - int(WINDOW_SIZE/2): pixel_Y + int(WINDOW_SIZE/2)]
        img2_crop = img2[pixel_X - int(WINDOW_SIZE/2): pixel_X + int(WINDOW_SIZE/2),
                            pixel_Y - d -int(WINDOW_SIZE/2): pixel_Y -d +int(WINDOW_SIZE/2)]
        base_disparity_map[d] = np.sum((img2_crop - img1_crop)**2)
    return base_disparity_map

## Sending messages in all four directions.
## Referred pass_messages function in https://github.com/AustinCStone/StereoVisionMRF/blob/master/stereoVision.py.
def send_messages_to_neighbours(i,j,width,height,curr_msg_map,new_msgMap, pixel_disparity):
    for dir in range(MAX_NEIGHBOURS):
        msgN = np.sum(curr_msg_map[i][j], axis=0) - curr_msg_map[i][j][dir] ## Receiving messages from the neighbours.
        if (dir == 0 and j - 1 >= 0):
            new_msgMap[i][j - 1][1] = send_msg_individual(i, j, pixel_disparity, msgN)
        elif (dir == 1 and j + 1 < width):
            new_msgMap[i][j + 1][0] = send_msg_individual(i, j, pixel_disparity, msgN)
        elif (dir == 2 and i - 1 >= 0):
            new_msgMap[i - 1][j][3] = send_msg_individual(i, j, pixel_disparity, msgN)
        elif (dir == 3 and i + 1 < height):
            new_msgMap[i + 1][j][2] = send_msg_individual(i, j, pixel_disparity, msgN)
    return new_msgMap

## Computing and updating individual messages.
def send_msg_individual(x, y, pixel_disparity, msg_from_neighbours):
    base_msg = np.tile(pixel_disparity[x][y] + msg_from_neighbours, (MAX_DISPARITY, 1))
    pair_wise_disp = smooth_filter()
    v_func = ALPHA*pair_wise_disp
    compute_msg = base_msg + v_func
    final_msg = np.min(compute_msg, axis=1)
    return final_msg / np.sum(final_msg)

## This function computes the belief based on the final message map.
def compute_belief(pixel_disparity,msgMap,h,w):
    output_belief = np.zeros((h,w))
    for i in range(h):
        for j in range(w):
            min_disp = 0
            min_disparity_val = 10000007.0
            connected_pixel_cost = np.sum(msgMap[i][j], axis = 0)
            curr_disp = connected_pixel_cost + pixel_disparity[i][j]
            output_belief[i][j] = np.argmin(curr_disp)
    return output_belief

## Main function that implements the loopy belief algorithm.
def mrf_stereo(img1, img2, disp_costs):
    result = np.zeros((img1.shape[0], img1.shape[1]))
    height, width,  = img1.shape
    curr_msg_map = np.ones((height,width,MAX_NEIGHBOURS,MAX_DISPARITY)) ## Initializing all messages to ones
    new_msgMap = np.ones((height, width, MAX_NEIGHBOURS, MAX_DISPARITY))
    pixel_disparity = disp_costs
    for iter in range(ITERATIONS):
        print("Iteration: ", iter)
        for i in range(height):
            for j in range(width):
                new_msgMap = send_messages_to_neighbours(i,j,width,height,curr_msg_map,new_msgMap,pixel_disparity)
        print("Difference between message maps:", np.sum(abs(curr_msg_map - new_msgMap)))
        curr_msg_map = new_msgMap.copy()
    final_msg_map = new_msgMap.copy()
    print("Computing Belief")
    resultant_disparity_map = compute_belief(pixel_disparity,final_msg_map,height,width)
    return resultant_disparity_map

# This function should compute the function D() in the assignment
def disparity_costs(img1, img2):
    result = np.zeros((img1.shape[0], img1.shape[1], MAX_DISPARITY))
    w = img1.shape[1]
    h = img1.shape[0]
    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
                result[i,j] = compute_disparity(i,j,w,h,img1,img2)
    return result

# This function finds the minimum cost at each pixel
def naive_stereo(img1, img2, disp_costs):
    return np.argmin(disp_costs, axis=2)

if __name__ == "__main__":
    if len(sys.argv) != 3 and len(sys.argv) != 4:
        raise Exception("usage: " + sys.argv[0] + " image_file1 image_file2 [gt_file]")
    input_filename1, input_filename2 = sys.argv[1], sys.argv[2]

    resize_factor = 2
    # read in images and gt
    pad_value = MAX_DISPARITY+1
    img1 = Image.open(input_filename1).convert('L')
    orig_w, orig_h = img1.size
    w, h = img1.size
    resized_img1 = img1.resize((w//resize_factor,h//resize_factor)) ## Resizing the original image.
    image1 = np.array(image_padding(resized_img1,pad_value)) ## Padding the image to prevent missing of border pixels.
    img2 = Image.open(input_filename2).convert('L')
    w, h = img2.size
    resized_img2 = img2.resize((w//resize_factor, h//resize_factor))
    image2 = np.array(image_padding(resized_img2, pad_value))

    gt = None
    if len(sys.argv) == 4:
        gt = np.array(Image.open(sys.argv[3]))[:,:,0]

        # gt maps are scaled by a factor of 3, undo this...
        gt = gt / 3.0

    # compute the disparity costs (function D_2())
    start_naive = datetime.now()
    print("Computing Disparity using Naive")
    disp_costs = disparity_costs(image1, image2)

    # do stereo using naive technique
    disp1_tmp = naive_stereo(image1, image2, disp_costs)
    end_naive = datetime.now()
    print("Time taken for naive: ", end_naive-start_naive)
    disp1 = remove_padding(disp1_tmp,pad_value)
    ## This is the original disparity image without any scaling.
    res_imgNO = Image.fromarray(disp1.astype(np.uint8))
    #res_imgNO.save("out_naive.png")
    (res_imgNO.resize((orig_w, orig_h))).save("out_naive.png")
    disp1 = (disp1 * 255.0) / MAX_DISPARITY
    ## This is the scaled disparity image for better visualization.
    res_imgNS = Image.fromarray(disp1.astype(np.uint8))
    #res_imgNS.save("output-naive.png")
    (res_imgNS.resize((orig_w, orig_h))).save("output-naive.png")

    # do stereo using mrf
    start_mrf = datetime.now()
    print("Computing Disparity using MRF")
    disp3_tmp = mrf_stereo(image1, image2, disp_costs)
    end_mrf = datetime.now()
    print("Time taken for mrf: ", end_mrf - start_mrf)
    disp3 = remove_padding(disp3_tmp, pad_value)
    ## This is the original disparity image without any scaling.
    res_imgMO = Image.fromarray(disp3.astype(np.uint8))
    #res_imgMO.save("out_mrf.png")
    (res_imgMO.resize((orig_w, orig_h))).save("out_mrf.png")
    disp3 = (disp3 * 255.0) / MAX_DISPARITY
    ## This is the scaled disparity image for better visualization.
    res_imgMS = Image.fromarray(disp3.astype(np.uint8))
    #res_imgMS.save("output-mrf.png")
    (res_imgMS.resize((orig_w, orig_h))).save("output-mrf.png")
    
    ## Uncomment these lines for 3-d output:
    #image_for_3d = Image.open("output-mrf.png")
    #displaying_3d(image_for_3d,orig_w,orig_h)

    disp_img1 = np.array(Image.open('out_naive.png').convert('L'))
    disp_img2 = np.array(Image.open('out_mrf.png').convert('L'))

    # Measure error with respect to ground truth, if we have it...
    if gt is not None:
        err = np.sum((disp_img1- gt)**2)/gt.shape[0]/gt.shape[1]
        print("Naive stereo technique mean error = " + str(err))

        err = np.sum((disp_img2- gt)**2)/gt.shape[0]/gt.shape[1]
        print("MRF stereo technique mean error = " + str(err))




