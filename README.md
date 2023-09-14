## Inferring depth from stereo.

### Input images:
| view1 | view5 |
| - | - |
|![alt text](https://github.com/tanmayiballa/inferring-depth-from-stereo/blob/main/Input/view1.png)| ![alt text](https://github.com/tanmayiballa/inferring-depth-from-stereo/blob/main/Input/view5.png) |

### Pre-processing:

**Border-pixel padding:**

Since the edge and corner pixels are ignored while computing the disparity costs of the images using a fixed-size window, we have padded the input images with the maximum disparity values. This helps in efficiently computing the disparity costs for the corder and edge pixels in the image.

This is a sample image after padding.

**Resizing the image:**

Owing to the time complexity, we have down-sized the input images by a factor of 2, and have upsampled the disparity map later on. However, we have provided result images obtained without resizing.

### Algorithms for inferring depth.

#### Naive Method:
- In this technique, we have considered a window of size 7 and computed the disparity by moving the window through the image, and calculated the abosolute mean squared difference between the 2 image patches.
- The image patches of the left and right images are translated by d distance, where d is the disparity value.
- For each pixel in the image, the costs for each disparity values are considered and the value with minimum cost is assigned as the final disparity of the pixel.
- The maximum disparity considered for the algorithm is 50.

**Results:**

The mean squared error value as compared to the ground truth is: 

The disparity image and the scaled depth image for the given input are shown below.

| Naive_disparity | Naive_depth |
| - | - |
| ![alt text](https://github.com/tanmayiballa/inferring-depth-from-stereo/blob/main/out_naive.png) | ![alt text](https://github.com/tanmayiballa/inferring-depth-from-stereo/blob/main/output-naive.png) |

The disparity image and the scaled depth image for the resized input are shown below.
| Naive_disparity_resized | Naive_depth_resized |
| - | - |
|![alt text](https://github.com/tanmayiballa/inferring-depth-from-stereo/blob/main/out_naive_resized.png) | ![alt text](https://github.com/tanmayiballa/inferring-depth-from-stereo/blob/main/output-naive-resized.png) |

As we can see from the results, the algorithm is able to differentiate between the background and the foreground properly. The disparity map also looks decent. However, there is some noise in the image and we probably need different techniques to treat the same.

The time taken for the naive method is approximately 1 minute if the image is not scaled, and under 20 seconds if the image is scaled.

#### Loopy Belief Propagation (LBP):
- LBP is a message passing algorithm, we consider the belief of all the neighbour nodes about the disparity value of the current pixel, and mimize the energy function, which depends on the connected pixels cost, current pixel disparity, and pair-wise disparity cost (this is based on the neighbour's disparity).
- We have experimented with 20 iterations for updating the message maps, if the images are not downsampled; and 50 iterations if the images are downsampled by a factor of 2. 
- The maximum disparity value considered is 50.
- Alpha value, that is used as a smoothening parameter for the pair-wise disparity is kept to a higher value of 10000 to remove the noise obtained in the naive approach.
- We have used the quadratic distance for the pair-wise distance function. We have tried Potts model, but found the quadratic distance function more efficient.

##### Results:

The mean squared error value as compared to the ground truth is: 

The mean squared error has slightly decreased as compared to the naive approach. 

The disparity image and the scaled depth image for the given input are shown below.

| MRF_disparity | MRF_depth |
| - | - |
| ![alt text](https://github.com/tanmayiballa/inferring-depth-from-stereo/blob/main/out_mrf.png) | ![alt text](https://github.com/tanmayiballa/inferring-depth-from-stereo/blob/main/output-mrf.png) |

| MRF_disparity_resized | MRF_depth_resized |
| - | - |
| ![alt text](https://github.com/tanmayiballa/inferring-depth-from-stereo/blob/main/out_mrf_resized.png) | ![alt text](https://github.com/tanmayiballa/inferring-depth-from-stereo/blob/main/output-mrf-resized.png) |

Though the error has decreased, there is no evident visual difference from the output images. The algorithm might require more number of iterations to further converge to the optimum.

### Experimental Results:

| Input | Downsampled by 2 ? | Iterations for MRF | Error_naive | Error MRF |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Aloe  | NO  | 20 | 79.28 | 65.83 |
| Aloe  | YES  | 50 | 208 | 189.97 |
| Baby  | NO  | 20 | 104.3 | 93.31 |
| Baby  | YES  | 50 | 293.3 | 276.6 |
| Flowerpots | NO  | 20 | 440.41 | 411.35 |
| Flowerpots | YES  | 50 | 707.7 | 660.9 |

- From the experiments, it is clear that the MRF performed well as compared to the Naive technique by reducing the mean squared error.
- However, we might need to allow the MRF to run for more number of iterations to converge and provide better results.
- The error values for the Flowerpots images are higher as compared to the other inputs. Probable noise in the image might effect the accuracy of the algorithm.
- Also, we have used the Max-Sum energy function for the loopy belief algorithm. Using some other techniques such as Max-Product, might improve the accuracy further.
- The window size and disparity values can be considered as hyperparameters and tuned accordingly to obtain better efficiency.

## 3D output

We have generated a 3D version of the output for green and cyan glasses.

![alt text](https://github.com/tanmayiballa/inferring-depth-from-stereo/blob/main/output-3d.png)

We have translated the image horizontally and considered it as a red channel, and merged it with the actual image in green and blue channels for a cyan color effect. 
