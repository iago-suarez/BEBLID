![Graffter Banner](imgs/banner.jpg)
# BEBLID: Boosted Efficient Binary Local Image Descriptor
This repository contains the source code of the BEBLID local image descriptor


![BEBLID Matching Result](imgs/inliners_img.jpg)

## Dependencies

The code depends on OpenCV 4. To install OpenCV in Ubuntu 18.04 compile it from sources with the following instructions:

```shell script
# Install dependencies (Ubuntu 18.04)
sudo apt-get install -y build-essential cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
# Download source code
git clone https://github.com/opencv/opencv.git --branch 4.1.0 --depth 1
# Create build directory
cd opencv && mkdir build && cd build
# Generate makefiles, compile and install
cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local ..
make -j$(nproc)
sudo make install
```

## Compile and Run

With the BEBLID descriptor code we provide a small demo to register a pair of images. 
The demo detects feature points using ORB detector (FAST + Harris score) and describes using BEBLID. 

The code can be compiled with Cmake:

```shell script
mkdir build && cd build
cmake .. && make
./beblid_demo
```

The result for the provided images should be several imshows and something like this in the standard output:

```
Detected  500 kps in image1
Detected  500 kps in image2
Points described
Number of matches: 228
Number of inliers 181 ( 79.386% )

Process finished with exit code 0
```

## References

If you use this code, please cite our Pattern Recognition Letters paper:
```bibtex
@article{SUAREZ2020,
title = "BEBLID: Boosted Efficient Binary Local Image Descriptor",
journal = "Pattern Recognition Letters",
year = "2020",
issn = "0167-8655",
doi = "https://doi.org/10.1016/j.patrec.2020.04.005",
url = "http://www.sciencedirect.com/science/article/pii/S0167865520301252",
author = "Iago Suárez and Ghesn Sfeir and José M. Buenaposada and Luis Baumela",
keywords = "Local image descriptors, Binary descriptors, Real-time, Efficient matching, Boosting",
abstract = "Efficient matching of local image features is a fundamental task in many computer vision applications. However, the real-time performance of top matching algorithms is compromised in computationally limited devices, such as mobile phones or drones, due to the simplicity of their hardware and their finite energy supply. In this paper we introduce BEBLID, an efficient learned binary image descriptor. It improves our previous real-valued descriptor, BELID, making it both more efficient for matching and more accurate. To this end we use AdaBoost with an improved weak-learner training scheme that produces better local descriptions. Further, we binarize our descriptor by forcing all weak-learners to have the same weight in the strong learner combination and train it in an unbalanced data set to address the asymmetries arising in matching and retrieval tasks. In our experiments BEBLID achieves an accuracy close to SIFT and better computational efficiency than ORB, the fastest algorithm in the literature."
}
```
## Contact and Licence
We provide a free pre-trained version of the execution code. Full execution and training code can be obtained under license, if you are interested please contact us:

* Iago Suárez ( iago.suarez@thegraffter.com ) for technical issues.
* Miguel Ángel Orellana Sainz ( miguel.orellana@thegraffter.com ) for commercial issues.

This software was developed by [The Graffter S.L.](http://www.thegraffter.com) in collaboration with the [PCR lab of the Universidad Politécnica de Madrid](http://www.dia.fi.upm.es/~pcr/research.html).
