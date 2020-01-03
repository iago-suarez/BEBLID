/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2020, The Graffter S.L. All rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef BEBLID_DESCRIPTOR_H_
#define BEBLID_DESCRIPTOR_H_

#include <opencv2/opencv.hpp>

struct BelidData {
  BelidData() = default;

  BelidData(float shrinkage,
            const cv::Size &patchSize,
            int nWLs,
            int minBoxRadius,
            int maxBoxRadius,
            std::vector<int> x1s,
            std::vector<int> y1s,
            std::vector<int> x2s,
            std::vector<int> y2s,
            std::vector<int> boxSize,
            std::vector<float> alpha
  ) : shrinkage(shrinkage),
      patchSize(patchSize),
      nWLs(nWLs),
      x1s(std::move(x1s)),
      y1s(std::move(y1s)),
      x2s(std::move(x2s)),
      y2s(std::move(y2s)),
      boxSize(std::move(boxSize)),
      alphas(std::move(alpha)) {}

  bool operator==(const BelidData &rhs) const {
    return shrinkage == rhs.shrinkage &&
        patchSize == rhs.patchSize &&
        nWLs == rhs.nWLs &&
        x1s == rhs.x1s &&
        y1s == rhs.y1s &&
        x2s == rhs.x2s &&
        y2s == rhs.y2s &&
        boxSize == rhs.boxSize &&
        alphas == rhs.alphas &&
        thresholds == rhs.thresholds;
  }

  bool operator!=(const BelidData &rhs) const {
    return !(rhs == *this);
  }

  // The learning rate of the Adaboost learning process
  float shrinkage;

  // The size of the patch used in training
  cv::Size patchSize;

  // The number of weak learners trained in the ensemble
  int nWLs;

  // The number of output dimensions of the descriptor
  int nDims;

  // Matrix containing the x1, y1, x2 and y2 values of every weak learner
  std::vector<int> x1s;
  std::vector<int> y1s;
  std::vector<int> x2s;
  std::vector<int> y2s;
  std::vector<int> boxSize;

  // Matrix containing the weak learner weights set when training the Adaboost ensemble
  std::vector<float> alphas;

  // Matrix containing the weak learner thresholds
  std::vector<float> thresholds;
};

struct ABWLParams {
  int x1, x2, y1, y2, boxRadius;
};

/**
 * Class implementation of BEBLID (Boosted Efficient Binary Local Image Descriptor).
 * The algorithm is a fast binary descriptor with an accuracy similar to SIFT but much faster.
 * This C++ Implementation contain 2 pre-trained versions of our descriptor BEBLID-512 and BEBLID-256.
 * It can be used with any feature detector, the algorithms were trained in the
 * Liberty dataset that contains patches detected with Difference of Gaussians (DoG).
 * To re-train the method with other data sets, please contact the authors.
 */
class BEBLID : public cv::Feature2D {
 public:
  /** @brief Creates the BEBLID descriptor.

  @param n_wls The number of final weak-learners in the descriptor. It must be a multiple of 8 such as 256 or 512.
  @param scale_factor Adjusts the sampling window of detected keypoints
  6.25f is default and fits for KAZE, SURF detected keypoints window ratio
  6.75f should be the scale for SIFT detected keypoints window ratio
  5.00f should be the scale for AKAZE, MSD, AGAST, FAST, BRISK keypoints window ratio
  0.75f should be the scale for ORB keypoints ratio
  1.50f was the default in original implementation
  */
  BEBLID(int n_wls = 512, float scale_factor = 1);

  /** @brief Computes the descriptors for a set of keypoints detected in an image (first variant) or image set
  (second variant).

  @param image Image.
  @param keypoints Input collection of keypoints. Keypoints for which a descriptor cannot be
  computed are removed. Sometimes new keypoints can be added, for example: SIFT duplicates keypoint
  with several dominant orientations (for each orientation).
  @param descriptors Computed descriptors. In the second variant of the method descriptors[i] are
  descriptors computed for a keypoints[i]. Row j is the keypoints (or keypoints[i]) is the
  descriptor for keypoint j-th keypoint.
   */
  CV_WRAP void compute(cv::InputArray image,
                       CV_OUT CV_IN_OUT std::vector<cv::KeyPoint> &keypoints,
                       cv::OutputArray descriptors) override;

  CV_WRAP int descriptorSize() const override;
  CV_WRAP int descriptorType() const override;
  CV_WRAP int defaultNorm() const override;

  //! Return true if detector object is empty
  CV_WRAP bool empty() const CV_OVERRIDE;
  CV_WRAP cv::String getDefaultName() const CV_OVERRIDE;

  /** @brief Creates the BEBLID descriptor.

  @param n_wls The number of final weak-learners in the descriptor. It must be a multiple of 8 such as 256 or 512.
  @param scale_factor Adjusts the sampling window of detected keypoints
  6.25f is default and fits for KAZE, SURF detected keypoints window ratio
  6.75f should be the scale for SIFT detected keypoints window ratio
  5.00f should be the scale for AKAZE, MSD, AGAST, FAST, BRISK keypoints window ratio
  0.75f should be the scale for ORB keypoints ratio
  1.50f was the default in original implementation
  */
  static cv::Ptr<BEBLID>
  create(int n_wls = 512, float scale_factor = 1);

 private:
  void computeBEBLID(const cv::Mat &integralImg,
                     const std::vector<cv::KeyPoint> &keypoints,
                     cv::Mat &descriptors);
  BelidData params_;
  std::vector<ABWLParams> wl_params_;
  float scale_factor_ = 1;
};

#endif //BEBLID_DESCRIPTOR_H_
