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
#include "BEBLID.h"

#define UPM_BEBLID_PARALLEL

#define UPM_ROUNDNUM(x) ((int)(x + 0.5f))
// An extra margin we leave in case of the keypoint is in the margin of the image

#define UPM_DEGREES_TO_RADS 0.017453292519943295 // (M_PI / 180.0)

#define UPM_BELID_EXTRA_RATIO_MARGIN 1.75

BelidData getPrecomBEBLIDData256() {
  return BelidData();
}

BelidData getPrecomBEBLIDData512() {
  return BelidData();
}

inline BelidData getBEBLIDPreTrainedData(int n_wls) {
  assert(n_wls > 0 && n_wls <= 512);
  if (n_wls <= 256) {
    return getPrecomBEBLIDData256();
  } else {
    return getPrecomBEBLIDData512();
  }
}

inline bool isKeypointInTheBorder(const cv::KeyPoint &kp,
                                  const cv::Size &imgSize,
                                  const cv::Size &patchSize = {32, 32},
                                  float scaleFactor = 1) {
  // This would be the correct measure but since we will compare with half of the size, use this as border size
  float s = scaleFactor * kp.size / (patchSize.width + patchSize.height);
  cv::Size2f border(patchSize.width * s * UPM_BELID_EXTRA_RATIO_MARGIN,
                    patchSize.height * s * UPM_BELID_EXTRA_RATIO_MARGIN);

  if (kp.pt.x < border.width || kp.pt.x + border.width >= imgSize.width) return true;
  if (kp.pt.y < border.height || kp.pt.y + border.height >= imgSize.height) return true;
  return false;
}

/**
 * @brief Rectifies the coordinates of the weak learners that conform the descriptor
 * @param wlPatchParams
 * @param wlImageParams
 * @param kp
 * @param scaleFactor
 * @param patchSize
 */
inline void rectifyABWL(const std::vector<ABWLParams> &wlPatchParams,
                        std::vector<ABWLParams> &wlImageParams,
                        const cv::KeyPoint &kp,
                        float scaleFactor = 1,
                        const cv::Size &patchSize = cv::Size(32, 32)) {
  float m00, m01, m02, m10, m11, m12;
  float s, cosine, sine;
  int i;

  s = scaleFactor * kp.size / (0.5f * (patchSize.width + patchSize.height));
  wlImageParams.resize(wlPatchParams.size());

  if (kp.angle == -1) {
    m00 = s;
    m01 = 0.0f;
    m02 = -0.5 * s * patchSize.width + kp.pt.x;
    m10 = 0.0f;
    m11 = s;
    m12 = -s * 0.5f * patchSize.height + kp.pt.y;
  } else {
    cosine = (kp.angle >= 0) ? cos(kp.angle * UPM_DEGREES_TO_RADS) : 1.f;
    sine = (kp.angle >= 0) ? sin(kp.angle * UPM_DEGREES_TO_RADS) : 0.f;

    m00 = s * cosine;
    m01 = -s * sine;
    m02 = (-s * cosine + s * sine) * patchSize.width * 0.5f + kp.pt.x;
    m10 = s * sine;
    m11 = s * cosine;
    m12 = (-s * sine - s * cosine) * patchSize.height * 0.5f + kp.pt.y;
  }

  for (i = 0; i < wlPatchParams.size(); i++) {
    wlImageParams[i].x1 = UPM_ROUNDNUM(m00 * wlPatchParams[i].x1 + m01 * wlPatchParams[i].y1 + m02);
    wlImageParams[i].y1 = UPM_ROUNDNUM(m10 * wlPatchParams[i].x1 + m11 * wlPatchParams[i].y1 + m12);
    wlImageParams[i].x2 = UPM_ROUNDNUM(m00 * wlPatchParams[i].x2 + m01 * wlPatchParams[i].y2 + m02);
    wlImageParams[i].y2 = UPM_ROUNDNUM(m10 * wlPatchParams[i].x2 + m11 * wlPatchParams[i].y2 + m12);
    wlImageParams[i].boxRadius = UPM_ROUNDNUM(s * wlPatchParams[i].boxRadius);
  }

  // LOGD << "H:\n" << m00 << ", " << m01 << ", " << m02 << "\n" << m10 << ", " << m11 << ", " << m12;
}

/**
 * @brief Computes the WL resoponse for a certain weak learner
 * @param wlImageParams
 * @param integralImage
 * @return
 */
inline float computeABWLResponse(const ABWLParams &wlImageParams, const cv::Mat &integralImage) {
  assert(!integralImage.empty());

  int frameWidth, frameHeight, box1x1, box1y1, box1x2, box1y2, box2x1, box2y1, box2x2, box2y2;
  int idx1, idx2, idx3, idx4;
  int A, B, C, D;
  const int *ptr;
  int box_area1, box_area2, width;
  float sum1, sum2, average1, average2;
  // Since the integral image has one extra row and col, calculate the patch dimensions
  frameWidth = integralImage.cols;
  frameHeight = integralImage.rows;

  // For the first box, we calculate its margin coordinates
  box1x1 = wlImageParams.x1 - wlImageParams.boxRadius;
  if (box1x1 < 0) box1x1 = 0;
  else if (box1x1 >= frameWidth - 1) box1x1 = frameWidth - 2;
  box1y1 = wlImageParams.y1 - wlImageParams.boxRadius;
  if (box1y1 < 0) box1y1 = 0;
  else if (box1y1 >= frameHeight - 1) box1y1 = frameHeight - 2;
  box1x2 = wlImageParams.x1 + wlImageParams.boxRadius + 1;
  if (box1x2 <= 0) box1x2 = 1;
  else if (box1x2 >= frameWidth) box1x2 = frameWidth - 1;
  box1y2 = wlImageParams.y1 + wlImageParams.boxRadius + 1;
  if (box1y2 <= 0) box1y2 = 1;
  else if (box1y2 >= frameHeight) box1y2 = frameHeight - 1;
  assert((box1x1 < box1x2 && box1y1 < box1y2) && "Box 1 has size 0");

  // For the second box, we calculate its margin coordinates
  box2x1 = wlImageParams.x2 - wlImageParams.boxRadius;
  if (box2x1 < 0) box2x1 = 0;
  else if (box2x1 >= frameWidth - 1) box2x1 = frameWidth - 2;
  box2y1 = wlImageParams.y2 - wlImageParams.boxRadius;
  if (box2y1 < 0) box2y1 = 0;
  else if (box2y1 >= frameHeight - 1) box2y1 = frameHeight - 2;
  box2x2 = wlImageParams.x2 + wlImageParams.boxRadius + 1;
  if (box2x2 <= 0) box2x2 = 1;
  else if (box2x2 >= frameWidth) box2x2 = frameWidth - 1;
  box2y2 = wlImageParams.y2 + wlImageParams.boxRadius + 1;
  if (box2y2 <= 0) box2y2 = 1;
  else if (box2y2 >= frameHeight) box2y2 = frameHeight - 1;
  assert((box2x1 < box2x2 && box2y1 < box2y2) && "Box 2 has size 0");

  // Calculate the indices on the integral image where the box falls
  width = integralImage.cols;
  idx1 = box1y1 * width + box1x1;
  idx2 = box1y1 * width + box1x2;
  idx3 = box1y2 * width + box1x1;
  idx4 = box1y2 * width + box1x2;
  assert(idx1 >= 0 && idx1 < integralImage.size().area());
  assert(idx2 >= 0 && idx2 < integralImage.size().area());
  assert(idx3 >= 0 && idx3 < integralImage.size().area());
  assert(idx4 >= 0 && idx4 < integralImage.size().area());
  ptr = integralImage.ptr<int>();

  // Read the integral image values for the first box
  A = ptr[idx1];
  B = ptr[idx2];
  C = ptr[idx3];
  D = ptr[idx4];

  // Calculate the mean intensity value of the pixels in the box
  sum1 = A + D - B - C;
  box_area1 = (box1y2 - box1y1) * (box1x2 - box1x1);
  assert(box_area1 > 0);
  average1 = sum1 / box_area1;

  // Calculate the indices on the integral image where the box falls
  idx1 = box2y1 * width + box2x1;
  idx2 = box2y1 * width + box2x2;
  idx3 = box2y2 * width + box2x1;
  idx4 = box2y2 * width + box2x2;

  assert(idx1 >= 0 && idx1 < integralImage.size().area());
  assert(idx2 >= 0 && idx2 < integralImage.size().area());
  assert(idx3 >= 0 && idx3 < integralImage.size().area());
  assert(idx4 >= 0 && idx4 < integralImage.size().area());

  // Read the integral image values for the first box
  A = ptr[idx1];
  B = ptr[idx2];
  C = ptr[idx3];
  D = ptr[idx4];

  // Calculate the mean intensity value of the pixels in the box
  sum2 = A + D - B - C;
  box_area2 = (box2y2 - box2y1) * (box2x2 - box2x1);
  assert(box_area2 > 0);
  average2 = sum2 / box_area2;

  return average1 - average2;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////// BEBLID Class ////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

BEBLID::BEBLID(int n_wls, float scale_factor) :
    scale_factor_(scale_factor) {

  assert(n_wls % 8 == 0);
  params_ = getBEBLIDPreTrainedData(n_wls);
  params_.nWLs = n_wls;
  // Populate wl_params_
  wl_params_.resize(params_.nWLs);
  for (int i = 0; i < params_.nWLs; i++) {
    wl_params_[i].x1 = params_.x1s[i];
    wl_params_[i].y1 = params_.y1s[i];
    wl_params_[i].x2 = params_.x2s[i];
    wl_params_[i].y2 = params_.y2s[i];
    wl_params_[i].boxRadius = (params_.boxSize[i] - 1) / 2;
  }
  params_.nDims = params_.nDims;
}

void BEBLID::compute(const cv::_InputArray &image,
                     std::vector<cv::KeyPoint> &keypoints,
                     const cv::_OutputArray &_descriptors) {
  assert(image.type() == CV_8UC1);
  cv::Mat integralImg;

  // compute the integral image
  cv::integral(image, integralImg);

  // Create the output array of descriptors
  _descriptors.create((int) keypoints.size(), descriptorSize(), descriptorType());

  // descriptor storage
  cv::Mat descriptors = _descriptors.getMat();
  assert(descriptors.type() == CV_8UC1);

  // Compute the BEBLID descriptors
  computeBEBLID(integralImg, keypoints, descriptors);
}

int BEBLID::descriptorSize() const {
  return params_.nDims / 8;
}

int BEBLID::descriptorType() const {
  return CV_8UC1;
}
int BEBLID::defaultNorm() const {
  return cv::NORM_HAMMING;
}

bool BEBLID::empty() const {
  // Set to false because we don't care at the moment
  return false;
}

cv::String BEBLID::getDefaultName() const {
  return std::string("BEBLID") + std::to_string(params_.nDims);
}

cv::Ptr<BEBLID> BEBLID::create(int n_wls, float scale_factor) {
  return cv::makePtr<BEBLID>(n_wls, scale_factor);
}

void BEBLID::computeBEBLID(const cv::Mat &integralImg,
                           const std::vector<cv::KeyPoint> &keypoints,
                           cv::Mat &descriptors) {
  assert(!integralImg.empty());
  assert(descriptors.rows == keypoints.size());

  const int *integralPtr = integralImg.ptr<int>();
  cv::Size frameSize(integralImg.cols - 1, integralImg.rows - 1);

  // Parallel Loop to process descriptors
#ifndef UPM_BEBLID_PARALLEL
  const cv::Range range(0, keypoints.size());
#else
  parallel_for_(cv::Range(0, keypoints.size()), [&](const cv::Range &range) {
#endif
    // Get a pointer to the first element in the range
    ABWLParams *wl;
    float responseFun;
    int areaResponseFun, kpIdx, wlIdx;
    int box1x1, box1y1, box1x2, box1y2, box2x1, box2y1, box2x2, box2y2, bit_idx, side;
    uchar byte = 0;
    std::vector<ABWLParams> imgWLParams(wl_params_.size());
    uchar *d = &descriptors.at<uchar>(range.start, 0);

    for (kpIdx = range.start; kpIdx < range.end; kpIdx++) {
      // Rectify the weak learners coordinates using the keypoint information
      rectifyABWL(wl_params_, imgWLParams, keypoints[kpIdx], scale_factor_, params_.patchSize);
      if (isKeypointInTheBorder(keypoints[kpIdx], frameSize, params_.patchSize, scale_factor_)) {
        // Code to process the keypoints in the image margins
        for (wlIdx = 0; wlIdx < params_.nWLs; wlIdx++) {
          bit_idx = 7 - wlIdx % 8;
          responseFun = computeABWLResponse(imgWLParams[wlIdx], integralImg);
          // Set the bit to 1 if the response function is less or equal to the threshod
          byte |= (responseFun <= params_.thresholds[wlIdx]) << bit_idx;
          // If we filled the byte, save it
          if (bit_idx == 0) {
            *d = byte;
            byte = 0;
            d++;
          }
        }
      } else {
        // Code to process the keypoints in the image center
        wl = imgWLParams.data();
        for (wlIdx = 0; wlIdx < params_.nWLs; wlIdx++) {
          bit_idx = 7 - wlIdx % 8;

          // For the first box, we calculate its margin coordinates
          box1x1 = wl->x1 - wl->boxRadius;
          box1y1 = (wl->y1 - wl->boxRadius) * integralImg.cols;
          box1x2 = wl->x1 + wl->boxRadius + 1;
          box1y2 = (wl->y1 + wl->boxRadius + 1) * integralImg.cols;
          // For the second box, we calculate its margin coordinates
          box2x1 = wl->x2 - wl->boxRadius;
          box2y1 = (wl->y2 - wl->boxRadius) * integralImg.cols;
          box2x2 = wl->x2 + wl->boxRadius + 1;
          box2y2 = (wl->y2 + wl->boxRadius + 1) * integralImg.cols;
          side = 1 + (wl->boxRadius << 1);

#ifndef NDEBUG

          // Check that all the image indices are inside the image. Only in debug mode
          int A1 = box1y1 + box1x1;
          assert(A1 >= 0 && A1 < integralImg.size().area());
          int D1 = box1y2 + box1x2;
          assert(D1 >= 0 && D1 < integralImg.size().area());
          int B1 = box1y1 + box1x2;
          assert(B1 >= 0 && B1 < integralImg.size().area());
          int C1 = box1y2 + box1x1;
          assert(C1 >= 0 && C1 < integralImg.size().area());
          int A2 = box2y1 + box2x1;
          assert(A2 >= 0 && A2 < integralImg.size().area());
          int D2 = box2y2 + box2x2;
          assert(D2 >= 0 && D2 < integralImg.size().area());
          int B2 = box2y1 + box2x2;
          assert(B2 >= 0 && B2 < integralImg.size().area());
          int C2 = box2y2 + box2x1;
          assert(C2 >= 0 && C2 < integralImg.size().area());
#endif
          // Get the difference between the average level of the two boxes
          areaResponseFun = (integralPtr[box1y1 + box1x1]  // A of Box1
              + integralPtr[box1y2 + box1x2]               // D of Box1
              - integralPtr[box1y1 + box1x2]               // B of Box1
              - integralPtr[box1y2 + box1x1]               // C of Box1
              - integralPtr[box2y1 + box2x1]               // A of Box2
              - integralPtr[box2y2 + box2x2]               // D of Box2
              + integralPtr[box2y1 + box2x2]               // B of Box2
              + integralPtr[box2y2 + box2x1]);             // C of Box2

          // Set the bit to 1 if the response function is less or equal to the threshod
          byte |= (areaResponseFun <= (params_.thresholds[wlIdx] * (side * side))) << bit_idx;
          wl++;
          // If we filled the byte, save it
          if (bit_idx == 0) {
            *d = byte;
            byte = 0;
            d++;
          }
        }  // End of for each dimension
      }  // End of else (of pixels in the image center)
    }  // End of for each keypoint
#ifdef UPM_BEBLID_PARALLEL
  });
#endif
}