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
// Copyright (c) 2020, The Graffter and Universidad Politecnica de Madrid.
// All rights reserved.
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

#include <iostream>
#include <opencv2/opencv.hpp>
#include "BEBLID.h"

/**
 * This demo shows how BEBLID descriptor can be used with a feature detector (here ORB) to
 * detect, describe and match two images of the same scene.
 */
int main() {

  // Read the input images in grayscale format (CV_8UC1)
  cv::Mat img1 = cv::imread("../imgs/img1.jpg", cv::IMREAD_GRAYSCALE);
  cv::Mat img2 = cv::imread("../imgs/img3.jpg", cv::IMREAD_GRAYSCALE);

  // Create the feature detector, for example ORB
  auto detector = cv::ORB::create();

  // Detect features in both images
  std::vector<cv::KeyPoint> points1, points2;
  detector->detect(img1, points1);
  detector->detect(img2, points2);
  std::cout << "Detected  " << points1.size() << " kps in image1" << std::endl;
  std::cout << "Detected  " << points2.size() << " kps in image2" << std::endl;

  // Use 64 bytes per descriptor and configure the scale factor for ORB detector
  auto descriptor = BEBLID::create(256, 0.75);

  // Describe the detected features i both images
  cv::Mat descriptors1, descriptors2;
  descriptor->compute(img1, points1, descriptors1);
  descriptor->compute(img2, points2, descriptors2);
  std::cout << "Points described" << std::endl;

  // Match the generated descriptors for img1 and img2 using brute force matching
  cv::BFMatcher matcher(cv::NORM_HAMMING, true);
  std::vector<cv::DMatch> matches;
  matcher.match(descriptors1, descriptors2, matches);
  std::cout << "Number of matches: " << matches.size() << std::endl;

  // If there is not enough matches exit
  if (matches.size() < 4) exit(-1);

  // Take only the matched points that will be used to calculate the
  // transformation between both images
  std::vector<cv::Point2d> matched_pts1, matched_pts2;
  for (cv::DMatch match : matches) {
    matched_pts1.push_back(points1[match.queryIdx].pt);
    matched_pts2.push_back(points2[match.trainIdx].pt);
  }

  // Find the homography that transforms a point in the first image to a point in the second image.
  cv::Mat inliers;
  cv::Mat H = cv::findHomography(matched_pts1, matched_pts2, cv::RANSAC, 3, inliers);
  // Print the number of inliers, that is, the number of points correctly
  // mapped by the transformation that we have estimated
  std::cout << "Number of inliers " << cv::sum(inliers)[0]
            << " ( " << (100.0f * cv::sum(inliers)[0] / matches.size()) << "% )" << std::endl;

  // Convert the image to BRG format from grayscale
  cv::cvtColor(img1, img1, cv::COLOR_GRAY2BGR);
  cv::cvtColor(img2, img2, cv::COLOR_GRAY2BGR);

  // Draw all the matched keypoints in red color
  cv::Mat all_matches_img;
  cv::drawMatches(img1,
                  points1,
                  img2,
                  points2,
                  matches,
                  all_matches_img,
                  CV_RGB(255, 0, 0),  // Red color
                  CV_RGB(255, 0, 0));  // Red color

  // Draw the inliers in green color
  for (int i = 0; i < matched_pts1.size(); i++) {
    if (inliers.at<uchar>(i, 0)) {
      cv::circle(all_matches_img, matched_pts1[i], 3, CV_RGB(0, 255, 0), 2);
      // Calculate second point assuming that img1 and img2 have the same height
      cv::Point p2(matched_pts2[i].x + img1.cols, matched_pts2[i].y);
      cv::circle(all_matches_img, p2, 3, CV_RGB(0, 255, 0), 2);
      cv::line(all_matches_img, matched_pts1[i], p2, CV_RGB(0, 255, 0), 2);
    }
  }

  // Show and save the result
  cv::imshow("All matches", all_matches_img);
  cv::imwrite("../imgs/inliners_img.jpg", all_matches_img);

  // Transform the first image to look like the second one
  cv::Mat transformed;
  cv::warpPerspective(img1, transformed, H, img2.size());
  cv::imshow("Original image", img2);
  cv::imshow("Transformed image", transformed);

  cv::waitKey();
  return 0;
}
