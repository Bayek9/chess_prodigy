/*
 * Copyright (c) 2021 fgsoruco.
 * See LICENSE for more details.
 */
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter/services.dart';
import 'package:flutter_cache_manager/flutter_cache_manager.dart';
import 'package:opencv_4/factory/pathfrom.dart';
import 'package:opencv_4/factory/utils.dart';

class RefineBoardCornersFactory {
  static const MethodChannel platform = MethodChannel('opencv_4');

  static Future<Map<dynamic, dynamic>?> refineBoardCorners({
    required CVPathFrom pathFrom,
    required String pathString,
    required List<double> corners,
    Uint8List? data,
    double roiPaddingRatio = 0.22,
    double minAreaRatio = 0.015,
    double approxEpsilonRatio = 0.02,
    int cannyLow = 40,
    int cannyHigh = 140,
    int subPixWinSize = 5,
    int maxCandidates = 120,
  }) async {
    Uint8List payloadBytes = data ?? Uint8List(0);
    int pathType = 2;
    String pathPayload = '';

    if (payloadBytes.isEmpty) {
      switch (pathFrom) {
        case CVPathFrom.GALLERY_CAMERA:
          pathType = 1;
          pathPayload = pathString;
          break;
        case CVPathFrom.URL:
          final File file =
              await DefaultCacheManager().getSingleFile(pathString);
          payloadBytes = await file.readAsBytes();
          pathType = 2;
          break;
        case CVPathFrom.ASSETS:
          payloadBytes = await Utils.imgAssets2Uint8List(pathString);
          pathType = 3;
          break;
      }
    } else {
      switch (pathFrom) {
        case CVPathFrom.ASSETS:
          pathType = 3;
          break;
        case CVPathFrom.URL:
          pathType = 2;
          break;
        case CVPathFrom.GALLERY_CAMERA:
          pathType = 2;
          break;
      }
    }

    try {
      final dynamic result = await platform.invokeMethod('refineBoardCorners', {
        'pathType': pathType,
        'pathString': pathPayload,
        'data': payloadBytes,
        'corners': corners,
        'roiPaddingRatio': roiPaddingRatio,
        'minAreaRatio': minAreaRatio,
        'approxEpsilonRatio': approxEpsilonRatio,
        'cannyLow': cannyLow,
        'cannyHigh': cannyHigh,
        'subPixWinSize': subPixWinSize,
        'maxCandidates': maxCandidates,
      });
      if (result is Map) {
        return result;
      }
      return null;
    } on PlatformException {
      return null;
    }
  }
}
