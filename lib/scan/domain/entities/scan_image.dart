import 'package:flutter/foundation.dart';

@immutable
class ScanInputImage {
  const ScanInputImage({required this.path, required this.bytes});

  final String path;
  final Uint8List bytes;
}

@immutable
class RectifiedBoardImage {
  const RectifiedBoardImage({
    required this.bytes,
    required this.width,
    required this.height,
    this.debugLabel = 'rectified_board',
  });

  final Uint8List bytes;
  final int width;
  final int height;
  final String debugLabel;
}
