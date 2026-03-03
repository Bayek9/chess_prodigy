import 'dart:async';
import 'dart:math' as math;
import 'dart:typed_data';
import 'dart:ui' as ui;

import '../../domain/entities/board_geometry.dart';
import '../../domain/entities/scan_image.dart';
import '../../domain/services/board_rectifier.dart';

class PerspectiveBoardRectifier implements BoardRectifier {
  const PerspectiveBoardRectifier({this.targetSize = 1024});

  final int targetSize;

  @override
  Future<RectifiedBoardImage> rectify({
    required ScanInputImage image,
    required BoardGeometry geometry,
  }) async {
    final source = await _decodeRgba(image.bytes);
    if (source == null || !geometry.isValid) {
      return RectifiedBoardImage(
        bytes: image.bytes,
        width: source?.width ?? 0,
        height: source?.height ?? 0,
        debugLabel: 'warp_fallback_source',
      );
    }

    final orderedCorners = _orderCorners(geometry.corners);
    final homography = _homographyUnitToQuad(orderedCorners);
    if (homography == null) {
      return RectifiedBoardImage(
        bytes: image.bytes,
        width: source.width,
        height: source.height,
        debugLabel: 'warp_failed_homography',
      );
    }

    final warped = Uint8List(targetSize * targetSize * 4);
    for (int y = 0; y < targetSize; y++) {
      final v = (y + 0.5) / targetSize;
      final rowOffset = y * targetSize;
      for (int x = 0; x < targetSize; x++) {
        final u = (x + 0.5) / targetSize;
        final p = _applyHomography(homography, u, v);
        final color = _sampleRgbaBilinear(
          rgba: source.rgba,
          width: source.width,
          height: source.height,
          x: p.x,
          y: p.y,
        );
        final dst = (rowOffset + x) * 4;
        warped[dst] = color[0];
        warped[dst + 1] = color[1];
        warped[dst + 2] = color[2];
        warped[dst + 3] = color[3];
      }
    }

    final encoded = await _encodePngRgba(
      rgba: warped,
      width: targetSize,
      height: targetSize,
    );
    if (encoded == null || encoded.isEmpty) {
      return RectifiedBoardImage(
        bytes: image.bytes,
        width: source.width,
        height: source.height,
        debugLabel: 'warp_failed_encode',
      );
    }

    return RectifiedBoardImage(
      bytes: encoded,
      width: targetSize,
      height: targetSize,
      debugLabel: 'warp_homography',
    );
  }

  Future<_DecodedRgbaImage?> _decodeRgba(Uint8List bytes) async {
    try {
      final codec = await ui.instantiateImageCodec(bytes);
      final frame = await codec.getNextFrame();
      final image = frame.image;
      final data = await image.toByteData(format: ui.ImageByteFormat.rawRgba);
      if (data == null) {
        image.dispose();
        return null;
      }
      final decoded = _DecodedRgbaImage(
        width: image.width,
        height: image.height,
        rgba: Uint8List.fromList(data.buffer.asUint8List()),
      );
      image.dispose();
      return decoded;
    } catch (_) {
      return null;
    }
  }

  Future<Uint8List?> _encodePngRgba({
    required Uint8List rgba,
    required int width,
    required int height,
  }) async {
    try {
      final completer = Completer<ui.Image>();
      ui.decodeImageFromPixels(
        rgba,
        width,
        height,
        ui.PixelFormat.rgba8888,
        (ui.Image image) => completer.complete(image),
      );
      final image = await completer.future;
      final data = await image.toByteData(format: ui.ImageByteFormat.png);
      image.dispose();
      return data?.buffer.asUint8List();
    } catch (_) {
      return null;
    }
  }

  List<BoardCorner> _orderCorners(List<BoardCorner> corners) {
    BoardCorner topLeft = corners.first;
    BoardCorner topRight = corners.first;
    BoardCorner bottomRight = corners.first;
    BoardCorner bottomLeft = corners.first;
    double topLeftScore = double.infinity;
    double bottomRightScore = -double.infinity;
    double topRightScore = -double.infinity;
    double bottomLeftScore = double.infinity;

    for (final corner in corners) {
      final sum = corner.x + corner.y;
      final diff = corner.x - corner.y;
      if (sum < topLeftScore) {
        topLeftScore = sum;
        topLeft = corner;
      }
      if (sum > bottomRightScore) {
        bottomRightScore = sum;
        bottomRight = corner;
      }
      if (diff > topRightScore) {
        topRightScore = diff;
        topRight = corner;
      }
      if (diff < bottomLeftScore) {
        bottomLeftScore = diff;
        bottomLeft = corner;
      }
    }

    return <BoardCorner>[topLeft, topRight, bottomRight, bottomLeft];
  }

  List<double>? _homographyUnitToQuad(List<BoardCorner> corners) {
    if (corners.length != 4) {
      return null;
    }

    final points = <_PointPair>[
      _PointPair(u: 0.0, v: 0.0, x: corners[0].x, y: corners[0].y),
      _PointPair(u: 1.0, v: 0.0, x: corners[1].x, y: corners[1].y),
      _PointPair(u: 1.0, v: 1.0, x: corners[2].x, y: corners[2].y),
      _PointPair(u: 0.0, v: 1.0, x: corners[3].x, y: corners[3].y),
    ];

    final a = List<List<double>>.generate(
      8,
      (_) => List<double>.filled(8, 0.0),
      growable: false,
    );
    final b = List<double>.filled(8, 0.0);

    int row = 0;
    for (final p in points) {
      a[row][0] = p.u;
      a[row][1] = p.v;
      a[row][2] = 1.0;
      a[row][6] = -p.u * p.x;
      a[row][7] = -p.v * p.x;
      b[row] = p.x;
      row += 1;

      a[row][3] = p.u;
      a[row][4] = p.v;
      a[row][5] = 1.0;
      a[row][6] = -p.u * p.y;
      a[row][7] = -p.v * p.y;
      b[row] = p.y;
      row += 1;
    }

    final solution = _solveLinearSystem(a, b);
    if (solution == null) {
      return null;
    }
    return <double>[
      solution[0],
      solution[1],
      solution[2],
      solution[3],
      solution[4],
      solution[5],
      solution[6],
      solution[7],
      1.0,
    ];
  }

  List<double>? _solveLinearSystem(List<List<double>> a, List<double> b) {
    final n = b.length;
    final aug = List<List<double>>.generate(
      n,
      (r) => <double>[...a[r], b[r]],
      growable: false,
    );

    for (int col = 0; col < n; col++) {
      int pivot = col;
      double pivotAbs = aug[pivot][col].abs();
      for (int row = col + 1; row < n; row++) {
        final v = aug[row][col].abs();
        if (v > pivotAbs) {
          pivotAbs = v;
          pivot = row;
        }
      }
      if (pivotAbs < 1e-9) {
        return null;
      }
      if (pivot != col) {
        final tmp = aug[pivot];
        aug[pivot] = aug[col];
        aug[col] = tmp;
      }

      final pivotValue = aug[col][col];
      for (int k = col; k <= n; k++) {
        aug[col][k] /= pivotValue;
      }
      for (int row = 0; row < n; row++) {
        if (row == col) {
          continue;
        }
        final factor = aug[row][col];
        if (factor.abs() < 1e-12) {
          continue;
        }
        for (int k = col; k <= n; k++) {
          aug[row][k] -= factor * aug[col][k];
        }
      }
    }

    final x = List<double>.filled(n, 0.0);
    for (int i = 0; i < n; i++) {
      x[i] = aug[i][n];
    }
    return x;
  }

  _PointD _applyHomography(List<double> h, double u, double v) {
    final den = (h[6] * u) + (h[7] * v) + h[8];
    if (den.abs() < 1e-9) {
      return const _PointD(0.0, 0.0);
    }
    final x = ((h[0] * u) + (h[1] * v) + h[2]) / den;
    final y = ((h[3] * u) + (h[4] * v) + h[5]) / den;
    return _PointD(x, y);
  }

  List<int> _sampleRgbaBilinear({
    required Uint8List rgba,
    required int width,
    required int height,
    required double x,
    required double y,
  }) {
    final clampedX = x.clamp(0.0, (width - 1).toDouble());
    final clampedY = y.clamp(0.0, (height - 1).toDouble());

    final x0 = clampedX.floor();
    final y0 = clampedY.floor();
    final x1 = math.min(width - 1, x0 + 1);
    final y1 = math.min(height - 1, y0 + 1);
    final tx = clampedX - x0;
    final ty = clampedY - y0;

    final p00 = (y0 * width + x0) * 4;
    final p10 = (y0 * width + x1) * 4;
    final p01 = (y1 * width + x0) * 4;
    final p11 = (y1 * width + x1) * 4;

    int sample(int channel) {
      final c00 = rgba[p00 + channel].toDouble();
      final c10 = rgba[p10 + channel].toDouble();
      final c01 = rgba[p01 + channel].toDouble();
      final c11 = rgba[p11 + channel].toDouble();
      final top = c00 + ((c10 - c00) * tx);
      final bottom = c01 + ((c11 - c01) * tx);
      return (top + ((bottom - top) * ty)).round().clamp(0, 255);
    }

    return <int>[sample(0), sample(1), sample(2), sample(3)];
  }
}

class _DecodedRgbaImage {
  const _DecodedRgbaImage({
    required this.width,
    required this.height,
    required this.rgba,
  });

  final int width;
  final int height;
  final Uint8List rgba;
}

class _PointPair {
  const _PointPair({
    required this.u,
    required this.v,
    required this.x,
    required this.y,
  });

  final double u;
  final double v;
  final double x;
  final double y;
}

class _PointD {
  const _PointD(this.x, this.y);

  final double x;
  final double y;
}
