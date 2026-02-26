import 'dart:math' as math;
import 'dart:typed_data';
import 'dart:ui' as ui;

import '../../domain/entities/board_geometry.dart';
import '../../domain/entities/scan_image.dart';
import '../../domain/services/board_detector.dart';

class StatisticalBoardDetector implements BoardDetector {
  const StatisticalBoardDetector({
    this.minRelativeSide = 0.45,
    this.maxAnalysisDimension = 240,
  });

  final double minRelativeSide;
  final int maxAnalysisDimension;

  @override
  Future<BoardGeometry> detect(ScanInputImage image) async {
    try {
      final decoded = await _decodeRgba(image.bytes);
      if (decoded == null || decoded.width < 16 || decoded.height < 16) {
        return const BoardGeometry(corners: <BoardCorner>[]);
      }

      final sourceLuminance = _toLuminance(decoded.rgba);
      final analyzed = _downsampleLuminance(
        sourceLuminance: sourceLuminance,
        sourceWidth: decoded.width,
        sourceHeight: decoded.height,
      );

      if (analyzed.width < 16 || analyzed.height < 16) {
        return const BoardGeometry(corners: <BoardCorner>[]);
      }

      final gradients = _buildGradients(
        luminance: analyzed.luminance,
        width: analyzed.width,
        height: analyzed.height,
      );
      final rowEnergy = _axisEnergyRows(
        gradients: gradients,
        width: analyzed.width,
        height: analyzed.height,
      );
      final colEnergy = _axisEnergyCols(
        gradients: gradients,
        width: analyzed.width,
        height: analyzed.height,
      );
      final rowPrefix = _buildPrefix1D(rowEnergy);
      final colPrefix = _buildPrefix1D(colEnergy);
      final rowBand = _estimateAxisBand(
        values: rowEnergy,
        minSpan: math.max(8, (analyzed.height * 0.18).round()),
      );
      final colBand = _estimateAxisBand(
        values: colEnergy,
        minSpan: math.max(8, (analyzed.width * 0.18).round()),
      );
      final contentBounds = _estimateContentBounds(
        luminance: analyzed.luminance,
        gradients: gradients,
        width: analyzed.width,
        height: analyzed.height,
      );
      final integralX = _buildIntegral(
        values: gradients.dx,
        width: analyzed.width,
        height: analyzed.height,
      );
      final integralY = _buildIntegral(
        values: gradients.dy,
        width: analyzed.width,
        height: analyzed.height,
      );

      final minDim = math.min(analyzed.width, analyzed.height);
      final aspectSkew =
          (analyzed.width - analyzed.height).abs() /
          math.max(analyzed.width, analyzed.height);
      final effectiveMinRelativeSide = aspectSkew < 0.10
          ? math.max(minRelativeSide, 0.78)
          : minRelativeSide;
      final minSide = math.max(18, (minDim * effectiveMinRelativeSide).round());
      final maxSide = math.max(minSide, (minDim * 0.98).round());
      if (maxSide <= minSide) {
        return const BoardGeometry(corners: <BoardCorner>[]);
      }

      final candidates = _buildCandidateSeeds(
        width: analyzed.width,
        height: analyzed.height,
        minSide: minSide,
        maxSide: maxSide,
        minDim: minDim,
        integralX: integralX,
        integralY: integralY,
        rowPrefix: rowPrefix,
        colPrefix: colPrefix,
        rowLength: analyzed.height,
        colLength: analyzed.width,
        rowBand: rowBand,
        colBand: colBand,
        searchBounds: contentBounds,
      );
      if (candidates.isEmpty) {
        return const BoardGeometry(corners: <BoardCorner>[]);
      }
      final luminanceIntegral = _buildIntegralUint8(
        values: analyzed.luminance,
        width: analyzed.width,
        height: analyzed.height,
      );
      final maxSeedScore = candidates
          .map((c) => c.score)
          .reduce(math.max)
          .clamp(1e-6, double.infinity)
          .toDouble();
      final minSeedScore = candidates
          .map((c) => c.score)
          .reduce(math.min)
          .toDouble();

      _CandidateEvaluation? bestEval;
      for (final seed in candidates) {
        final refined = _refineCandidateByEnergyBands(
          candidate: seed,
          gradients: gradients,
          width: analyzed.width,
          height: analyzed.height,
        );
        final refinedQuad = _refineToQuadrilateral(
          rect: refined,
          gradients: gradients,
          width: analyzed.width,
          height: analyzed.height,
        );
        final normalizedSeed = _normalizeSeedScore(
          score: seed.score,
          minScore: minSeedScore,
          maxScore: maxSeedScore,
        );
        final boardScore = _boardLikenessScore(
          quad: refinedQuad,
          rect: refined,
          seedScore: normalizedSeed,
          luminance: analyzed.luminance,
          gradients: gradients,
          luminanceIntegral: luminanceIntegral,
          width: analyzed.width,
          height: analyzed.height,
          contentBounds: contentBounds,
        );
        final candidate = _CandidateEvaluation(
          seed: seed,
          rect: refined,
          quad: refinedQuad,
          score: boardScore,
        );
        if (bestEval == null || candidate.score > bestEval.score) {
          bestEval = candidate;
        }
      }

      if (bestEval == null) {
        return const BoardGeometry(corners: <BoardCorner>[]);
      }
      final refinedQuad = _snapQuadToOuterEdges(
        quad: bestEval.quad,
        gradients: gradients,
        width: analyzed.width,
        height: analyzed.height,
      );

      final scaleX = decoded.width / analyzed.width;
      final scaleY = decoded.height / analyzed.height;

      final scaledCorners = _scaleAndClampQuad(
        quad: refinedQuad,
        scaleX: scaleX,
        scaleY: scaleY,
        outWidth: decoded.width,
        outHeight: decoded.height,
      );

      final bounds = _quadBounds(scaledCorners);
      final detectedWidth = bounds.width;
      final detectedHeight = bounds.height;
      final sourceMinDim = math.min(decoded.width, decoded.height);
      final minDetectedSide = math.max(32, (sourceMinDim * 0.14).round());
      if (detectedWidth < minDetectedSide || detectedHeight < minDetectedSide) {
        return const BoardGeometry(corners: <BoardCorner>[]);
      }

      return BoardGeometry(
        corners: <BoardCorner>[
          BoardCorner(x: scaledCorners.tl.x, y: scaledCorners.tl.y),
          BoardCorner(x: scaledCorners.tr.x, y: scaledCorners.tr.y),
          BoardCorner(x: scaledCorners.br.x, y: scaledCorners.br.y),
          BoardCorner(x: scaledCorners.bl.x, y: scaledCorners.bl.y),
        ],
      );
    } catch (_) {
      return const BoardGeometry(corners: <BoardCorner>[]);
    }
  }

  _SearchBounds _estimateContentBounds({
    required Uint8List luminance,
    required _GradientMaps gradients,
    required int width,
    required int height,
  }) {
    if (width < 8 || height < 8) {
      return _SearchBounds(left: 0, top: 0, right: width - 1, bottom: height - 1);
    }

    final rowLuma = Float64List(height);
    final colLuma = Float64List(width);
    final rowEdge = Float64List(height);
    final colEdge = Float64List(width);

    for (int y = 0; y < height; y++) {
      final rowOffset = y * width;
      for (int x = 0; x < width; x++) {
        final idx = rowOffset + x;
        final l = luminance[idx].toDouble();
        final e = gradients.dx[idx] + gradients.dy[idx];
        rowLuma[y] += l;
        colLuma[x] += l;
        rowEdge[y] += e;
        colEdge[x] += e;
      }
    }

    for (int y = 0; y < height; y++) {
      rowLuma[y] /= width;
      rowEdge[y] /= width;
    }
    for (int x = 0; x < width; x++) {
      colLuma[x] /= height;
      colEdge[x] /= height;
    }

    final meanRowLuma = rowLuma.reduce((a, b) => a + b) / height;
    final meanColLuma = colLuma.reduce((a, b) => a + b) / width;
    final meanRowEdge = rowEdge.reduce((a, b) => a + b) / height;
    final meanColEdge = colEdge.reduce((a, b) => a + b) / width;

    final rowLumaThreshold = math.max(10.0, meanRowLuma * 0.42);
    final colLumaThreshold = math.max(10.0, meanColLuma * 0.42);
    final rowEdgeThreshold = math.max(4.0, meanRowEdge * 0.50);
    final colEdgeThreshold = math.max(4.0, meanColEdge * 0.50);

    int top = 0;
    while (
      top < height - 1 &&
      rowLuma[top] < rowLumaThreshold &&
      rowEdge[top] < rowEdgeThreshold
    ) {
      top += 1;
    }

    int bottom = height - 1;
    while (
      bottom > 0 &&
      rowLuma[bottom] < rowLumaThreshold &&
      rowEdge[bottom] < rowEdgeThreshold
    ) {
      bottom -= 1;
    }

    int left = 0;
    while (
      left < width - 1 &&
      colLuma[left] < colLumaThreshold &&
      colEdge[left] < colEdgeThreshold
    ) {
      left += 1;
    }

    int right = width - 1;
    while (
      right > 0 &&
      colLuma[right] < colLumaThreshold &&
      colEdge[right] < colEdgeThreshold
    ) {
      right -= 1;
    }

    if (top >= bottom) {
      top = 0;
      bottom = height - 1;
    }
    if (left >= right) {
      left = 0;
      right = width - 1;
    }

    final minHeight = math.max(12, (height * 0.40).round());
    final minWidth = math.max(12, (width * 0.68).round());
    if (bottom - top + 1 < minHeight) {
      top = 0;
      bottom = height - 1;
    }
    if (right - left + 1 < minWidth) {
      left = 0;
      right = width - 1;
    }

    final padX = math.max(2, (width * 0.02).round());
    final padY = math.max(2, (height * 0.02).round());
    final paddedLeft = math.max(0, left - padX);
    final paddedTop = math.max(0, top - padY);
    final paddedRight = math.min(width - 1, right + padX);
    final paddedBottom = math.min(height - 1, bottom + padY);

    final nearSquareFrame =
        ((width - height).abs() / math.max(width, height)) <= 0.16;
    final paddedWidth = paddedRight - paddedLeft + 1;
    final paddedHeight = paddedBottom - paddedTop + 1;
    if (nearSquareFrame && paddedWidth < (width * 0.78).round()) {
      return _SearchBounds(left: 0, top: 0, right: width - 1, bottom: height - 1);
    }
    if (nearSquareFrame && paddedHeight < (height * 0.78).round()) {
      return _SearchBounds(left: 0, top: 0, right: width - 1, bottom: height - 1);
    }

    return _SearchBounds(
      left: paddedLeft,
      top: paddedTop,
      right: paddedRight,
      bottom: paddedBottom,
    );
  }

  List<_WindowCandidate> _buildCandidateSeeds({
    required int width,
    required int height,
    required int minSide,
    required int maxSide,
    required int minDim,
    required Float64List integralX,
    required Float64List integralY,
    required Float64List rowPrefix,
    required Float64List colPrefix,
    required int rowLength,
    required int colLength,
    required _AxisBand rowBand,
    required _AxisBand colBand,
    required _SearchBounds searchBounds,
  }) {
    final candidates = <_WindowCandidate>[];
    void addCandidate(_WindowCandidate? candidate) {
      if (candidate == null) {
        return;
      }
      for (int i = 0; i < candidates.length; i++) {
        final existing = candidates[i];
        if ((existing.left - candidate.left).abs() <= 6 &&
            (existing.top - candidate.top).abs() <= 6 &&
            (existing.side - candidate.side).abs() <= 8) {
          if (candidate.score > existing.score) {
            candidates[i] = candidate;
          }
          return;
        }
      }
      candidates.add(candidate);
    }

    final fullBounds = _SearchBounds(
      left: 0,
      top: 0,
      right: width - 1,
      bottom: height - 1,
    );

    addCandidate(
      _searchBestCandidateCoarseFine(
        width: width,
        height: height,
        minSide: minSide,
        maxSide: maxSide,
        minDim: minDim,
        integralX: integralX,
        integralY: integralY,
        rowPrefix: rowPrefix,
        colPrefix: colPrefix,
        rowLength: rowLength,
        colLength: colLength,
        rowBand: rowBand,
        colBand: colBand,
        searchBounds: searchBounds,
      ),
    );

    final isContentAlmostFull =
        searchBounds.left <= 2 &&
        searchBounds.top <= 2 &&
        searchBounds.right >= width - 3 &&
        searchBounds.bottom >= height - 3;
    if (!isContentAlmostFull) {
      addCandidate(
        _searchBestCandidateCoarseFine(
          width: width,
          height: height,
          minSide: minSide,
          maxSide: maxSide,
          minDim: minDim,
          integralX: integralX,
          integralY: integralY,
          rowPrefix: rowPrefix,
          colPrefix: colPrefix,
          rowLength: rowLength,
          colLength: colLength,
          rowBand: rowBand,
          colBand: colBand,
          searchBounds: fullBounds,
        ),
      );
    }

    final bandCenterX = (colBand.start + colBand.end) * 0.5;
    final bandCenterY = (rowBand.start + rowBand.end) * 0.5;
    final bandSideBase = math.min(colBand.length, rowBand.length).clamp(
      minSide,
      maxSide,
    );
    final bandScales = <double>[0.90, 1.0, 1.08];
    for (final scale in bandScales) {
      final side = (bandSideBase * scale).round().clamp(minSide, maxSide);
      addCandidate(
        _approximateCandidateFromCenter(
          centerX: bandCenterX,
          centerY: bandCenterY,
          side: side,
          bounds: searchBounds,
          width: width,
          height: height,
          minDim: minDim,
          integralX: integralX,
          integralY: integralY,
          rowPrefix: rowPrefix,
          colPrefix: colPrefix,
          rowLength: rowLength,
          colLength: colLength,
          rowBand: rowBand,
          colBand: colBand,
        ),
      );
    }

    final contentCenterX = (searchBounds.left + searchBounds.right) * 0.5;
    final contentCenterY = (searchBounds.top + searchBounds.bottom) * 0.5;
    final contentSideBase = math.min(searchBounds.width, searchBounds.height)
        .clamp(minSide, maxSide);
    final contentScales = <double>[1.0, 0.92, 0.84];
    for (final scale in contentScales) {
      final side = (contentSideBase * scale).round().clamp(minSide, maxSide);
      addCandidate(
        _approximateCandidateFromCenter(
          centerX: contentCenterX,
          centerY: contentCenterY,
          side: side,
          bounds: searchBounds,
          width: width,
          height: height,
          minDim: minDim,
          integralX: integralX,
          integralY: integralY,
          rowPrefix: rowPrefix,
          colPrefix: colPrefix,
          rowLength: rowLength,
          colLength: colLength,
          rowBand: rowBand,
          colBand: colBand,
        ),
      );
    }

    final nearSquareFrame =
        ((width - height).abs() / math.max(width, height)) <= 0.12;
    if (nearSquareFrame) {
      final largeScales = <double>[1.0, 0.94];
      for (final scale in largeScales) {
        final side = (maxSide * scale).round().clamp(minSide, maxSide);
        addCandidate(
          _approximateCandidateFromCenter(
            centerX: (width - 1) * 0.5,
            centerY: (height - 1) * 0.5,
            side: side,
            bounds: fullBounds,
            width: width,
            height: height,
            minDim: minDim,
            integralX: integralX,
            integralY: integralY,
            rowPrefix: rowPrefix,
            colPrefix: colPrefix,
            rowLength: rowLength,
            colLength: colLength,
            rowBand: rowBand,
            colBand: colBand,
          ),
        );
      }
    }

    candidates.sort((a, b) => b.score.compareTo(a.score));
    return candidates.take(10).toList(growable: false);
  }

  _WindowCandidate? _approximateCandidateFromCenter({
    required double centerX,
    required double centerY,
    required int side,
    required _SearchBounds bounds,
    required int width,
    required int height,
    required int minDim,
    required Float64List integralX,
    required Float64List integralY,
    required Float64List rowPrefix,
    required Float64List colPrefix,
    required int rowLength,
    required int colLength,
    required _AxisBand rowBand,
    required _AxisBand colBand,
  }) {
    if (side <= 0 || side > width || side > height) {
      return null;
    }
    final maxLeft = math.min(bounds.right - side + 1, width - side);
    final maxTop = math.min(bounds.bottom - side + 1, height - side);
    final minLeft = bounds.left;
    final minTop = bounds.top;
    if (maxLeft < minLeft || maxTop < minTop) {
      return null;
    }

    final left = (centerX - ((side - 1) * 0.5)).round().clamp(minLeft, maxLeft);
    final top = (centerY - ((side - 1) * 0.5)).round().clamp(minTop, maxTop);
    final score = _scoreWindow(
      width: width,
      height: height,
      left: left,
      top: top,
      side: side,
      minDim: minDim,
      integralX: integralX,
      integralY: integralY,
      rowPrefix: rowPrefix,
      colPrefix: colPrefix,
      rowLength: rowLength,
      colLength: colLength,
      rowBand: rowBand,
      colBand: colBand,
    );
    return _WindowCandidate(left: left, top: top, side: side, score: score);
  }

  double _normalizeSeedScore({
    required double score,
    required double minScore,
    required double maxScore,
  }) {
    final span = maxScore - minScore;
    if (span <= 1e-9) {
      return 1.0;
    }
    return ((score - minScore) / span).clamp(0.0, 1.0).toDouble();
  }

  double _boardLikenessScore({
    required _QuadCandidate quad,
    required _RectCandidate rect,
    required double seedScore,
    required Uint8List luminance,
    required _GradientMaps gradients,
    required Float64List luminanceIntegral,
    required int width,
    required int height,
    required _SearchBounds contentBounds,
  }) {
    final quadConfidence = _quadConfidence(
      quad: quad,
      rect: rect,
      width: width,
      height: height,
    );
    final bounds = _quadBounds(quad);
    final checker = _checkerPatternScore(
      bounds: bounds,
      luminanceIntegral: luminanceIntegral,
      width: width,
      height: height,
    );
    final gridRegularity = _gridRegularityScore(
      rect: rect,
      gradients: gradients,
      width: width,
      height: height,
    );
    final frameScore = _frameEdgeScore(
      rect: rect,
      gradients: gradients,
      width: width,
      height: height,
    );

    final contentMinDim = math.max(
      1.0,
      math.min(contentBounds.width, contentBounds.height).toDouble(),
    );
    final quadMinDim = math.min(bounds.width.toDouble(), bounds.height.toDouble());
    final contentCoverage = quadMinDim / contentMinDim;
    final frameSkew = (width - height).abs() / math.max(width, height);
    final nearSquareFrame = frameSkew <= 0.12;

    final imageArea = math.max(1.0, (width * height).toDouble());
    final areaRatio = _quadArea(quad) / imageArea;
    double areaPrior = 1.0;
    if (areaRatio < 0.05) {
      areaPrior *= 0.22;
    } else if (areaRatio < 0.12) {
      areaPrior *= 0.60;
    } else if (areaRatio < 0.18) {
      areaPrior *= 0.80;
    }
    if (areaRatio > 0.96) {
      areaPrior *= 0.55;
    } else if (areaRatio > 0.88) {
      areaPrior *= 0.78;
    }

    final overlapWithContent = _overlapRatioWithContent(
      bounds: bounds,
      content: contentBounds,
    );
    if (overlapWithContent < 0.45) {
      areaPrior *= 0.62;
    } else if (overlapWithContent < 0.62) {
      areaPrior *= 0.80;
    }
    if (contentCoverage < 0.40) {
      areaPrior *= 0.35;
    } else if (contentCoverage < 0.50) {
      areaPrior *= 0.60;
    } else if (contentCoverage < 0.62) {
      areaPrior *= 0.82;
    }
    if (nearSquareFrame) {
      if (areaRatio < 0.50) {
        areaPrior *= 0.42;
      } else if (areaRatio < 0.60) {
        areaPrior *= 0.70;
      }
    }

    final borderTouchCount = <bool>[
      bounds.left <= 1.0,
      bounds.top <= 1.0,
      bounds.right >= (width - 2).toDouble(),
      bounds.bottom >= (height - 2).toDouble(),
    ].where((v) => v).length;
    double touchPenalty = 1.0;
    if (borderTouchCount >= 3) {
      touchPenalty = 0.58;
    } else if (borderTouchCount == 2) {
      touchPenalty = 0.78;
    } else if (borderTouchCount == 1) {
      touchPenalty = 0.92;
    }

    double patternPenalty = 1.0;
    if (checker < 0.10) {
      patternPenalty = 0.45;
    } else if (checker < 0.16) {
      patternPenalty = 0.70;
    }

    return (seedScore * 0.08 +
            quadConfidence * 0.18 +
            checker * 0.44 +
            gridRegularity * 0.22 +
            frameScore * 0.08) *
        areaPrior *
        touchPenalty *
        patternPenalty;
  }

  double _checkerPatternScore({
    required _Bounds bounds,
    required Float64List luminanceIntegral,
    required int width,
    required int height,
  }) {
    final left = bounds.left.floor().clamp(0, width - 1);
    final top = bounds.top.floor().clamp(0, height - 1);
    final right = bounds.right.ceil().clamp(0, width - 1);
    final bottom = bounds.bottom.ceil().clamp(0, height - 1);
    final boxWidth = right - left + 1;
    final boxHeight = bottom - top + 1;
    if (boxWidth < 32 || boxHeight < 32) {
      return 0.0;
    }

    final cellWidth = boxWidth / 8.0;
    final cellHeight = boxHeight / 8.0;
    final means = List<double>.filled(64, 0.0);
    double evenSum = 0;
    double oddSum = 0;
    int evenCount = 0;
    int oddCount = 0;

    for (int row = 0; row < 8; row++) {
      for (int col = 0; col < 8; col++) {
        final x0Raw = (left + (col * cellWidth)).floor();
        final x1Raw = (left + ((col + 1) * cellWidth)).floor() - 1;
        final y0Raw = (top + (row * cellHeight)).floor();
        final y1Raw = (top + ((row + 1) * cellHeight)).floor() - 1;

        final marginX = math.max(0, (((x1Raw - x0Raw + 1) * 0.16)).round());
        final marginY = math.max(0, (((y1Raw - y0Raw + 1) * 0.16)).round());

        final x0 = (x0Raw + marginX).clamp(left, right);
        final x1 = math.max(x0, x1Raw - marginX).clamp(left, right);
        final y0 = (y0Raw + marginY).clamp(top, bottom);
        final y1 = math.max(y0, y1Raw - marginY).clamp(top, bottom);

        final mean = _rectMeanFromIntegral(
          integral: luminanceIntegral,
          width: width,
          left: x0,
          top: y0,
          right: x1,
          bottom: y1,
        );

        final idx = row * 8 + col;
        means[idx] = mean;
        if (((row + col) & 1) == 0) {
          evenSum += mean;
          evenCount += 1;
        } else {
          oddSum += mean;
          oddCount += 1;
        }
      }
    }

    if (evenCount == 0 || oddCount == 0) {
      return 0.0;
    }

    final evenMean = evenSum / evenCount;
    final oddMean = oddSum / oddCount;
    final contrast = ((evenMean - oddMean).abs() / 255.0).clamp(0.0, 1.0);

    double adjacencyDiff = 0;
    int adjacencyCount = 0;
    for (int row = 0; row < 8; row++) {
      for (int col = 0; col < 7; col++) {
        final a = means[row * 8 + col];
        final b = means[row * 8 + col + 1];
        adjacencyDiff += (a - b).abs();
        adjacencyCount += 1;
      }
    }
    for (int col = 0; col < 8; col++) {
      for (int row = 0; row < 7; row++) {
        final a = means[row * 8 + col];
        final b = means[(row + 1) * 8 + col];
        adjacencyDiff += (a - b).abs();
        adjacencyCount += 1;
      }
    }
    final adjacencyScore = adjacencyCount == 0
        ? 0.0
        : (adjacencyDiff / adjacencyCount / 255.0).clamp(0.0, 1.0);

    final allMean = means.reduce((a, b) => a + b) / means.length;
    double variance = 0;
    for (final m in means) {
      final d = m - allMean;
      variance += d * d;
    }
    variance /= means.length;
    final stdScore = (math.sqrt(variance) / 80.0).clamp(0.0, 1.0);

    return (contrast * 0.55 + adjacencyScore * 0.30 + stdScore * 0.15)
        .clamp(0.0, 1.0)
        .toDouble();
  }

  double _gridRegularityScore({
    required _RectCandidate rect,
    required _GradientMaps gradients,
    required int width,
    required int height,
  }) {
    final rectWidth = rect.right - rect.left + 1;
    final rectHeight = rect.bottom - rect.top + 1;
    if (rectWidth < 24 || rectHeight < 24) {
      return 0.0;
    }

    final profileX = _profileEnergyX(
      gradients: gradients,
      width: width,
      left: rect.left,
      right: rect.right,
      top: rect.top,
      bottom: rect.bottom,
    );
    final profileY = _profileEnergyY(
      gradients: gradients,
      width: width,
      left: rect.left,
      right: rect.right,
      top: rect.top,
      bottom: rect.bottom,
    );

    final peaksX = _countProminentPeaks(profileX);
    final peaksY = _countProminentPeaks(profileY);

    double scoreForPeaks(int peaks) {
      final distanceFromExpected = (peaks - 9).abs();
      return (1.0 - (distanceFromExpected / 9.0)).clamp(0.0, 1.0).toDouble();
    }

    return ((scoreForPeaks(peaksX) + scoreForPeaks(peaksY)) * 0.5)
        .clamp(0.0, 1.0)
        .toDouble();
  }

  int _countProminentPeaks(Float64List values) {
    if (values.length < 3) {
      return 0;
    }

    double sum = 0;
    for (final v in values) {
      sum += v;
    }
    final mean = sum / values.length;
    double variance = 0;
    for (final v in values) {
      final d = v - mean;
      variance += d * d;
    }
    variance /= values.length;
    final std = math.sqrt(variance);
    final threshold = mean + (std * 0.55);
    final minPeakDistance = math.max(2, values.length ~/ 12);

    int count = 0;
    int lastPeak = -minPeakDistance;
    for (int i = 1; i < values.length - 1; i++) {
      final isPeak =
          values[i] >= threshold &&
          values[i] >= values[i - 1] &&
          values[i] >= values[i + 1];
      if (!isPeak) {
        continue;
      }
      if (i - lastPeak < minPeakDistance) {
        continue;
      }
      lastPeak = i;
      count += 1;
    }
    return count;
  }

  double _frameEdgeScore({
    required _RectCandidate rect,
    required _GradientMaps gradients,
    required int width,
    required int height,
  }) {
    final rectWidth = rect.right - rect.left + 1;
    final rectHeight = rect.bottom - rect.top + 1;
    if (rectWidth < 20 || rectHeight < 20) {
      return 0.0;
    }

    final border = math.max(1, (math.min(rectWidth, rectHeight) * 0.05).round());
    double borderSum = 0;
    int borderCount = 0;
    double innerSum = 0;
    int innerCount = 0;

    for (int y = rect.top; y <= rect.bottom; y++) {
      final rowOffset = y * width;
      for (int x = rect.left; x <= rect.right; x++) {
        final idx = rowOffset + x;
        final value = gradients.dx[idx] + gradients.dy[idx];
        final isBorder =
            x < rect.left + border ||
            x > rect.right - border ||
            y < rect.top + border ||
            y > rect.bottom - border;
        if (isBorder) {
          borderSum += value;
          borderCount += 1;
        } else {
          innerSum += value;
          innerCount += 1;
        }
      }
    }

    if (borderCount == 0 || innerCount == 0) {
      return 0.0;
    }

    final borderMean = borderSum / borderCount;
    final innerMean = innerSum / innerCount;
    final ratio = borderMean / math.max(1e-6, innerMean);
    return ((ratio - 0.90) / 0.90).clamp(0.0, 1.0).toDouble();
  }

  double _overlapRatioWithContent({
    required _Bounds bounds,
    required _SearchBounds content,
  }) {
    final overlapLeft = math.max(bounds.left, content.left.toDouble());
    final overlapTop = math.max(bounds.top, content.top.toDouble());
    final overlapRight = math.min(bounds.right, content.right.toDouble());
    final overlapBottom = math.min(bounds.bottom, content.bottom.toDouble());
    if (overlapRight < overlapLeft || overlapBottom < overlapTop) {
      return 0.0;
    }
    final overlapArea =
        (overlapRight - overlapLeft + 1) * (overlapBottom - overlapTop + 1);
    final boundsArea = (bounds.right - bounds.left + 1) * (bounds.bottom - bounds.top + 1);
    if (boundsArea <= 1e-6) {
      return 0.0;
    }
    return (overlapArea / boundsArea).clamp(0.0, 1.0).toDouble();
  }

  Float64List _buildIntegralUint8({
    required Uint8List values,
    required int width,
    required int height,
  }) {
    final stride = width + 1;
    final integral = Float64List((width + 1) * (height + 1));
    for (int y = 1; y <= height; y++) {
      double rowSum = 0;
      final srcRow = (y - 1) * width;
      final rowIndex = y * stride;
      final prevRowIndex = (y - 1) * stride;
      for (int x = 1; x <= width; x++) {
        rowSum += values[srcRow + x - 1];
        integral[rowIndex + x] = integral[prevRowIndex + x] + rowSum;
      }
    }
    return integral;
  }

  double _rectMeanFromIntegral({
    required Float64List integral,
    required int width,
    required int left,
    required int top,
    required int right,
    required int bottom,
  }) {
    final area = ((right - left + 1) * (bottom - top + 1)).toDouble();
    if (area <= 0) {
      return 0;
    }
    return _rectSum(
          integral: integral,
          width: width,
          left: left,
          top: top,
          right: right,
          bottom: bottom,
        ) /
        area;
  }

  _QuadCandidate _snapQuadToOuterEdges({
    required _QuadCandidate quad,
    required _GradientMaps gradients,
    required int width,
    required int height,
  }) {
    final bounds = _quadBounds(quad);
    final left = bounds.left.floor().clamp(0, width - 1);
    final top = bounds.top.floor().clamp(0, height - 1);
    final right = bounds.right.ceil().clamp(0, width - 1);
    final bottom = bounds.bottom.ceil().clamp(0, height - 1);
    final rectWidth = right - left + 1;
    final rectHeight = bottom - top + 1;
    if (rectWidth < 18 || rectHeight < 18) {
      return quad;
    }

    final searchX = math.max(3, (rectWidth * 0.28).round());
    final searchY = math.max(3, (rectHeight * 0.28).round());

    final snappedLeft = _bestVerticalEdgeX(
      gradients: gradients,
      width: width,
      height: height,
      top: top,
      bottom: bottom,
      targetX: left,
      searchRadius: searchX,
    );
    final snappedRight = _bestVerticalEdgeX(
      gradients: gradients,
      width: width,
      height: height,
      top: top,
      bottom: bottom,
      targetX: right,
      searchRadius: searchX,
    );
    if (snappedRight - snappedLeft < 14) {
      return quad;
    }

    final snappedTop = _bestHorizontalEdgeY(
      gradients: gradients,
      width: width,
      height: height,
      left: snappedLeft,
      right: snappedRight,
      targetY: top,
      searchRadius: searchY,
    );
    final snappedBottom = _bestHorizontalEdgeY(
      gradients: gradients,
      width: width,
      height: height,
      left: snappedLeft,
      right: snappedRight,
      targetY: bottom,
      searchRadius: searchY,
    );
    if (snappedBottom - snappedTop < 14) {
      return quad;
    }

    final snappedRect = _RectCandidate(
      left: snappedLeft,
      top: snappedTop,
      right: snappedRight,
      bottom: snappedBottom,
    );
    final oldRect = _RectCandidate(
      left: left,
      top: top,
      right: right,
      bottom: bottom,
    );
    final oldFrameScore = _frameEdgeScore(
      rect: oldRect,
      gradients: gradients,
      width: width,
      height: height,
    );
    final snappedFrameScore = _frameEdgeScore(
      rect: snappedRect,
      gradients: gradients,
      width: width,
      height: height,
    );
    if (snappedFrameScore + 0.02 < oldFrameScore) {
      return quad;
    }

    final oldArea = math.max(
      1.0,
      (oldRect.right - oldRect.left + 1) * (oldRect.bottom - oldRect.top + 1).toDouble(),
    );
    final newArea = math.max(
      1.0,
      (snappedRect.right - snappedRect.left + 1) *
          (snappedRect.bottom - snappedRect.top + 1).toDouble(),
    );
    if (newArea < oldArea * 0.90) {
      return quad;
    }

    final oldW = math.max(1e-6, bounds.right - bounds.left);
    final oldH = math.max(1e-6, bounds.bottom - bounds.top);
    final newW = math.max(1e-6, (snappedRect.right - snappedRect.left).toDouble());
    final newH = math.max(1e-6, (snappedRect.bottom - snappedRect.top).toDouble());

    _PointD remap(_PointD p) {
      final nx =
          snappedRect.left +
          ((p.x - bounds.left) / oldW).clamp(-0.25, 1.25) * newW;
      final ny =
          snappedRect.top +
          ((p.y - bounds.top) / oldH).clamp(-0.25, 1.25) * newH;
      return _PointD(
        nx.clamp(0.0, (width - 1).toDouble()),
        ny.clamp(0.0, (height - 1).toDouble()),
      );
    }

    final snappedQuad = _orderQuad(<_PointD>[
      remap(quad.tl),
      remap(quad.tr),
      remap(quad.br),
      remap(quad.bl),
    ]);
    final minArea = math.max(50.0, _quadArea(quad) * 0.55);
    if (!_isValidQuad(snappedQuad, minArea: minArea)) {
      return quad;
    }
    return snappedQuad;
  }

  int _bestVerticalEdgeX({
    required _GradientMaps gradients,
    required int width,
    required int height,
    required int top,
    required int bottom,
    required int targetX,
    required int searchRadius,
  }) {
    final y0 = top.clamp(1, height - 2);
    final y1 = bottom.clamp(1, height - 2);
    if (y1 <= y0) {
      return targetX.clamp(0, width - 1);
    }

    final start = (targetX - searchRadius).clamp(1, width - 2);
    final end = (targetX + searchRadius).clamp(1, width - 2);
    if (end < start) {
      return targetX.clamp(0, width - 1);
    }

    int bestX = targetX.clamp(start, end);
    double bestScore = double.negativeInfinity;
    final distanceNorm = math.max(1.0, searchRadius.toDouble());
    for (int x = start; x <= end; x++) {
      double sum = 0;
      for (int y = y0; y <= y1; y++) {
        final idx = (y * width) + x;
        sum += gradients.dx[idx] + (gradients.dy[idx] * 0.20);
      }
      final mean = sum / (y1 - y0 + 1);
      final drift = (x - targetX).abs() / distanceNorm;
      final driftPenalty = 1.0 - (drift * 0.10).clamp(0.0, 0.10);
      final borderDistance = math.min(x, (width - 1) - x).toDouble();
      final borderPenalty = borderDistance <= 1
          ? 0.72
          : borderDistance <= 3
          ? 0.86
          : 1.0;
      final score = mean * driftPenalty * borderPenalty;
      if (score > bestScore) {
        bestScore = score;
        bestX = x;
      }
    }
    return bestX;
  }

  int _bestHorizontalEdgeY({
    required _GradientMaps gradients,
    required int width,
    required int height,
    required int left,
    required int right,
    required int targetY,
    required int searchRadius,
  }) {
    final x0 = left.clamp(1, width - 2);
    final x1 = right.clamp(1, width - 2);
    if (x1 <= x0) {
      return targetY.clamp(0, height - 1);
    }

    final start = (targetY - searchRadius).clamp(1, height - 2);
    final end = (targetY + searchRadius).clamp(1, height - 2);
    if (end < start) {
      return targetY.clamp(0, height - 1);
    }

    int bestY = targetY.clamp(start, end);
    double bestScore = double.negativeInfinity;
    final distanceNorm = math.max(1.0, searchRadius.toDouble());
    for (int y = start; y <= end; y++) {
      double sum = 0;
      final rowOffset = y * width;
      for (int x = x0; x <= x1; x++) {
        final idx = rowOffset + x;
        sum += gradients.dy[idx] + (gradients.dx[idx] * 0.20);
      }
      final mean = sum / (x1 - x0 + 1);
      final drift = (y - targetY).abs() / distanceNorm;
      final driftPenalty = 1.0 - (drift * 0.10).clamp(0.0, 0.10);
      final borderDistance = math.min(y, (height - 1) - y).toDouble();
      final borderPenalty = borderDistance <= 1
          ? 0.72
          : borderDistance <= 3
          ? 0.86
          : 1.0;
      final score = mean * driftPenalty * borderPenalty;
      if (score > bestScore) {
        bestScore = score;
        bestY = y;
      }
    }
    return bestY;
  }

  _WindowCandidate? _searchBestCandidateCoarseFine({
    required int width,
    required int height,
    required int minSide,
    required int maxSide,
    required int minDim,
    required Float64List integralX,
    required Float64List integralY,
    required Float64List rowPrefix,
    required Float64List colPrefix,
    required int rowLength,
    required int colLength,
    required _AxisBand rowBand,
    required _AxisBand colBand,
    required _SearchBounds searchBounds,
  }) {
    final maxLeft = width - minSide;
    final maxTop = height - minSide;
    final boundedLeftStart = searchBounds.left
        .clamp(0, math.max(0, maxLeft))
        .toInt();
    final boundedTopStart = searchBounds.top
        .clamp(0, math.max(0, maxTop))
        .toInt();

    _WindowCandidate? coarseBest;
    final coarseSideStep = math.max(4, (minDim * 0.10).round());
    for (int side = minSide; side <= maxSide; side += coarseSideStep) {
      final localLeftEnd = math.min(searchBounds.right - side + 1, width - side);
      final localTopEnd = math.min(searchBounds.bottom - side + 1, height - side);
      if (localLeftEnd < boundedLeftStart || localTopEnd < boundedTopStart) {
        continue;
      }
      final candidate = _bestCandidateForSide(
        width: width,
        height: height,
        side: side,
        minDim: minDim,
        integralX: integralX,
        integralY: integralY,
        rowPrefix: rowPrefix,
        colPrefix: colPrefix,
        rowLength: rowLength,
        colLength: colLength,
        rowBand: rowBand,
        colBand: colBand,
        leftStart: boundedLeftStart,
        leftEnd: localLeftEnd,
        topStart: boundedTopStart,
        topEnd: localTopEnd,
        step: math.max(2, (side * 0.18).round()),
      );
      if (candidate == null) {
        continue;
      }
      if (coarseBest == null || candidate.score > coarseBest.score) {
        coarseBest = candidate;
      }
    }

    if (coarseBest == null) {
      return null;
    }

    _WindowCandidate refinedBest = coarseBest;
    final sideRadius = math.max(8, (coarseBest.side * 0.25).round());
    final refineMinSide = math.max(minSide, coarseBest.side - sideRadius);
    final refineMaxSide = math.min(maxSide, coarseBest.side + sideRadius);
    final refineSideStep = math.max(1, (minDim * 0.02).round());

    for (
      int side = refineMinSide;
      side <= refineMaxSide;
      side += refineSideStep
    ) {
      final localLeftEnd = math.min(searchBounds.right - side + 1, width - side);
      final localTopEnd = math.min(searchBounds.bottom - side + 1, height - side);
      if (localLeftEnd < boundedLeftStart || localTopEnd < boundedTopStart) {
        continue;
      }
      final positionRadius = math.max(8, (side * 0.25).round());
      final candidate = _bestCandidateForSide(
        width: width,
        height: height,
        side: side,
        minDim: minDim,
        integralX: integralX,
        integralY: integralY,
        rowPrefix: rowPrefix,
        colPrefix: colPrefix,
        rowLength: rowLength,
        colLength: colLength,
        rowBand: rowBand,
        colBand: colBand,
        leftStart: math.max(
          boundedLeftStart,
          coarseBest.left - positionRadius,
        ),
        leftEnd: math.min(localLeftEnd, coarseBest.left + positionRadius),
        topStart: math.max(boundedTopStart, coarseBest.top - positionRadius),
        topEnd: math.min(localTopEnd, coarseBest.top + positionRadius),
        step: math.max(1, (side * 0.05).round()),
      );
      if (candidate == null) {
        continue;
      }
      if (candidate.score > refinedBest.score) {
        refinedBest = candidate;
      }
    }

    return _microRefineCandidate(
      base: refinedBest,
      width: width,
      height: height,
      minSide: minSide,
      maxSide: maxSide,
      minDim: minDim,
      integralX: integralX,
      integralY: integralY,
      rowBand: rowBand,
      colBand: colBand,
      rowPrefix: rowPrefix,
      colPrefix: colPrefix,
      rowLength: rowLength,
      colLength: colLength,
      searchBounds: searchBounds,
    );
  }

  _WindowCandidate _microRefineCandidate({
    required _WindowCandidate base,
    required int width,
    required int height,
    required int minSide,
    required int maxSide,
    required int minDim,
    required Float64List integralX,
    required Float64List integralY,
    required Float64List rowPrefix,
    required Float64List colPrefix,
    required int rowLength,
    required int colLength,
    required _AxisBand rowBand,
    required _AxisBand colBand,
    required _SearchBounds searchBounds,
  }) {
    _WindowCandidate best = base;
    final minLocalSide = math.max(minSide, base.side - 3);
    final maxLocalSide = math.min(maxSide, base.side + 3);
    for (int side = minLocalSide; side <= maxLocalSide; side++) {
      final localLeftEnd = math.min(searchBounds.right - side + 1, width - side);
      final localTopEnd = math.min(searchBounds.bottom - side + 1, height - side);
      if (localLeftEnd < searchBounds.left || localTopEnd < searchBounds.top) {
        continue;
      }
      final leftStart = math.max(searchBounds.left, best.left - 4);
      final leftEnd = math.min(localLeftEnd, best.left + 4);
      final topStart = math.max(searchBounds.top, best.top - 4);
      final topEnd = math.min(localTopEnd, best.top + 4);
      if (leftStart > leftEnd || topStart > topEnd) {
        continue;
      }

      for (int top = topStart; top <= topEnd; top++) {
        for (int left = leftStart; left <= leftEnd; left++) {
          final score = _scoreWindow(
            width: width,
            height: height,
            left: left,
            top: top,
            side: side,
            minDim: minDim,
            integralX: integralX,
            integralY: integralY,
            rowPrefix: rowPrefix,
            colPrefix: colPrefix,
            rowLength: rowLength,
            colLength: colLength,
            rowBand: rowBand,
            colBand: colBand,
          );
          if (score > best.score) {
            best = _WindowCandidate(
              left: left,
              top: top,
              side: side,
              score: score,
            );
          }
        }
      }
    }
    return best;
  }

  _WindowCandidate? _bestCandidateForSide({
    required int width,
    required int height,
    required int side,
    required int minDim,
    required Float64List integralX,
    required Float64List integralY,
    required Float64List rowPrefix,
    required Float64List colPrefix,
    required int rowLength,
    required int colLength,
    required _AxisBand rowBand,
    required _AxisBand colBand,
    required int leftStart,
    required int leftEnd,
    required int topStart,
    required int topEnd,
    required int step,
  }) {
    if (leftStart > leftEnd || topStart > topEnd) {
      return null;
    }
    _WindowCandidate? best;

    _WindowCandidate? evalRange({
      required int x0,
      required int x1,
      required int y0,
      required int y1,
      required int s,
    }) {
      if (x0 > x1 || y0 > y1) {
        return null;
      }
      _WindowCandidate? localBest;
      for (int top = y0; top <= y1; top += s) {
        for (int left = x0; left <= x1; left += s) {
          final score = _scoreWindow(
            width: width,
            height: height,
            left: left,
            top: top,
            side: side,
            minDim: minDim,
            integralX: integralX,
            integralY: integralY,
            rowPrefix: rowPrefix,
            colPrefix: colPrefix,
            rowLength: rowLength,
            colLength: colLength,
            rowBand: rowBand,
            colBand: colBand,
          );
          final candidate = _WindowCandidate(
            left: left,
            top: top,
            side: side,
            score: score,
          );
          if (localBest == null || candidate.score > localBest.score) {
            localBest = candidate;
          }
        }
      }
      return localBest;
    }

    final preferredLeft = _bestAxisStart(
      prefix: colPrefix,
      windowSize: side,
      limit: width,
    );
    final preferredTop = _bestAxisStart(
      prefix: rowPrefix,
      windowSize: side,
      limit: height,
    );
    final focusRadius = math.max(8, (side * 0.26).round());
    final focusLeftStart = math.max(leftStart, preferredLeft - focusRadius);
    final focusLeftEnd = math.min(leftEnd, preferredLeft + focusRadius);
    final focusTopStart = math.max(topStart, preferredTop - focusRadius);
    final focusTopEnd = math.min(topEnd, preferredTop + focusRadius);

    best = evalRange(
      x0: focusLeftStart,
      x1: focusLeftEnd,
      y0: focusTopStart,
      y1: focusTopEnd,
      s: step,
    );

    final wideBest = evalRange(
      x0: leftStart,
      x1: leftEnd,
      y0: topStart,
      y1: topEnd,
      s: math.max(step, 2),
    );

    if (wideBest != null && (best == null || wideBest.score > best.score)) {
      best = wideBest;
    }
    return best;
  }

  _RectCandidate _refineCandidateByEnergyBands({
    required _WindowCandidate candidate,
    required _GradientMaps gradients,
    required int width,
    required int height,
  }) {
    final left = candidate.left.clamp(0, width - 1);
    final top = candidate.top.clamp(0, height - 1);
    final right = (candidate.left + candidate.side - 1).clamp(0, width - 1);
    final bottom = (candidate.top + candidate.side - 1).clamp(0, height - 1);
    if (right <= left || bottom <= top) {
      return _RectCandidate(left: left, top: top, right: right, bottom: bottom);
    }

    final profileX = _profileEnergyX(
      gradients: gradients,
      width: width,
      left: left,
      right: right,
      top: top,
      bottom: bottom,
    );
    final profileY = _profileEnergyY(
      gradients: gradients,
      width: width,
      left: left,
      right: right,
      top: top,
      bottom: bottom,
    );

    final intervalX = _bestEnergyInterval(profileX);
    final intervalY = _bestEnergyInterval(profileY);

    final candidateRight = candidate.left + candidate.side - 1;
    final candidateBottom = candidate.top + candidate.side - 1;
    final maxTrim = math.max(4, (candidate.side * 0.10).round());
    final minLeftAllowed = math.min(width - 1, candidate.left + maxTrim);
    final maxRightAllowed = math.max(0, candidateRight - maxTrim);
    final minTopAllowed = math.min(height - 1, candidate.top + maxTrim);
    final maxBottomAllowed = math.max(0, candidateBottom - maxTrim);

    final refinedLeft = (left + intervalX.start).clamp(0, width - 1).toInt();
    final refinedTop = (top + intervalY.start).clamp(0, height - 1).toInt();
    final refinedRight = (left + intervalX.end).clamp(0, width - 1).toInt();
    final refinedBottom = (top + intervalY.end).clamp(0, height - 1).toInt();

    final leftClamped = math.min(refinedLeft, minLeftAllowed);
    final topClamped = math.min(refinedTop, minTopAllowed);
    final rightClamped = math.max(refinedRight, maxRightAllowed);
    final bottomClamped = math.max(refinedBottom, maxBottomAllowed);

    return _RectCandidate(
      left: leftClamped.clamp(0, width - 1).toInt(),
      top: topClamped.clamp(0, height - 1).toInt(),
      right: rightClamped.clamp(0, width - 1).toInt(),
      bottom: bottomClamped.clamp(0, height - 1).toInt(),
    );
  }

  Float64List _profileEnergyX({
    required _GradientMaps gradients,
    required int width,
    required int left,
    required int right,
    required int top,
    required int bottom,
  }) {
    final len = right - left + 1;
    final out = Float64List(len);
    final rowCount = (bottom - top + 1).toDouble();
    for (int x = left; x <= right; x++) {
      double sum = 0;
      for (int y = top; y <= bottom; y++) {
        final idx = (y * width) + x;
        sum += gradients.dx[idx] + gradients.dy[idx];
      }
      out[x - left] = sum / rowCount;
    }
    return _smooth1D(out, radius: math.max(2, len ~/ 24));
  }

  Float64List _profileEnergyY({
    required _GradientMaps gradients,
    required int width,
    required int left,
    required int right,
    required int top,
    required int bottom,
  }) {
    final len = bottom - top + 1;
    final out = Float64List(len);
    final colCount = (right - left + 1).toDouble();
    for (int y = top; y <= bottom; y++) {
      double sum = 0;
      final rowOffset = y * width;
      for (int x = left; x <= right; x++) {
        final idx = rowOffset + x;
        sum += gradients.dx[idx] + gradients.dy[idx];
      }
      out[y - top] = sum / colCount;
    }
    return _smooth1D(out, radius: math.max(2, len ~/ 24));
  }

  Float64List _smooth1D(Float64List values, {required int radius}) {
    final n = values.length;
    if (n == 0 || radius <= 0) {
      return values;
    }
    final prefix = Float64List(n + 1);
    for (int i = 0; i < n; i++) {
      prefix[i + 1] = prefix[i] + values[i];
    }
    final out = Float64List(n);
    for (int i = 0; i < n; i++) {
      final start = math.max(0, i - radius);
      final end = math.min(n - 1, i + radius);
      out[i] = (prefix[end + 1] - prefix[start]) / (end - start + 1);
    }
    return out;
  }

  _Interval _bestEnergyInterval(Float64List profile) {
    final n = profile.length;
    if (n <= 2) {
      return _Interval(0, math.max(0, n - 1));
    }

    final minLen = math.max(8, (n * 0.50).round());
    final maxLen = n;
    final center = (n - 1) / 2;
    final half = math.max(1.0, n / 2);

    final prefix = Float64List(n + 1);
    for (int i = 0; i < n; i++) {
      prefix[i + 1] = prefix[i] + profile[i];
    }
    final globalMean = prefix[n] / n;

    _Interval best = _Interval(0, n - 1);
    double bestScore = double.negativeInfinity;
    double bestMean = globalMean;
    double bestGain = 0;

    for (int start = 0; start <= n - minLen; start++) {
      final maxEnd = math.min(n - 1, start + maxLen - 1);
      for (int end = start + minLen - 1; end <= maxEnd; end++) {
        final len = end - start + 1;
        final mean = (prefix[end + 1] - prefix[start]) / len;
        final gain = mean - (globalMean * 0.92);
        final sizePrior = math.sqrt(len / n);
        final intervalCenter = (start + end) / 2;
        final normDist = ((intervalCenter - center).abs() / half)
            .clamp(0.0, 1.0)
            .toDouble();
        final centerPrior = 1.0 - 0.20 * normDist;
        final score = gain * sizePrior * centerPrior;
        if (score > bestScore) {
          bestScore = score;
          best = _Interval(start, end);
          bestMean = mean;
          bestGain = gain;
        }
      }
    }

    if (bestGain < globalMean * 0.02) {
      return _Interval(0, n - 1);
    }
    final bestLen = best.end - best.start + 1;
    if (bestLen < (n * 0.55) && bestMean < globalMean * 1.15) {
      return _Interval(0, n - 1);
    }
    return best;
  }

  _QuadCandidate _refineToQuadrilateral({
    required _RectCandidate rect,
    required _GradientMaps gradients,
    required int width,
    required int height,
  }) {
    final rectQuad = _rectToQuad(rect);
    final rectWidth = rect.right - rect.left + 1;
    final rectHeight = rect.bottom - rect.top + 1;
    if (rectWidth < 12 || rectHeight < 12) {
      return rectQuad;
    }

    final marginX = math.max(2, (rectWidth * 0.14).round());
    final marginY = math.max(2, (rectHeight * 0.14).round());
    final left = math.max(0, rect.left - marginX);
    final top = math.max(0, rect.top - marginY);
    final right = math.min(width - 1, rect.right + marginX);
    final bottom = math.min(height - 1, rect.bottom + marginY);

    double sum = 0;
    double sumSq = 0;
    double maxV = 0;
    int count = 0;
    for (int y = top; y <= bottom; y++) {
      final rowOffset = y * width;
      for (int x = left; x <= right; x++) {
        final idx = rowOffset + x;
        final v = gradients.dx[idx] + gradients.dy[idx];
        sum += v;
        sumSq += v * v;
        if (v > maxV) {
          maxV = v;
        }
        count += 1;
      }
    }
    if (count <= 0 || maxV <= 1e-6) {
      return rectQuad;
    }

    final mean = sum / count;
    final variance = math.max(0.0, (sumSq / count) - (mean * mean));
    final std = math.sqrt(variance);

    final pointsPrimary = _collectEdgePoints(
      gradients: gradients,
      width: width,
      left: left,
      top: top,
      right: right,
      bottom: bottom,
      threshold: math.max(mean + std * 0.75, maxV * 0.30),
    );
    final points = pointsPrimary.length >= 20
        ? pointsPrimary
        : _collectEdgePoints(
            gradients: gradients,
            width: width,
            left: left,
            top: top,
            right: right,
            bottom: bottom,
            threshold: math.max(mean + std * 0.35, maxV * 0.14),
          );
    if (points.length < 12) {
      return rectQuad;
    }

    final tl = _extremePoint(points, (p) => -(p.x + p.y));
    final tr = _extremePoint(points, (p) => p.x - p.y);
    final br = _extremePoint(points, (p) => p.x + p.y);
    final bl = _extremePoint(points, (p) => p.y - p.x);

    final searchRadius = math.max(6.0, math.min(rectWidth, rectHeight) * 0.24);
    final refinedTl = _refineCornerFromPoints(
      seed: tl,
      points: points,
      radius: searchRadius,
    );
    final refinedTr = _refineCornerFromPoints(
      seed: tr,
      points: points,
      radius: searchRadius,
    );
    final refinedBr = _refineCornerFromPoints(
      seed: br,
      points: points,
      radius: searchRadius,
    );
    final refinedBl = _refineCornerFromPoints(
      seed: bl,
      points: points,
      radius: searchRadius,
    );

    final ordered = _orderQuad(<_PointD>[
      refinedTl,
      refinedTr,
      refinedBr,
      refinedBl,
    ]);
    if (!_isValidQuad(ordered, minArea: rectWidth * rectHeight * 0.16)) {
      return rectQuad;
    }
    final confidenceOrdered = _quadConfidence(
      quad: ordered,
      rect: rect,
      width: width,
      height: height,
    );
    if (confidenceOrdered < 0.24) {
      return rectQuad;
    }

    final expanded = _expandQuadFromCenter(
      quad: ordered,
      factor: 1.02,
      maxX: (width - 1).toDouble(),
      maxY: (height - 1).toDouble(),
    );
    if (!_isValidQuad(expanded, minArea: rectWidth * rectHeight * 0.22)) {
      return ordered;
    }
    final confidenceExpanded = _quadConfidence(
      quad: expanded,
      rect: rect,
      width: width,
      height: height,
    );
    if (confidenceExpanded + 0.03 < confidenceOrdered) {
      return ordered;
    }
    if (confidenceExpanded < 0.26) {
      return ordered;
    }
    return expanded;
  }

  List<_EdgePoint> _collectEdgePoints({
    required _GradientMaps gradients,
    required int width,
    required int left,
    required int top,
    required int right,
    required int bottom,
    required double threshold,
  }) {
    final points = <_EdgePoint>[];
    for (int y = top; y <= bottom; y++) {
      final rowOffset = y * width;
      for (int x = left; x <= right; x++) {
        final idx = rowOffset + x;
        final weight = gradients.dx[idx] + gradients.dy[idx];
        if (weight >= threshold) {
          points.add(_EdgePoint(x: x.toDouble(), y: y.toDouble(), w: weight));
        }
      }
    }
    return points;
  }

  _PointD _extremePoint(
    List<_EdgePoint> points,
    double Function(_PointD) scoreFn,
  ) {
    _EdgePoint best = points.first;
    double bestScore = scoreFn(_PointD(best.x, best.y));
    for (int i = 1; i < points.length; i++) {
      final p = points[i];
      final s = scoreFn(_PointD(p.x, p.y));
      if (s > bestScore) {
        bestScore = s;
        best = p;
      }
    }
    return _PointD(best.x, best.y);
  }

  _PointD _refineCornerFromPoints({
    required _PointD seed,
    required List<_EdgePoint> points,
    required double radius,
  }) {
    double cx = seed.x;
    double cy = seed.y;
    double currentRadius = radius;
    for (int iter = 0; iter < 2; iter++) {
      final r2 = currentRadius * currentRadius;
      double sumW = 0;
      double sumX = 0;
      double sumY = 0;
      for (final p in points) {
        final dx = p.x - cx;
        final dy = p.y - cy;
        final d2 = dx * dx + dy * dy;
        if (d2 > r2) {
          continue;
        }
        final weight = p.w / (1.0 + d2 * 0.03);
        sumW += weight;
        sumX += p.x * weight;
        sumY += p.y * weight;
      }
      if (sumW > 1e-6) {
        cx = sumX / sumW;
        cy = sumY / sumW;
      }
      currentRadius *= 0.60;
    }
    return _PointD(cx, cy);
  }

  _QuadCandidate _rectToQuad(_RectCandidate rect) {
    return _QuadCandidate(
      tl: _PointD(rect.left.toDouble(), rect.top.toDouble()),
      tr: _PointD(rect.right.toDouble(), rect.top.toDouble()),
      br: _PointD(rect.right.toDouble(), rect.bottom.toDouble()),
      bl: _PointD(rect.left.toDouble(), rect.bottom.toDouble()),
    );
  }

  _QuadCandidate _orderQuad(List<_PointD> points) {
    if (points.length != 4) {
      throw ArgumentError('Expected exactly 4 points');
    }
    final sums = points.map((p) => p.x + p.y).toList(growable: false);
    final diffs = points.map((p) => p.x - p.y).toList(growable: false);

    int idxTl = 0;
    int idxBr = 0;
    int idxTr = 0;
    int idxBl = 0;
    for (int i = 1; i < 4; i++) {
      if (sums[i] < sums[idxTl]) {
        idxTl = i;
      }
      if (sums[i] > sums[idxBr]) {
        idxBr = i;
      }
      if (diffs[i] > diffs[idxTr]) {
        idxTr = i;
      }
      if (diffs[i] < diffs[idxBl]) {
        idxBl = i;
      }
    }
    final unique = <int>{idxTl, idxTr, idxBr, idxBl};
    if (unique.length == 4) {
      return _QuadCandidate(
        tl: points[idxTl],
        tr: points[idxTr],
        br: points[idxBr],
        bl: points[idxBl],
      );
    }

    final centroid = _PointD(
      (points[0].x + points[1].x + points[2].x + points[3].x) / 4.0,
      (points[0].y + points[1].y + points[2].y + points[3].y) / 4.0,
    );
    final sorted = List<_PointD>.from(points)
      ..sort((a, b) {
        final aa = math.atan2(a.y - centroid.y, a.x - centroid.x);
        final bb = math.atan2(b.y - centroid.y, b.x - centroid.x);
        return aa.compareTo(bb);
      });
    int tlIndex = 0;
    double bestSum = sorted.first.x + sorted.first.y;
    for (int i = 1; i < 4; i++) {
      final s = sorted[i].x + sorted[i].y;
      if (s < bestSum) {
        bestSum = s;
        tlIndex = i;
      }
    }
    final rotated = <_PointD>[
      sorted[tlIndex],
      sorted[(tlIndex + 1) % 4],
      sorted[(tlIndex + 2) % 4],
      sorted[(tlIndex + 3) % 4],
    ];
    final secondIsBottom = rotated[1].y > rotated[3].y;
    if (secondIsBottom) {
      return _QuadCandidate(
        tl: rotated[0],
        tr: rotated[3],
        br: rotated[2],
        bl: rotated[1],
      );
    }
    return _QuadCandidate(
      tl: rotated[0],
      tr: rotated[1],
      br: rotated[2],
      bl: rotated[3],
    );
  }

  bool _isValidQuad(_QuadCandidate quad, {required double minArea}) {
    final area = _quadArea(quad);
    if (area < minArea) {
      return false;
    }
    final sides = <double>[
      _pointDistance(quad.tl, quad.tr),
      _pointDistance(quad.tr, quad.br),
      _pointDistance(quad.br, quad.bl),
      _pointDistance(quad.bl, quad.tl),
    ];
    final minSide = sides.reduce(math.min);
    if (minSide < 6.0) {
      return false;
    }
    final diag1 = _pointDistance(quad.tl, quad.br);
    final diag2 = _pointDistance(quad.tr, quad.bl);
    if (diag1 < 8.0 || diag2 < 8.0) {
      return false;
    }
    final avgHoriz = (sides[0] + sides[2]) * 0.5;
    final avgVert = (sides[1] + sides[3]) * 0.5;
    final aspect =
        math.max(avgHoriz, avgVert) /
        math.max(1e-6, math.min(avgHoriz, avgVert));
    if (aspect > 1.45) {
      return false;
    }
    final oppRatioH =
        math.max(sides[0], sides[2]) /
        math.max(1e-6, math.min(sides[0], sides[2]));
    final oppRatioV =
        math.max(sides[1], sides[3]) /
        math.max(1e-6, math.min(sides[1], sides[3]));
    if (oppRatioH > 1.85 || oppRatioV > 1.85) {
      return false;
    }
    return _isConvexQuad(quad);
  }

  double _quadConfidence({
    required _QuadCandidate quad,
    required _RectCandidate rect,
    required int width,
    required int height,
  }) {
    final rectArea = math.max(
      1.0,
      (rect.right - rect.left + 1) * (rect.bottom - rect.top + 1).toDouble(),
    );
    final quadArea = _quadArea(quad);
    final areaRatio = quadArea / rectArea;

    final rectCx = (rect.left + rect.right) * 0.5;
    final rectCy = (rect.top + rect.bottom) * 0.5;
    final quadCx = (quad.tl.x + quad.tr.x + quad.br.x + quad.bl.x) * 0.25;
    final quadCy = (quad.tl.y + quad.tr.y + quad.br.y + quad.bl.y) * 0.25;
    final shift = math.sqrt(
      (quadCx - rectCx) * (quadCx - rectCx) +
          (quadCy - rectCy) * (quadCy - rectCy),
    );
    final rectDiag = math.sqrt(
      (rect.right - rect.left + 1) * (rect.right - rect.left + 1) +
          (rect.bottom - rect.top + 1) * (rect.bottom - rect.top + 1),
    );
    final shiftRatio = shift / math.max(1e-6, rectDiag);

    final topTilt =
        (quad.tr.y - quad.tl.y).abs() /
        math.max(1e-6, (quad.tr.x - quad.tl.x).abs());
    final botTilt =
        (quad.br.y - quad.bl.y).abs() /
        math.max(1e-6, (quad.br.x - quad.bl.x).abs());
    final tilt = math.max(topTilt, botTilt);

    final bounds = _quadBounds(quad);
    final borderTouchCount = <bool>[
      bounds.left <= 1.0,
      bounds.top <= 1.0,
      bounds.right >= (width - 2).toDouble(),
      bounds.bottom >= (height - 2).toDouble(),
    ].where((v) => v).length;

    double score = 1.0;
    if (areaRatio < 0.55) {
      score *= 0.35;
    } else if (areaRatio < 0.70) {
      score *= 0.70;
    }
    if (areaRatio > 1.10) {
      score *= 0.75;
    }
    if (shiftRatio > 0.22) {
      score *= 0.55;
    } else if (shiftRatio > 0.14) {
      score *= 0.78;
    }
    if (tilt > 0.22) {
      score *= 0.52;
    } else if (tilt > 0.14) {
      score *= 0.75;
    }
    if (borderTouchCount >= 3) {
      score *= 0.70;
    }
    return score.clamp(0.0, 1.0).toDouble();
  }

  bool _isConvexQuad(_QuadCandidate q) {
    final pts = <_PointD>[q.tl, q.tr, q.br, q.bl];
    double? sign;
    for (int i = 0; i < 4; i++) {
      final a = pts[i];
      final b = pts[(i + 1) % 4];
      final c = pts[(i + 2) % 4];
      final cross = (b.x - a.x) * (c.y - b.y) - (b.y - a.y) * (c.x - b.x);
      if (cross.abs() < 1e-6) {
        continue;
      }
      final currentSign = cross > 0 ? 1.0 : -1.0;
      sign ??= currentSign;
      if (sign != currentSign) {
        return false;
      }
    }
    return true;
  }

  double _quadArea(_QuadCandidate q) {
    final pts = <_PointD>[q.tl, q.tr, q.br, q.bl];
    double sum = 0;
    for (int i = 0; i < 4; i++) {
      final p = pts[i];
      final n = pts[(i + 1) % 4];
      sum += p.x * n.y - n.x * p.y;
    }
    return sum.abs() * 0.5;
  }

  double _pointDistance(_PointD a, _PointD b) {
    final dx = a.x - b.x;
    final dy = a.y - b.y;
    return math.sqrt(dx * dx + dy * dy);
  }

  _QuadCandidate _expandQuadFromCenter({
    required _QuadCandidate quad,
    required double factor,
    required double maxX,
    required double maxY,
  }) {
    final center = _PointD(
      (quad.tl.x + quad.tr.x + quad.br.x + quad.bl.x) / 4.0,
      (quad.tl.y + quad.tr.y + quad.br.y + quad.bl.y) / 4.0,
    );
    _PointD scaleOut(_PointD p) {
      final nx = center.x + (p.x - center.x) * factor;
      final ny = center.y + (p.y - center.y) * factor;
      return _PointD(nx.clamp(0.0, maxX), ny.clamp(0.0, maxY));
    }

    return _QuadCandidate(
      tl: scaleOut(quad.tl),
      tr: scaleOut(quad.tr),
      br: scaleOut(quad.br),
      bl: scaleOut(quad.bl),
    );
  }

  _QuadCandidate _scaleAndClampQuad({
    required _QuadCandidate quad,
    required double scaleX,
    required double scaleY,
    required int outWidth,
    required int outHeight,
  }) {
    _PointD transform(_PointD p) {
      final x = (p.x * scaleX).clamp(0.0, (outWidth - 1).toDouble());
      final y = (p.y * scaleY).clamp(0.0, (outHeight - 1).toDouble());
      return _PointD(x, y);
    }

    final scaled = _QuadCandidate(
      tl: transform(quad.tl),
      tr: transform(quad.tr),
      br: transform(quad.br),
      bl: transform(quad.bl),
    );
    return _orderQuad(<_PointD>[scaled.tl, scaled.tr, scaled.br, scaled.bl]);
  }

  _Bounds _quadBounds(_QuadCandidate quad) {
    final xs = <double>[quad.tl.x, quad.tr.x, quad.br.x, quad.bl.x];
    final ys = <double>[quad.tl.y, quad.tr.y, quad.br.y, quad.bl.y];
    final minX = xs.reduce(math.min);
    final maxX = xs.reduce(math.max);
    final minY = ys.reduce(math.min);
    final maxY = ys.reduce(math.max);
    return _Bounds(left: minX, top: minY, right: maxX, bottom: maxY);
  }

  double _scoreWindow({
    required int width,
    required int height,
    required int left,
    required int top,
    required int side,
    required int minDim,
    required Float64List integralX,
    required Float64List integralY,
    required Float64List rowPrefix,
    required Float64List colPrefix,
    required int rowLength,
    required int colLength,
    required _AxisBand rowBand,
    required _AxisBand colBand,
  }) {
    final right = left + side - 1;
    final bottom = top + side - 1;
    final insideX = _rectMean(
      integral: integralX,
      width: width,
      left: left,
      top: top,
      right: right,
      bottom: bottom,
    );
    final insideY = _rectMean(
      integral: integralY,
      width: width,
      left: left,
      top: top,
      right: right,
      bottom: bottom,
    );
    final insideEnergy = math.sqrt(math.max(1e-6, insideX * insideY));

    final margin = math.max(2, (side * 0.09).round());
    final outerLeft = math.max(0, left - margin);
    final outerTop = math.max(0, top - margin);
    final outerRight = math.min(width - 1, right + margin);
    final outerBottom = math.min(height - 1, bottom + margin);

    final innerSumX = _rectSum(
      integral: integralX,
      width: width,
      left: left,
      top: top,
      right: right,
      bottom: bottom,
    );
    final innerSumY = _rectSum(
      integral: integralY,
      width: width,
      left: left,
      top: top,
      right: right,
      bottom: bottom,
    );
    final outerSumX = _rectSum(
      integral: integralX,
      width: width,
      left: outerLeft,
      top: outerTop,
      right: outerRight,
      bottom: outerBottom,
    );
    final outerSumY = _rectSum(
      integral: integralY,
      width: width,
      left: outerLeft,
      top: outerTop,
      right: outerRight,
      bottom: outerBottom,
    );
    final outerArea =
        ((outerRight - outerLeft + 1) * (outerBottom - outerTop + 1))
            .toDouble();
    final innerArea = (side * side).toDouble();
    final ringArea = math.max(1.0, outerArea - innerArea);
    final ringX = math.max(0.0, outerSumX - innerSumX) / ringArea;
    final ringY = math.max(0.0, outerSumY - innerSumY) / ringArea;
    final ringEnergy = math.sqrt(math.max(1e-6, ringX * ringY));

    final contrast = insideEnergy - ringEnergy;
    final coverage = side / minDim;

    final centerX = left + (side / 2);
    final centerY = top + (side / 2);
    final normDx = (centerX - (width / 2)).abs() / math.max(1.0, width / 2);
    final normDy = (centerY - (height / 2)).abs() / math.max(1.0, height / 2);
    final centerDistance = math.sqrt(normDx * normDx + normDy * normDy);
    final normalizedDistance = (centerDistance / 1.4)
        .clamp(0.0, 1.0)
        .toDouble();
    final centerPrior = 1.0 - (0.22 * normalizedDistance);
    final bandPrior = _bandPrior(
      left: left,
      top: top,
      side: side,
      rowBand: rowBand,
      colBand: colBand,
    );
    final edgePrior = _edgeConsistencyPrior(
      left: left,
      top: top,
      side: side,
      width: width,
      height: height,
      rowPrefix: rowPrefix,
      colPrefix: colPrefix,
      rowLength: rowLength,
      colLength: colLength,
    );

    final contrastSafe = math.max(0.0, contrast);
    final sizeBoost = math.pow(coverage.clamp(0.0, 1.0), 1.8).toDouble();
    final tooSmallPenalty = coverage < 0.48 ? 0.06 : 1.0;

    double largeCoveragePenalty = 1.0;
    if (coverage > 0.92) {
      largeCoveragePenalty = 0.78;
    } else if (coverage > 0.86) {
      largeCoveragePenalty = 0.88;
    }

    return (insideEnergy * 0.55 + contrastSafe * 1.35) *
        (0.28 + 0.72 * sizeBoost) *
        centerPrior *
        bandPrior *
        edgePrior *
        largeCoveragePenalty *
        tooSmallPenalty;
  }

  double _edgeConsistencyPrior({
    required int left,
    required int top,
    required int side,
    required int width,
    required int height,
    required Float64List rowPrefix,
    required Float64List colPrefix,
    required int rowLength,
    required int colLength,
  }) {
    final right = left + side - 1;
    final bottom = top + side - 1;
    final edgeX = math.max(2, (width * 0.04).round());
    final edgeY = math.max(2, (height * 0.04).round());

    final insideXMean = _rangeMean(
      prefix: colPrefix,
      start: left,
      end: right,
      length: colLength,
    );
    final insideYMean = _rangeMean(
      prefix: rowPrefix,
      start: top,
      end: bottom,
      length: rowLength,
    );

    double penalty = 1.0;
    if (left <= edgeX) {
      final edgeMean = _rangeMean(
        prefix: colPrefix,
        start: 0,
        end: edgeX - 1,
        length: colLength,
      );
      final ratio = edgeMean / math.max(1e-6, insideXMean);
      if (ratio < 0.78) {
        penalty *= 0.78;
      }
    }
    if (right >= width - 1 - edgeX) {
      final edgeMean = _rangeMean(
        prefix: colPrefix,
        start: width - edgeX,
        end: width - 1,
        length: colLength,
      );
      final ratio = edgeMean / math.max(1e-6, insideXMean);
      if (ratio < 0.78) {
        penalty *= 0.78;
      }
    }
    if (top <= edgeY) {
      final edgeMean = _rangeMean(
        prefix: rowPrefix,
        start: 0,
        end: edgeY - 1,
        length: rowLength,
      );
      final ratio = edgeMean / math.max(1e-6, insideYMean);
      if (ratio < 0.78) {
        penalty *= 0.80;
      }
    }
    if (bottom >= height - 1 - edgeY) {
      final edgeMean = _rangeMean(
        prefix: rowPrefix,
        start: height - edgeY,
        end: height - 1,
        length: rowLength,
      );
      final ratio = edgeMean / math.max(1e-6, insideYMean);
      if (ratio < 0.78) {
        penalty *= 0.80;
      }
    }
    return penalty;
  }

  double _rangeMean({
    required Float64List prefix,
    required int start,
    required int end,
    required int length,
  }) {
    if (length <= 0) {
      return 0;
    }
    final s = start.clamp(0, length - 1);
    final e = end.clamp(0, length - 1);
    if (e < s) {
      return 0;
    }
    final sum = prefix[e + 1] - prefix[s];
    return sum / (e - s + 1);
  }

  double _bandPrior({
    required int left,
    required int top,
    required int side,
    required _AxisBand rowBand,
    required _AxisBand colBand,
  }) {
    final right = left + side - 1;
    final bottom = top + side - 1;
    final overlapX = _overlapLength(left, right, colBand.start, colBand.end);
    final overlapY = _overlapLength(top, bottom, rowBand.start, rowBand.end);
    final iouX = overlapX <= 0
        ? 0.0
        : overlapX /
              ((side + colBand.length - overlapX).clamp(1, 1 << 30).toDouble());
    final iouY = overlapY <= 0
        ? 0.0
        : overlapY /
              ((side + rowBand.length - overlapY).clamp(1, 1 << 30).toDouble());
    final aligned = math.sqrt(math.max(0.0, iouX * iouY));
    return 0.55 + (0.45 * aligned);
  }

  double _overlapLength(int aStart, int aEnd, int bStart, int bEnd) {
    final start = math.max(aStart, bStart);
    final end = math.min(aEnd, bEnd);
    if (end < start) {
      return 0;
    }
    return (end - start + 1).toDouble();
  }

  Float32List _axisEnergyRows({
    required _GradientMaps gradients,
    required int width,
    required int height,
  }) {
    final output = Float32List(height);
    for (int y = 0; y < height; y++) {
      final rowOffset = y * width;
      double sum = 0;
      for (int x = 0; x < width; x++) {
        final idx = rowOffset + x;
        sum += gradients.dx[idx] + gradients.dy[idx];
      }
      output[y] = (sum / width).toDouble();
    }
    return output;
  }

  Float32List _axisEnergyCols({
    required _GradientMaps gradients,
    required int width,
    required int height,
  }) {
    final output = Float32List(width);
    for (int x = 0; x < width; x++) {
      double sum = 0;
      for (int y = 0; y < height; y++) {
        final idx = (y * width) + x;
        sum += gradients.dx[idx] + gradients.dy[idx];
      }
      output[x] = (sum / height).toDouble();
    }
    return output;
  }

  _AxisBand _estimateAxisBand({
    required Float32List values,
    required int minSpan,
  }) {
    if (values.isEmpty) {
      return const _AxisBand(start: 0, end: 0);
    }

    double maxV = 0;
    double sum = 0;
    for (final v in values) {
      final d = v.toDouble();
      if (d > maxV) {
        maxV = d;
      }
      sum += d;
    }
    final mean = sum / values.length;
    double variance = 0;
    for (final v in values) {
      final d = v - mean;
      variance += d * d;
    }
    variance /= values.length;
    final std = math.sqrt(variance);

    _AxisBand? best;
    double bestScore = double.negativeInfinity;
    final thresholds = <double>[
      maxV * 0.82,
      maxV * 0.74,
      maxV * 0.66,
      mean + std * 0.55,
      mean + std * 0.35,
    ];

    for (final threshold in thresholds) {
      final runs = _runsAbove(values: values, threshold: threshold);
      for (final run in runs) {
        final expanded = _expandRun(
          run: run,
          maxLength: values.length,
          minSpan: minSpan,
        );
        final runMean = _runMean(
          values: values,
          start: expanded.start,
          end: expanded.end,
        );
        final score = runMean * expanded.length;
        if (score > bestScore) {
          bestScore = score;
          best = expanded;
        }
      }
    }

    if (best != null) {
      return best;
    }

    int peak = 0;
    double peakV = values.first.toDouble();
    for (int i = 1; i < values.length; i++) {
      if (values[i] > peakV) {
        peakV = values[i].toDouble();
        peak = i;
      }
    }
    final half = math.max(1, minSpan ~/ 2);
    final start = math.max(0, peak - half);
    final end = math.min(values.length - 1, peak + half);
    return _AxisBand(start: start, end: end);
  }

  List<_AxisBand> _runsAbove({
    required Float32List values,
    required double threshold,
  }) {
    final runs = <_AxisBand>[];
    int? start;
    for (int i = 0; i < values.length; i++) {
      final active = values[i] >= threshold;
      if (active) {
        start ??= i;
      } else if (start != null) {
        runs.add(_AxisBand(start: start, end: i - 1));
        start = null;
      }
    }
    if (start != null) {
      runs.add(_AxisBand(start: start, end: values.length - 1));
    }
    return runs;
  }

  _AxisBand _expandRun({
    required _AxisBand run,
    required int maxLength,
    required int minSpan,
  }) {
    if (run.length >= minSpan) {
      return run;
    }
    int start = run.start;
    int end = run.end;
    while ((end - start + 1) < minSpan) {
      if (start > 0) {
        start -= 1;
      }
      if ((end - start + 1) >= minSpan) {
        break;
      }
      if (end < maxLength - 1) {
        end += 1;
      } else if (start == 0) {
        break;
      }
    }
    return _AxisBand(start: start, end: end);
  }

  double _runMean({
    required Float32List values,
    required int start,
    required int end,
  }) {
    double sum = 0;
    for (int i = start; i <= end; i++) {
      sum += values[i];
    }
    return sum / (end - start + 1);
  }

  Float64List _buildPrefix1D(Float32List values) {
    final prefix = Float64List(values.length + 1);
    for (int i = 0; i < values.length; i++) {
      prefix[i + 1] = prefix[i] + values[i];
    }
    return prefix;
  }

  int _bestAxisStart({
    required Float64List prefix,
    required int windowSize,
    required int limit,
  }) {
    if (limit <= 0 || windowSize <= 0 || windowSize >= limit) {
      return 0;
    }
    int bestStart = 0;
    double bestSum = double.negativeInfinity;
    for (int start = 0; start <= limit - windowSize; start++) {
      final sum = prefix[start + windowSize] - prefix[start];
      if (sum > bestSum) {
        bestSum = sum;
        bestStart = start;
      }
    }
    return bestStart;
  }

  Float64List _buildIntegral({
    required Float32List values,
    required int width,
    required int height,
  }) {
    final stride = width + 1;
    final integral = Float64List((width + 1) * (height + 1));
    for (int y = 1; y <= height; y++) {
      double rowSum = 0;
      final srcRow = (y - 1) * width;
      final rowIndex = y * stride;
      final prevRowIndex = (y - 1) * stride;
      for (int x = 1; x <= width; x++) {
        rowSum += values[srcRow + x - 1];
        integral[rowIndex + x] = integral[prevRowIndex + x] + rowSum;
      }
    }
    return integral;
  }

  double _rectSum({
    required Float64List integral,
    required int width,
    required int left,
    required int top,
    required int right,
    required int bottom,
  }) {
    if (right < left || bottom < top) {
      return 0;
    }
    final stride = width + 1;
    final a = integral[top * stride + left];
    final b = integral[top * stride + (right + 1)];
    final c = integral[(bottom + 1) * stride + left];
    final d = integral[(bottom + 1) * stride + (right + 1)];
    return d - b - c + a;
  }

  double _rectMean({
    required Float64List integral,
    required int width,
    required int left,
    required int top,
    required int right,
    required int bottom,
  }) {
    final area = ((right - left + 1) * (bottom - top + 1)).toDouble();
    if (area <= 0) {
      return 0;
    }
    return _rectSum(
          integral: integral,
          width: width,
          left: left,
          top: top,
          right: right,
          bottom: bottom,
        ) /
        area;
  }

  _GradientMaps _buildGradients({
    required Uint8List luminance,
    required int width,
    required int height,
  }) {
    final dx = Float32List(width * height);
    final dy = Float32List(width * height);
    for (int y = 0; y < height; y++) {
      final rowOffset = y * width;
      for (int x = 0; x < width; x++) {
        final idx = rowOffset + x;
        final pixel = luminance[idx];
        if (x > 0) {
          dx[idx] = (pixel - luminance[idx - 1]).abs().toDouble();
        }
        if (y > 0) {
          dy[idx] = (pixel - luminance[idx - width]).abs().toDouble();
        }
      }
    }
    return _GradientMaps(dx: dx, dy: dy);
  }

  _AnalyzedImage _downsampleLuminance({
    required Uint8List sourceLuminance,
    required int sourceWidth,
    required int sourceHeight,
  }) {
    final sourceMax = math.max(sourceWidth, sourceHeight);
    if (sourceMax <= maxAnalysisDimension) {
      return _AnalyzedImage(
        width: sourceWidth,
        height: sourceHeight,
        luminance: sourceLuminance,
      );
    }

    final ratio = maxAnalysisDimension / sourceMax;
    final targetWidth = math.max(16, (sourceWidth * ratio).round());
    final targetHeight = math.max(16, (sourceHeight * ratio).round());
    final target = Uint8List(targetWidth * targetHeight);

    for (int y = 0; y < targetHeight; y++) {
      final srcY = ((y + 0.5) * sourceHeight / targetHeight).floor().clamp(
        0,
        sourceHeight - 1,
      );
      final targetRow = y * targetWidth;
      final sourceRow = srcY * sourceWidth;
      for (int x = 0; x < targetWidth; x++) {
        final srcX = ((x + 0.5) * sourceWidth / targetWidth).floor().clamp(
          0,
          sourceWidth - 1,
        );
        target[targetRow + x] = sourceLuminance[sourceRow + srcX];
      }
    }

    return _AnalyzedImage(
      width: targetWidth,
      height: targetHeight,
      luminance: target,
    );
  }

  Future<_DecodedRgbaImage?> _decodeRgba(Uint8List bytes) async {
    final codec = await ui.instantiateImageCodec(bytes);
    final frame = await codec.getNextFrame();
    final image = frame.image;
    final byteData = await image.toByteData(format: ui.ImageByteFormat.rawRgba);
    if (byteData == null) {
      return null;
    }
    return _DecodedRgbaImage(
      width: image.width,
      height: image.height,
      rgba: byteData.buffer.asUint8List(),
    );
  }

  Uint8List _toLuminance(Uint8List rgba) {
    final pixelCount = rgba.length ~/ 4;
    final output = Uint8List(pixelCount);
    int src = 0;
    for (int i = 0; i < pixelCount; i++) {
      final r = rgba[src];
      final g = rgba[src + 1];
      final b = rgba[src + 2];
      output[i] = ((r * 77) + (g * 150) + (b * 29)) >> 8;
      src += 4;
    }
    return output;
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

class _AnalyzedImage {
  const _AnalyzedImage({
    required this.width,
    required this.height,
    required this.luminance,
  });

  final int width;
  final int height;
  final Uint8List luminance;
}

class _GradientMaps {
  const _GradientMaps({required this.dx, required this.dy});

  final Float32List dx;
  final Float32List dy;
}

class _WindowCandidate {
  const _WindowCandidate({
    required this.left,
    required this.top,
    required this.side,
    required this.score,
  });

  final int left;
  final int top;
  final int side;
  final double score;
}

class _SearchBounds {
  const _SearchBounds({
    required this.left,
    required this.top,
    required this.right,
    required this.bottom,
  });

  final int left;
  final int top;
  final int right;
  final int bottom;

  int get width => right - left + 1;
  int get height => bottom - top + 1;
}

class _CandidateEvaluation {
  const _CandidateEvaluation({
    required this.seed,
    required this.rect,
    required this.quad,
    required this.score,
  });

  final _WindowCandidate seed;
  final _RectCandidate rect;
  final _QuadCandidate quad;
  final double score;
}

class _PointD {
  const _PointD(this.x, this.y);

  final double x;
  final double y;
}

class _EdgePoint extends _PointD {
  const _EdgePoint({required double x, required double y, required this.w})
    : super(x, y);

  final double w;
}

class _QuadCandidate {
  const _QuadCandidate({
    required this.tl,
    required this.tr,
    required this.br,
    required this.bl,
  });

  final _PointD tl;
  final _PointD tr;
  final _PointD br;
  final _PointD bl;
}

class _Bounds {
  const _Bounds({
    required this.left,
    required this.top,
    required this.right,
    required this.bottom,
  });

  final double left;
  final double top;
  final double right;
  final double bottom;

  int get width => (right - left + 1).round();
  int get height => (bottom - top + 1).round();
}

class _RectCandidate {
  const _RectCandidate({
    required this.left,
    required this.top,
    required this.right,
    required this.bottom,
  });

  final int left;
  final int top;
  final int right;
  final int bottom;
}

class _Interval {
  const _Interval(this.start, this.end);

  final int start;
  final int end;
}

class _AxisBand {
  const _AxisBand({required this.start, required this.end});

  final int start;
  final int end;

  int get length => end - start + 1;
}
