import 'dart:math' as math;

import '../entities/board_geometry.dart';
import '../entities/scan_image.dart';
import '../entities/scan_validation_dataset.dart';
import '../services/scan_validation_dataset_loader.dart';
import 'scan_position_use_case.dart';

class ScanFieldComparison {
  const ScanFieldComparison({
    required this.field,
    required this.expected,
    required this.detected,
    required this.matched,
  });

  final String field;
  final Object? expected;
  final Object? detected;
  final bool matched;
}

class ScanCaseEvaluation {
  const ScanCaseEvaluation({
    required this.testCase,
    required this.comparisons,
    this.sourceImage,
    this.result,
    this.cornerErrorMetrics,
    this.error,
  });

  final ScanValidationCase testCase;
  final ScanInputImage? sourceImage;
  final ScanPipelineResult? result;
  final List<ScanFieldComparison> comparisons;
  final CornerErrorMetrics? cornerErrorMetrics;
  final String? error;

  static const double _qualityMeanThreshold = 8.0;
  static const double _qualityMaxThreshold = 15.0;
  static const double _excellentMeanThreshold = 4.0;
  static const double _excellentMaxThreshold = 8.0;

  bool get functionalPassed {
    if (error != null) {
      return false;
    }
    for (final c in comparisons) {
      if (c.field == 'corners') {
        continue;
      }
      if (!c.matched) {
        return false;
      }
    }
    return true;
  }

  bool get qualityPassed {
    return _passesQualityGate(
      meanThreshold: _qualityMeanThreshold,
      maxThreshold: _qualityMaxThreshold,
    );
  }

  bool get excellentPassed {
    return _passesQualityGate(
      meanThreshold: _excellentMeanThreshold,
      maxThreshold: _excellentMaxThreshold,
    );
  }

  bool get passed => functionalPassed;

  String get statusLabel {
    if (excellentPassed) {
      return 'PASS_EXCELLENT';
    }
    if (qualityPassed) {
      return 'PASS_QUALITY';
    }
    if (functionalPassed) {
      return 'PASS_FUNCTIONAL';
    }
    return 'FAIL';
  }

  bool _passesQualityGate({
    required double meanThreshold,
    required double maxThreshold,
  }) {
    if (!functionalPassed) {
      return false;
    }
    final mean = cornerErrorMetrics?.meanPercent;
    final max = cornerErrorMetrics?.maxPercent;
    if (mean != null && mean >= meanThreshold) {
      return false;
    }
    if (max != null && max >= maxThreshold) {
      return false;
    }

    final warpDetected = _detectedBool('warp_ok');
    if (warpDetected != null && warpDetected == false) {
      return false;
    }
    final orientationDetected = _detectedBool('orientation_ok');
    if (orientationDetected != null && orientationDetected == false) {
      return false;
    }
    return true;
  }

  bool? _detectedBool(String field) {
    for (final c in comparisons) {
      if (c.field != field) {
        continue;
      }
      final value = c.detected;
      if (value is bool) {
        return value;
      }
      return null;
    }
    return null;
  }
}

class CornerErrorMetrics {
  const CornerErrorMetrics({
    required this.meanPx,
    required this.maxPx,
    required this.meanPercent,
    required this.maxPercent,
  });

  final double meanPx;
  final double maxPx;
  final double meanPercent;
  final double maxPercent;
}

class ScanDatasetValidationReport {
  const ScanDatasetValidationReport({
    required this.dataset,
    required this.evaluations,
  });

  final ScanValidationDataset dataset;
  final List<ScanCaseEvaluation> evaluations;

  int get total => evaluations.length;
  int get passedCount => evaluations.where((e) => e.functionalPassed).length;
  int get excellentPassedCount =>
      evaluations.where((e) => e.excellentPassed).length;
  int get qualityPassedCount =>
      evaluations.where((e) => e.qualityPassed).length;
  int get functionalPassedCount =>
      evaluations.where((e) => e.functionalPassed).length;
  int get failedCount => evaluations.where((e) => !e.functionalPassed).length;
}

class RunScanDatasetValidationUseCase {
  RunScanDatasetValidationUseCase({
    required ScanPositionUseCase scanPipeline,
    required ScanValidationDatasetLoader datasetLoader,
    this.cornerTolerance = 20.0,
    this.cornerMeanPercentTolerance = 10.0,
    this.cornerMaxPercentTolerance = 16.0,
    this.warpMeanErrorPercentThreshold = 15.0,
    this.warpMaxErrorPercentThreshold = 25.0,
    this.orientationMeanErrorPercentThreshold = 16.0,
  }) : _scanPipeline = scanPipeline,
       _datasetLoader = datasetLoader;

  final ScanPositionUseCase _scanPipeline;
  final ScanValidationDatasetLoader _datasetLoader;
  final double cornerTolerance;
  final double cornerMeanPercentTolerance;
  final double cornerMaxPercentTolerance;
  final double warpMeanErrorPercentThreshold;
  final double warpMaxErrorPercentThreshold;
  final double orientationMeanErrorPercentThreshold;

  Future<ScanDatasetValidationReport> run({
    required String datasetAssetPath,
    bool includePayloads = false,
    void Function(int done, int total, String caseId)? onProgress,
  }) async {
    final dataset = await _datasetLoader.loadDataset(datasetAssetPath);
    final evaluations = <ScanCaseEvaluation>[];
    final total = dataset.cases.length;

    for (int i = 0; i < dataset.cases.length; i++) {
      final testCase = dataset.cases[i];
      final evaluation = await runSingleCase(
        testCase: testCase,
        includePayload: includePayloads,
      );
      evaluations.add(evaluation);
      onProgress?.call(i + 1, total, testCase.id);
      await Future<void>.delayed(const Duration(milliseconds: 1));
    }

    return ScanDatasetValidationReport(
      dataset: dataset,
      evaluations: evaluations,
    );
  }

  Future<ScanCaseEvaluation> runSingleCase({
    required ScanValidationCase testCase,
    bool includePayload = true,
  }) async {
    try {
      final image = await _datasetLoader.loadCaseImage(testCase);
      final result = await _scanPipeline.execute(image);
      final compareResult = _compare(testCase: testCase, result: result);
      return ScanCaseEvaluation(
        testCase: testCase,
        sourceImage: includePayload ? image : null,
        result: includePayload ? result : null,
        comparisons: compareResult.comparisons,
        cornerErrorMetrics: compareResult.cornerErrorMetrics,
      );
    } catch (e) {
      return ScanCaseEvaluation(
        testCase: testCase,
        comparisons: const <ScanFieldComparison>[],
        error: e.toString(),
      );
    }
  }

  _CompareResult _compare({
    required ScanValidationCase testCase,
    required ScanPipelineResult result,
  }) {
    final expected = testCase.expected;
    final fields = <ScanFieldComparison>[];
    CornerErrorMetrics? cornerErrorMetrics;

    if (expected.corners != null && !_usesNormalizedMockCorners(result)) {
      cornerErrorMetrics = _computeCornerErrorMetrics(
        expected: expected.corners!,
        actual: result.geometry.corners,
      );
    }

    final effectiveWarpOk = _effectiveWarpOk(
      result: result,
      cornerErrorMetrics: cornerErrorMetrics,
    );
    final effectiveOrientationOk = _effectiveOrientationOk(
      result: result,
      cornerErrorMetrics: cornerErrorMetrics,
    );

    if (expected.boardDetected != null) {
      fields.add(
        ScanFieldComparison(
          field: 'board_detected',
          expected: expected.boardDetected,
          detected: result.boardDetected,
          matched: result.boardDetected == expected.boardDetected,
        ),
      );
    }
    if (expected.warpOk != null) {
      fields.add(
        ScanFieldComparison(
          field: 'warp_ok',
          expected: expected.warpOk,
          detected: effectiveWarpOk,
          matched: effectiveWarpOk == expected.warpOk,
        ),
      );
    }
    if (expected.orientationOk != null) {
      fields.add(
        ScanFieldComparison(
          field: 'orientation_ok',
          expected: expected.orientationOk,
          detected: effectiveOrientationOk,
          matched: effectiveOrientationOk == expected.orientationOk,
        ),
      );
    }

    if (expected.corners != null && !_usesNormalizedMockCorners(result)) {
      final cornersMatched = _compareCorners(
        expected.corners!,
        result.geometry.corners,
        cornerErrorMetrics: cornerErrorMetrics,
      );
      fields.add(
        ScanFieldComparison(
          field: 'corners',
          expected: _cornersLabel(expected.corners!),
          detected: _cornersLabel(result.geometry.corners),
          matched: cornersMatched,
        ),
      );
    }

    if (expected.fen != null) {
      fields.add(
        ScanFieldComparison(
          field: 'fen',
          expected: expected.fen,
          detected: result.detectedFen,
          matched: result.detectedFen == expected.fen,
        ),
      );
    }

    return _CompareResult(
      comparisons: fields,
      cornerErrorMetrics: cornerErrorMetrics,
    );
  }

  bool _effectiveWarpOk({
    required ScanPipelineResult result,
    required CornerErrorMetrics? cornerErrorMetrics,
  }) {
    if (!result.warpOk) {
      return false;
    }
    if (cornerErrorMetrics == null) {
      return true;
    }
    return cornerErrorMetrics.meanPercent <= warpMeanErrorPercentThreshold &&
        cornerErrorMetrics.maxPercent <= warpMaxErrorPercentThreshold;
  }

  bool _effectiveOrientationOk({
    required ScanPipelineResult result,
    required CornerErrorMetrics? cornerErrorMetrics,
  }) {
    if (!result.orientationOk) {
      return false;
    }
    if (cornerErrorMetrics == null) {
      return true;
    }
    return cornerErrorMetrics.meanPercent <=
        orientationMeanErrorPercentThreshold;
  }

  CornerErrorMetrics? _computeCornerErrorMetrics({
    required List<BoardCorner> expected,
    required List<BoardCorner> actual,
  }) {
    if (expected.length != 4 || actual.length != 4) {
      return null;
    }

    double sumError = 0;
    double maxError = 0;
    for (int i = 0; i < 4; i++) {
      final dx = expected[i].x - actual[i].x;
      final dy = expected[i].y - actual[i].y;
      final dist = math.sqrt((dx * dx) + (dy * dy));
      sumError += dist;
      if (dist > maxError) {
        maxError = dist;
      }
    }
    final meanError = sumError / 4.0;

    final sideRef = _averageExpectedSide(expected);
    final safeSideRef = sideRef <= 1e-6 ? 1.0 : sideRef;

    return CornerErrorMetrics(
      meanPx: meanError,
      maxPx: maxError,
      meanPercent: (meanError / safeSideRef) * 100.0,
      maxPercent: (maxError / safeSideRef) * 100.0,
    );
  }

  double _averageExpectedSide(List<BoardCorner> corners) {
    if (corners.length != 4) {
      return 1.0;
    }
    final d01 = _distance(corners[0], corners[1]);
    final d12 = _distance(corners[1], corners[2]);
    final d23 = _distance(corners[2], corners[3]);
    final d30 = _distance(corners[3], corners[0]);
    return (d01 + d12 + d23 + d30) / 4.0;
  }

  double _distance(BoardCorner a, BoardCorner b) {
    final dx = a.x - b.x;
    final dy = a.y - b.y;
    return math.sqrt((dx * dx) + (dy * dy));
  }

  bool _usesNormalizedMockCorners(ScanPipelineResult result) {
    final corners = result.geometry.corners;
    if (corners.length != 4) {
      return false;
    }
    return corners.every((c) => c.x >= 0 && c.x <= 1 && c.y >= 0 && c.y <= 1);
  }

  bool _compareCorners(
    List<BoardCorner> expected,
    List<BoardCorner> actual, {
    CornerErrorMetrics? cornerErrorMetrics,
  }) {
    if (expected.length != actual.length) {
      return false;
    }

    if (cornerErrorMetrics != null) {
      if (cornerErrorMetrics.meanPercent <= cornerMeanPercentTolerance &&
          cornerErrorMetrics.maxPercent <= cornerMaxPercentTolerance) {
        return true;
      }
    }

    for (int i = 0; i < expected.length; i++) {
      final dx = (expected[i].x - actual[i].x).abs();
      final dy = (expected[i].y - actual[i].y).abs();
      if (dx > cornerTolerance || dy > cornerTolerance) {
        return false;
      }
    }
    return true;
  }

  String _cornersLabel(List<BoardCorner> corners) {
    return corners
        .map((p) => '(${p.x.toStringAsFixed(1)},${p.y.toStringAsFixed(1)})')
        .join(' ');
  }
}

class _CompareResult {
  const _CompareResult({
    required this.comparisons,
    required this.cornerErrorMetrics,
  });

  final List<ScanFieldComparison> comparisons;
  final CornerErrorMetrics? cornerErrorMetrics;
}
