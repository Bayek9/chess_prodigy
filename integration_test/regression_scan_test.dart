import 'dart:convert';
import 'dart:io';

import 'package:chess_prodigy/scan/data/services/basic_fen_builder.dart';
import 'package:chess_prodigy/scan/data/services/basic_position_validator.dart';
import 'package:chess_prodigy/scan/data/services/default_scan_pipeline.dart';
import 'package:chess_prodigy/scan/domain/entities/scan_image.dart';
import 'package:chess_prodigy/scan/domain/services/board_presence_classifier.dart';
import 'package:chess_prodigy/scan/domain/usecases/scan_position_use_case.dart';
import 'package:flutter/services.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';

const _photoModelAsset = 'assets/scan_models/board_binary_photo.tflite';
const _screenModelAsset = 'assets/scan_models/board_binary_screen.tflite';

const _photoAcceptThreshold = 0.81;
const _photoRejectThreshold = 0.50;
const _screenAcceptThreshold = 0.89;
const _screenRejectThreshold = 0.57;

const _screenOpenCvMinBoardConfidence = 0.30;
const _screenOpenCvMinBoardConfidenceLineFallback = 0.34;
const _screenStrongAcceptRescueOpenCvMinBoardConfidence = 0.16;
const _screenStrongAcceptRescueOpenCvMinBoardConfidenceLineFallback = 0.26;
const _screenMinPostWarpGridness = 0.11;
const _screenGridnessRescueMinPostWarpGridness = 0.08;

const _autoRoutingAmbiguousScoreDelta = 0.05;
const _autoRoutingAlternateRetryMinScore = 0.35;
const _alternateBypassMinBoardQuality = 0.40;
const _alternateBypassMinBoardConfidence = 0.35;
const _alternateBypassMinBoardAreaRatio = 0.16;
const _gridnessRescueMinBoardQuality = 0.30;
const _gridnessRescueMinBoardConfidence = 0.35;
const _gridnessRescueMinBoardAreaRatio = 0.12;
const _strongAcceptNoBoardRescueMinAreaRatio = 0.10;
const _strongAcceptNoBoardRescueMinEdgeFrame = 0.50;
const _defaultRegressionSuite = String.fromEnvironment(
  'REGRESSION_SUITE',
  defaultValue: 'core',
);
const _regressionOnly = String.fromEnvironment(
  'REGRESSION_ONLY',
  defaultValue: '',
);

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  testWidgets('regression scan runner', (tester) async {
    final suite = _normalizedSuiteName(_defaultRegressionSuite);
    final casesAssetPath = suite == 'holdout'
        ? 'assets/regression_holdout/cases.json'
        : 'assets/regression/cases.json';
    final cases = await _loadCases(casesAssetPath, suite: suite);
    final onlyRaw = _regressionOnly.trim();
    final onlySet = onlyRaw.isEmpty
        ? null
        : onlyRaw
              .split(',')
              .map((entry) => entry.trim())
              .where((entry) => entry.isNotEmpty)
              .toSet();
    debugPrint(
      '[regression][config] suite=$suite '
      'REGRESSION_ONLY=${onlyRaw.isEmpty ? "<all>" : onlyRaw}',
    );
    if (cases.isEmpty) {
      debugPrint('[regression] no cases in $casesAssetPath (suite=$suite)');
      expect(true, isTrue);
      return;
    }

    final validator = const BasicPositionValidator();
    final fenBuilder = const BasicFenBuilder();

    final photoGate =
        DefaultScanPipelineFactory.boardPresenceClassifierForAsset(
          modelAssetPath: _photoModelAsset,
        );
    final screenGate =
        DefaultScanPipelineFactory.boardPresenceClassifierForAsset(
          modelAssetPath: _screenModelAsset,
        );

    final photoUseCase = DefaultScanPipelineFactory.create(
      validator: validator,
      fenBuilder: fenBuilder,
      useBoardPresenceGate: true,
      boardPresenceThreshold: _photoAcceptThreshold,
      boardPresenceRejectThreshold: _photoRejectThreshold,
      boardPresenceModelAssetPath: _photoModelAsset,
      useFallbackForReject: true,
    );

    final screenUseCase = DefaultScanPipelineFactory.create(
      validator: validator,
      fenBuilder: fenBuilder,
      useBoardPresenceGate: true,
      boardPresenceThreshold: _screenAcceptThreshold,
      boardPresenceRejectThreshold: _screenRejectThreshold,
      boardPresenceModelAssetPath: _screenModelAsset,
      useFallbackForReject: false,
      openCvMinBoardConfidence: _screenOpenCvMinBoardConfidence,
      openCvMinBoardConfidenceLineFallback:
          _screenOpenCvMinBoardConfidenceLineFallback,
      minPostWarpGridness: _screenMinPostWarpGridness,
    );

    final screenUseCaseStrongAcceptRescue = DefaultScanPipelineFactory.create(
      validator: validator,
      fenBuilder: fenBuilder,
      useBoardPresenceGate: true,
      boardPresenceThreshold: _screenAcceptThreshold,
      boardPresenceRejectThreshold: _screenRejectThreshold,
      boardPresenceModelAssetPath: _screenModelAsset,
      useFallbackForReject: false,
      openCvMinBoardConfidence:
          _screenStrongAcceptRescueOpenCvMinBoardConfidence,
      openCvMinBoardConfidenceLineFallback:
          _screenStrongAcceptRescueOpenCvMinBoardConfidenceLineFallback,
      minPostWarpGridness: _screenMinPostWarpGridness,
      openCvRescueMode: true,
    );

    final screenUseCaseGridnessRescue = DefaultScanPipelineFactory.create(
      validator: validator,
      fenBuilder: fenBuilder,
      useBoardPresenceGate: true,
      boardPresenceThreshold: _screenAcceptThreshold,
      boardPresenceRejectThreshold: _screenRejectThreshold,
      boardPresenceModelAssetPath: _screenModelAsset,
      useFallbackForReject: false,
      openCvMinBoardConfidence: _screenOpenCvMinBoardConfidence,
      openCvMinBoardConfidenceLineFallback:
          _screenOpenCvMinBoardConfidenceLineFallback,
      minPostWarpGridness: _screenGridnessRescueMinPostWarpGridness,
    );

    final photoUseCaseNoGate = DefaultScanPipelineFactory.create(
      validator: validator,
      fenBuilder: fenBuilder,
      useBoardPresenceGate: false,
    );

    final screenUseCaseNoGate = DefaultScanPipelineFactory.create(
      validator: validator,
      fenBuilder: fenBuilder,
      useBoardPresenceGate: false,
      openCvMinBoardConfidence: _screenOpenCvMinBoardConfidence,
      openCvMinBoardConfidenceLineFallback:
          _screenOpenCvMinBoardConfidenceLineFallback,
      minPostWarpGridness: _screenMinPostWarpGridness,
    );

    final entries = <Map<String, Object?>>[];
    var tp = 0;
    var tn = 0;
    var fp = 0;
    var fn = 0;
    var retryCount = 0;

    for (final c in cases) {
      if (onlySet != null && !onlySet.contains(c.id)) {
        continue;
      }
      final imageBytes = await _loadAssetBytes(c.assetPath);
      if (imageBytes == null) {
        final missing = <String, Object?>{
          'id': c.id,
          'asset_path': c.assetPath,
          'error': 'asset_not_found',
        };
        entries.add(missing);
        debugPrint('[regression][missing] ${jsonEncode(missing)}');
        continue;
      }

      final tempPath = await _materializeTempImage(
        id: c.id,
        originalAssetPath: c.assetPath,
        bytes: imageBytes,
      );
      final image = ScanInputImage(path: tempPath, bytes: imageBytes);

      final routing = await _resolveRouting(
        image: image,
        screenGate: screenGate,
        photoGate: photoGate,
      );

      final primaryUseCase = routing.domain == 'screen'
          ? screenUseCase
          : photoUseCase;
      final alternateUseCase = routing.alternateDomain == null
          ? null
          : (routing.alternateDomain == 'screen'
                ? screenUseCase
                : photoUseCase);
      final alternateUseCaseNoGate = routing.alternateDomain == null
          ? null
          : (routing.alternateDomain == 'screen'
                ? screenUseCaseNoGate
                : photoUseCaseNoGate);

      final primaryWatch = Stopwatch()..start();
      var primaryResult = await primaryUseCase.execute(image);
      primaryWatch.stop();
      final tPrimaryMs = primaryWatch.elapsedMilliseconds;

      var finalResult = primaryResult;
      var finalDomain = routing.domain;
      var tAltMs = 0;

      final primaryGateRaw = _extractGateDecisionRaw(
        primaryResult.detectorDebug,
      );
      var retries = 0;
      const maxRetries = 2;
      final shouldRetryAlternate =
          !primaryResult.boardDetected &&
          alternateUseCase != null &&
          (routing.ambiguous ||
              ((routing.alternateScore ?? double.negativeInfinity) >=
                  _autoRoutingAlternateRetryMinScore) ||
              primaryGateRaw == 'allow_strong_accept');

      var alternateRanWithoutGate = false;
      var switchedToAlternate = false;
      var switchedToGridnessRescue = false;
      var switchedToStrongAcceptNoBoardRescue = false;
      bool? alternateBypassQualityPass;
      if (shouldRetryAlternate && retries < maxRetries) {
        retryCount += 1;
        retries += 1;
        final retryAlternateWithoutGate =
            primaryGateRaw == 'allow_strong_accept';
        final selectedAlternateUseCase = retryAlternateWithoutGate
            ? alternateUseCaseNoGate!
            : alternateUseCase;
        alternateRanWithoutGate = retryAlternateWithoutGate;
        final altWatch = Stopwatch()..start();
        final alternateResult = await selectedAlternateUseCase.execute(image);
        altWatch.stop();
        tAltMs = altWatch.elapsedMilliseconds;

        alternateBypassQualityPass =
            !retryAlternateWithoutGate ||
            _passesAlternateBypassQualityGate(alternateResult);
        final switched = _shouldSwitchToAlternateResult(
          primary: primaryResult,
          alternate: alternateResult,
          requireBypassQualityGate: retryAlternateWithoutGate,
        );
        if (switched) {
          finalResult = alternateResult;
          finalDomain = routing.alternateDomain!;
          switchedToAlternate = true;
        }
      }

      final currentGateDecisionRaw =
          alternateRanWithoutGate && switchedToAlternate
          ? 'bypass_no_gate'
          : _extractGateDecisionRaw(finalResult.detectorDebug);
      final currentFinalDecisionRaw = _extractFinalDecisionRaw(
        finalResult.detectorDebug,
      );
      final shouldRetryGridnessRescue =
          retries < maxRetries &&
          finalDomain == 'screen' &&
          currentFinalDecisionRaw == 'reject_post_warp_gridness' &&
          (currentGateDecisionRaw == 'allow_strong_accept' ||
              currentGateDecisionRaw == 'bypass_no_gate') &&
          _passesGridnessRescuePrecheck(finalResult);
      bool? gridnessRescueQualityPass;
      bool? strongAcceptNoBoardRescueQualityPass;
      if (shouldRetryGridnessRescue) {
        retryCount += 1;
        retries += 1;
        final rescueWatch = Stopwatch()..start();
        final rescueResult = await screenUseCaseGridnessRescue.execute(image);
        rescueWatch.stop();
        tAltMs += rescueWatch.elapsedMilliseconds;
        gridnessRescueQualityPass = _passesGridnessRescuePrecheck(rescueResult);
        if (rescueResult.boardDetected && gridnessRescueQualityPass) {
          finalResult = rescueResult;
          finalDomain = 'screen';
          switchedToGridnessRescue = true;
        }
      }

      final shouldRetryStrongAcceptNoBoardRescue =
          retries < maxRetries &&
          finalDomain == 'screen' &&
          currentGateDecisionRaw == 'allow_strong_accept' &&
          !finalResult.boardDetected &&
          _isRejectedNoBoard(finalResult.detectorDebug);
      if (shouldRetryStrongAcceptNoBoardRescue) {
        retryCount += 1;
        retries += 1;
        final rescueWatch = Stopwatch()..start();
        final rescueResult = await screenUseCaseStrongAcceptRescue.execute(
          image,
        );
        rescueWatch.stop();
        tAltMs += rescueWatch.elapsedMilliseconds;
        String fmt(double? value) =>
            value == null ? 'na' : value.toStringAsFixed(3);
        final rescueArea = _extractMetric(
          rescueResult.detectorDebug,
          'board_area_ratio',
        );
        final rescueEdge = _extractMetric(
          rescueResult.detectorDebug,
          'board_edge_frame',
        );
        final rescueConf = _extractMetric(
          rescueResult.detectorDebug,
          'board_confidence',
        );
        final rescueChecker = _extractMetric(
          rescueResult.detectorDebug,
          'board_checker',
        );
        final rescueCombined = _extractMetric(
          rescueResult.detectorDebug,
          'board_quality',
        );
        final rescueReject = _extractRejectReason(rescueResult.detectorDebug);
        debugPrint(
          '[regression][sa_rescue] id=${c.id} board=${rescueResult.boardDetected} '
          'area=${fmt(rescueArea)} edge=${fmt(rescueEdge)} '
          'conf=${fmt(rescueConf)} chk=${fmt(rescueChecker)} comb=${fmt(rescueCombined)} '
          'path=${_extractWhichPath(rescueResult.detectorDebug)} reject=$rescueReject',
        );
        strongAcceptNoBoardRescueQualityPass =
            _passesStrongAcceptNoBoardRescueGate(rescueResult);
        if (rescueResult.boardDetected &&
            strongAcceptNoBoardRescueQualityPass) {
          finalResult = rescueResult;
          finalDomain = 'screen';
          switchedToStrongAcceptNoBoardRescue = true;
        }
      }
      var gateRaw = _extractGateDecisionRaw(finalResult.detectorDebug);
      var finalDecisionRaw = _extractFinalDecisionRaw(
        finalResult.detectorDebug,
      );
      String? bypassReason;
      if (alternateRanWithoutGate &&
          switchedToAlternate &&
          !switchedToGridnessRescue &&
          !switchedToStrongAcceptNoBoardRescue) {
        gateRaw = 'bypass_no_gate';
        finalDecisionRaw = 'bypass_no_gate';
        bypassReason = 'primary_allow_strong_accept_detector_failed';
      }
      if (switchedToGridnessRescue) {
        bypassReason = 'gridness_rescue_post_warp_gridness';
      }
      if (switchedToStrongAcceptNoBoardRescue) {
        bypassReason = 'strong_accept_rejected_no_board_rescue';
      }
      final decision = _decisionBucket(finalResult.detectorDebug);
      final expectedBoard = c.expectedClass == 'board';
      final boardDetected = finalResult.boardDetected;

      final outcome = expectedBoard
          ? (boardDetected ? 'TP' : 'FN')
          : (boardDetected ? 'FP' : 'TN');

      switch (outcome) {
        case 'TP':
          tp += 1;
          break;
        case 'TN':
          tn += 1;
          break;
        case 'FP':
          fp += 1;
          break;
        case 'FN':
          fn += 1;
          break;
      }

      final entry = <String, Object?>{
        'id': c.id,
        'asset_path': c.assetPath,
        'temp_path': tempPath,
        'expected_domain': c.expectedDomain,
        'expected_class': c.expectedClass,
        'acquisition': c.acquisition,
        'chosen_domain': finalDomain,
        'route_match': _normalizedDomain(finalDomain) == c.expectedDomain,
        'screen_strong': routing.screenStrong,
        'photo_strong': routing.photoStrong,
        'screen_score': routing.screenScore,
        'photo_score': routing.photoScore,
        'delta': routing.delta,
        'ambiguous': routing.ambiguous,
        'gate_decision_raw': gateRaw,
        'final_decision_raw': finalDecisionRaw,
        'decision': decision,
        'board_detected': boardDetected,
        'outcome': outcome,
        't_primary_ms': tPrimaryMs,
        't_alt_ms': tAltMs,
        'alternate_ran_without_gate': alternateRanWithoutGate,
        'alternate_bypass_quality_pass': alternateBypassQualityPass,
        'gridness_rescue_attempted': shouldRetryGridnessRescue,
        'gridness_rescue_quality_pass': gridnessRescueQualityPass,
        'gridness_rescue_switched': switchedToGridnessRescue,
        'strong_accept_noboard_rescue_attempted':
            shouldRetryStrongAcceptNoBoardRescue,
        'strong_accept_noboard_rescue_quality_pass':
            strongAcceptNoBoardRescueQualityPass,
        'strong_accept_noboard_rescue_switched':
            switchedToStrongAcceptNoBoardRescue,
        'bypass_reason': bypassReason,
        'post_warp_gridness': _extractMetric(
          finalResult.detectorDebug,
          'post_warp_gridness',
        ),
        'min_post_warp_gridness': _extractMetric(
          finalResult.detectorDebug,
          'min_post_warp_gridness',
        ),
        'rejected_post_warp_gridness': finalResult.detectorDebug.contains(
          'decision=reject_post_warp_gridness',
        ),
        'board_quality': _extractMetric(
          finalResult.detectorDebug,
          'board_quality',
        ),
        'board_confidence': _extractMetric(
          finalResult.detectorDebug,
          'board_confidence',
        ),
        'board_checker': _extractMetric(
          finalResult.detectorDebug,
          'board_checker',
        ),
        'board_regularity': _extractMetric(
          finalResult.detectorDebug,
          'board_regularity',
        ),
        'board_edge_frame': _extractMetric(
          finalResult.detectorDebug,
          'board_edge_frame',
        ),
        'board_area_ratio': _extractMetric(
          finalResult.detectorDebug,
          'board_area_ratio',
        ),
        'which_path_won': _extractWhichPath(finalResult.detectorDebug),
        'reject_reason': _extractRejectReason(finalResult.detectorDebug),
      };

      entries.add(entry);
      debugPrint('[regression][entry] ${jsonEncode(entry)}');
    }

    final total = tp + tn + fp + fn;
    final retryRate = total == 0 ? 0.0 : (retryCount * 100.0 / total);

    final report = <String, Object?>{
      'generated_at': DateTime.now().toIso8601String(),
      'suite': suite,
      'total': total,
      'tp': tp,
      'tn': tn,
      'fp': fp,
      'fn': fn,
      'retry_count': retryCount,
      'retry_rate': double.parse(retryRate.toStringAsFixed(1)),
      'entries': entries,
    };

    debugPrint(
      '[regression][summary] total=$total tp=$tp tn=$tn fp=$fp fn=$fn '
      'retry_count=$retryCount retry_rate=${retryRate.toStringAsFixed(1)}',
    );
    debugPrint('[regression][json] ${jsonEncode(report)}');

    expect(entries.isNotEmpty, isTrue);
  });
}

class _RegressionCase {
  const _RegressionCase({
    required this.id,
    required this.assetPath,
    required this.expectedDomain,
    required this.expectedClass,
    required this.acquisition,
  });

  final String id;
  final String assetPath;
  final String expectedDomain;
  final String expectedClass;
  final String acquisition;
}

class _RoutingResult {
  const _RoutingResult({
    required this.domain,
    required this.alternateDomain,
    required this.screenStrong,
    required this.photoStrong,
    required this.screenScore,
    required this.photoScore,
    required this.delta,
    required this.ambiguous,
    required this.alternateScore,
  });

  final String domain;
  final String? alternateDomain;
  final double? screenStrong;
  final double? photoStrong;
  final double? screenScore;
  final double? photoScore;
  final double? delta;
  final bool ambiguous;
  final double? alternateScore;
}

Future<List<_RegressionCase>> _loadCases(
  String assetPath, {
  required String suite,
}) async {
  final raw = await rootBundle.loadString(assetPath);
  final decoded = jsonDecode(raw);

  final items = switch (decoded) {
    List<dynamic> list => list,
    Map<String, dynamic> map when map['cases'] is List<dynamic> =>
      map['cases'] as List<dynamic>,
    _ => <dynamic>[],
  };

  final out = <_RegressionCase>[];
  for (final item in items) {
    if (item is! Map) {
      continue;
    }
    final map = Map<String, dynamic>.from(item);
    final id = map['id']?.toString() ?? '';
    final path = map['path']?.toString() ?? '';
    if (id.isEmpty || path.isEmpty) {
      continue;
    }
    out.add(
      _RegressionCase(
        id: id,
        assetPath: _resolveAssetPath(path, suite: suite),
        expectedDomain: _normalizedDomain(
          map['expected_domain']?.toString() ?? 'photo_real',
        ),
        expectedClass: map['expected_class']?.toString() == 'board'
            ? 'board'
            : 'no_board',
        acquisition: map['acquisition']?.toString() ?? 'unknown',
      ),
    );
  }
  return out;
}

String _resolveAssetPath(String rawPath, {required String suite}) {
  final normalized = rawPath.replaceAll('\\', '/');
  final assetRoot = suite == 'holdout'
      ? 'assets/regression_holdout/images'
      : 'assets/regression/images';
  final toolRoot = suite == 'holdout'
      ? 'tools/regression/holdout/images/'
      : 'tools/regression/images/';

  if (normalized.startsWith('assets/')) {
    return normalized;
  }
  if (normalized.startsWith(toolRoot)) {
    return normalized.replaceFirst(toolRoot, '$assetRoot/');
  }
  if (normalized.startsWith('images/')) {
    return '$assetRoot/${normalized.substring('images/'.length)}';
  }
  return '$assetRoot/$normalized';
}

Future<Uint8List?> _loadAssetBytes(String assetPath) async {
  try {
    final data = await rootBundle.load(assetPath);
    return data.buffer.asUint8List();
  } catch (_) {
    return null;
  }
}

Future<String> _materializeTempImage({
  required String id,
  required String originalAssetPath,
  required Uint8List bytes,
}) async {
  final lower = originalAssetPath.toLowerCase();
  final extension = lower.endsWith('.png')
      ? '.png'
      : (lower.endsWith('.jpg') || lower.endsWith('.jpeg'))
      ? '.jpg'
      : lower.endsWith('.webp')
      ? '.webp'
      : '.img';
  final file = File('${Directory.systemTemp.path}/reg_scan_$id$extension');
  await file.writeAsBytes(bytes, flush: true);
  return file.path;
}

Future<_RoutingResult> _resolveRouting({
  required ScanInputImage image,
  required BoardPresenceClassifier screenGate,
  required BoardPresenceClassifier photoGate,
}) async {
  final screenPrediction = await screenGate.predict(image);
  final photoPrediction = await photoGate.predict(image);

  final screenStrong = screenPrediction.isAvailable
      ? screenPrediction.probability.clamp(0.0, 1.0).toDouble()
      : null;
  final photoStrong = photoPrediction.isAvailable
      ? photoPrediction.probability.clamp(0.0, 1.0).toDouble()
      : null;

  final screenScore = _scoreForRouting(
    prediction: screenPrediction,
    rejectThreshold: _screenRejectThreshold,
  );
  final photoScore = _scoreForRouting(
    prediction: photoPrediction,
    rejectThreshold: _photoRejectThreshold,
  );

  if (screenScore == null && photoScore == null) {
    return const _RoutingResult(
      domain: 'photo_real',
      alternateDomain: 'screen',
      screenStrong: null,
      photoStrong: null,
      screenScore: null,
      photoScore: null,
      delta: null,
      ambiguous: false,
      alternateScore: null,
    );
  }

  final chooseScreen =
      photoScore == null || (screenScore != null && screenScore >= photoScore);
  final domain = chooseScreen ? 'screen' : 'photo_real';
  final alternateDomain = chooseScreen ? 'photo_real' : 'screen';
  final delta = (screenScore != null && photoScore != null)
      ? (screenScore - photoScore).abs()
      : null;
  final ambiguous = delta != null && delta < _autoRoutingAmbiguousScoreDelta;
  final alternateScore = chooseScreen ? photoScore : screenScore;

  return _RoutingResult(
    domain: domain,
    alternateDomain: alternateDomain,
    screenStrong: screenStrong,
    photoStrong: photoStrong,
    screenScore: screenScore,
    photoScore: photoScore,
    delta: delta,
    ambiguous: ambiguous,
    alternateScore: alternateScore,
  );
}

double? _scoreForRouting({
  required BoardPresencePrediction prediction,
  required double rejectThreshold,
}) {
  if (!prediction.isAvailable) {
    return null;
  }
  final routingProbability = prediction.fallbackOrProbability
      .clamp(0.0, 1.0)
      .toDouble();
  return routingProbability - rejectThreshold;
}

bool _shouldSwitchToAlternateResult({
  required ScanPipelineResult primary,
  required ScanPipelineResult alternate,
  bool requireBypassQualityGate = false,
}) {
  if (requireBypassQualityGate &&
      !_passesAlternateBypassQualityGate(alternate)) {
    return false;
  }
  if (alternate.boardDetected && !primary.boardDetected) {
    return true;
  }
  if (!alternate.boardDetected || !primary.boardDetected) {
    return false;
  }
  final primaryQuality = _extractMetric(primary.detectorDebug, 'board_quality');
  final alternateQuality = _extractMetric(
    alternate.detectorDebug,
    'board_quality',
  );
  if (primaryQuality != null && alternateQuality != null) {
    if (alternateQuality > primaryQuality) {
      return true;
    }
    if (alternateQuality < primaryQuality) {
      return false;
    }
  }
  final primaryConfidence = _extractMetric(
    primary.detectorDebug,
    'board_confidence',
  );
  final alternateConfidence = _extractMetric(
    alternate.detectorDebug,
    'board_confidence',
  );
  if (primaryConfidence != null && alternateConfidence != null) {
    return alternateConfidence > primaryConfidence;
  }
  return false;
}

bool _passesAlternateBypassQualityGate(ScanPipelineResult result) {
  if (!result.boardDetected) {
    return false;
  }
  final quality = _extractMetric(result.detectorDebug, 'board_quality');
  final confidence = _extractMetric(result.detectorDebug, 'board_confidence');
  final areaRatio = _extractMetric(result.detectorDebug, 'board_area_ratio');
  if (quality == null || confidence == null || areaRatio == null) {
    return false;
  }
  return quality >= _alternateBypassMinBoardQuality &&
      confidence >= _alternateBypassMinBoardConfidence &&
      areaRatio >= _alternateBypassMinBoardAreaRatio;
}

bool _passesGridnessRescuePrecheck(ScanPipelineResult result) {
  final quality = _extractMetric(result.detectorDebug, 'board_quality');
  final confidence = _extractMetric(result.detectorDebug, 'board_confidence');
  final areaRatio = _extractMetric(result.detectorDebug, 'board_area_ratio');
  if (quality == null || confidence == null || areaRatio == null) {
    return false;
  }
  return quality >= _gridnessRescueMinBoardQuality &&
      confidence >= _gridnessRescueMinBoardConfidence &&
      areaRatio >= _gridnessRescueMinBoardAreaRatio;
}

bool _passesStrongAcceptNoBoardRescueGate(ScanPipelineResult result) {
  if (!result.boardDetected) {
    return false;
  }
  final areaRatio = _extractMetric(result.detectorDebug, 'board_area_ratio');
  final edgeFrame = _extractMetric(result.detectorDebug, 'board_edge_frame');
  if (areaRatio == null || edgeFrame == null) {
    return false;
  }
  return areaRatio >= _strongAcceptNoBoardRescueMinAreaRatio &&
      edgeFrame >= _strongAcceptNoBoardRescueMinEdgeFrame;
}

bool _isRejectedNoBoard(String detectorDebug) {
  return detectorDebug.contains('which_path_won=rejected_no_board') ||
      (detectorDebug.contains('board_rejected=true') &&
          detectorDebug.contains('rejected_no_board'));
}

String _extractGateDecisionRaw(String detectorDebug) {
  final match = RegExp(r'decision=([a-z_]+)').firstMatch(detectorDebug);
  if (match == null || match.groupCount < 1) {
    return 'unknown';
  }
  return match.group(1)!;
}

String _extractFinalDecisionRaw(String detectorDebug) {
  final allMatches = RegExp(
    r'decision=([a-z_]+)',
  ).allMatches(detectorDebug).toList(growable: false);
  if (allMatches.isEmpty || allMatches.last.groupCount < 1) {
    return 'unknown';
  }
  return allMatches.last.group(1)!;
}

String _decisionBucket(String detectorDebug) {
  if (detectorDebug.contains('board_rejected=true') ||
      detectorDebug.contains('which_path_won=rejected_')) {
    return 'reject';
  }
  if (detectorDebug.contains('decision=reject_strong_no_board')) {
    return 'reject';
  }
  if (detectorDebug.contains('decision=allow_strong_accept')) {
    return 'accept';
  }
  return 'gray';
}

double? _extractMetric(String detectorDebug, String key) {
  final match = RegExp(
    '$key=([-+]?\\d+(?:\\.\\d+)?)',
  ).firstMatch(detectorDebug);
  if (match == null || match.groupCount < 1) {
    return null;
  }
  return double.tryParse(match.group(1)!);
}

String _extractWhichPath(String detectorDebug) {
  final match = RegExp(r'which_path_won=([^\s]+)').firstMatch(detectorDebug);
  if (match == null || match.groupCount < 1) {
    return 'unknown';
  }
  return match.group(1)!;
}

String _extractRejectReason(String detectorDebug) {
  final match = RegExp(r'reject=([^\s]+)').firstMatch(detectorDebug);
  if (match == null || match.groupCount < 1) {
    return 'unknown';
  }
  return match.group(1)!;
}

String _normalizedSuiteName(String raw) {
  switch (raw) {
    case 'holdout':
      return 'holdout';
    case 'core':
    default:
      return 'core';
  }
}

String _normalizedDomain(String raw) {
  switch (raw) {
    case 'screen':
      return 'screen';
    case 'photo_print':
    case 'photo_real':
    default:
      return 'photo_real';
  }
}
