import 'dart:math' as math;
import 'dart:ui' as ui;

import 'package:flutter/foundation.dart';
import 'package:opencv_4/factory/pathfrom.dart';
import 'package:opencv_4/opencv_4.dart';

import '../../domain/entities/board_geometry.dart';
import '../../domain/entities/scan_debug_trace.dart';
import '../../domain/entities/scan_image.dart';
import '../../domain/services/board_detector.dart';
import 'statistical_board_detector.dart';

/// Hybrid detector:
/// 1) runs the statistical detector on the original image
/// 2) runs OpenCV pre-processing variants
/// 3) keeps the best geometry score
///
/// If OpenCV is unavailable on the platform, it falls back transparently.
class OpenCvHybridBoardDetector implements BoardDetector {
  OpenCvHybridBoardDetector({
    BoardDetector? fallback,
    this.enableOpenCv = true,
    this.enableCornerRefinement = true,
    this.checkerTargetSize = 256,
    this.maxOpenCvVariants = 6,
    this.useLightPreprocessSet = false,
    this.minBoardConfidence = 0.20,
    this.minBoardConfidenceLineFallback = 0.24,
    this.rescueMode = false,
  }) : _fallback = fallback ?? const StatisticalBoardDetector();

  final BoardDetector _fallback;
  final bool enableOpenCv;
  final bool enableCornerRefinement;
  final int checkerTargetSize;
  final int maxOpenCvVariants;
  final bool useLightPreprocessSet;
  final double minBoardConfidence;
  final double minBoardConfidenceLineFallback;
  final bool rescueMode;

  @override
  Future<BoardGeometry> detect(ScanInputImage image) async {
    final baselineRaw = await _fallback.detect(image);
    final decodedOriginal = await _decodeLuma(image.bytes);
    _DecodedLuma? bestDecoded = decodedOriginal;

    _ScoredGeometry best = _ScoredGeometry(
      geometry: baselineRaw,
      score: _scoreGeometry(geometry: baselineRaw, decoded: decodedOriginal),
      source: 'baseline',
      decodeBytes: image.bytes,
    );

    if (!enableOpenCv || image.path.isEmpty) {
      final decodedForRefine =
          bestDecoded ?? await _decodeLuma(best.decodeBytes);
      return _refineAfterSelection(
        best: best,
        decoded: decodedForRefine,
        imagePath: image.path,
        imageBytes: image.bytes,
      );
    }

    final pathFrom = _resolvePathFrom(image.path);
    final variants = await _buildOpenCvVariants(
      pathFrom: pathFrom,
      path: image.path,
    );

    for (final variant in variants) {
      try {
        final geometryRaw = await _fallback.detect(
          ScanInputImage(path: image.path, bytes: variant.bytes),
        );
        final variantDecoded =
            decodedOriginal ?? await _decodeLuma(variant.bytes);
        final score = _scoreGeometry(
          geometry: geometryRaw,
          decoded: variantDecoded,
        );
        if (score > best.score) {
          best = _ScoredGeometry(
            geometry: geometryRaw,
            score: score,
            source: variant.label,
            decodeBytes: variant.bytes,
          );
          bestDecoded = variantDecoded;
        }
      } catch (_) {
        // Skip this variant and continue.
      }
    }

    final decodedForRefine = bestDecoded ?? await _decodeLuma(best.decodeBytes);
    return _refineAfterSelection(
      best: best,
      decoded: decodedForRefine,
      imagePath: image.path,
      imageBytes: image.bytes,
    );
  }

  Future<BoardGeometry> _refineAfterSelection({
    required _ScoredGeometry best,
    required _DecodedLuma? decoded,
    required String imagePath,
    required Uint8List imageBytes,
  }) async {
    if (decoded == null || !best.geometry.isValid) {
      final reasons = <String>[];
      if (decoded == null) {
        reasons.add('decoded_null');
      }
      if (!best.geometry.isValid) {
        reasons.add('invalid_geometry');
      }
      ScanDebugTrace.instance.record(
        'detector=hybrid early_return=true source=${best.source} '
        'reason=${reasons.join("|")}',
      );
      return best.geometry;
    }

    if (!enableCornerRefinement) {
      final quality = _geometryQualitySummary(
        decoded: decoded,
        geometry: best.geometry,
      );
      final confidence = _boardDetectionConfidence(
        decoded: decoded,
        geometry: best.geometry,
        quality: quality,
        lineFallbackAccepted: false,
        forcedProfileAccepted: false,
      );
      final sideBalance = _edgeSideBalanceScore(
        decoded: decoded,
        geometry: best.geometry,
      );
      final rejectReason = _boardRejectReason(
        decoded: decoded,
        geometry: best.geometry,
        quality: quality,
        confidence: confidence,
        lineFallbackAccepted: false,
        forcedProfileAccepted: false,
        sideBalance: sideBalance,
      );
      final rejected = rejectReason != null;
      final areaRatio =
          _quadArea(best.geometry.corners) /
          math.max(1.0, (decoded.width * decoded.height).toDouble());
      final finalGeometry = rejected
          ? const BoardGeometry(corners: <BoardCorner>[])
          : best.geometry;
      ScanDebugTrace.instance.record(
        'detector=hybrid early_return=true source=${best.source} '
        'reason=corner_refine_disabled '
        'quick_attempts=skip '
        'board_area_ratio=${areaRatio.toStringAsFixed(3)} '
        'board_confidence=${confidence.toStringAsFixed(3)} '
        'board_checker=${quality.checker.toStringAsFixed(3)} '
        'board_regularity=${quality.regularity.toStringAsFixed(3)} '
        'board_edge_frame=${quality.edgeFrame.toStringAsFixed(3)} '
        'board_quality=${quality.combined.toStringAsFixed(3)} '
        'board_side_balance=${sideBalance.toStringAsFixed(3)} '
        'board_rejected=$rejected '
        'reject=${rejectReason ?? 'none'} '
        'which_path_won=${rejected ? "rejected_no_board" : "baseline"}',
      );
      return finalGeometry;
    }

    final initialScore = _scoreGeometry(
      geometry: best.geometry,
      decoded: decoded,
    );
    final localRefined = _refineGeometryForScore(
      geometry: best.geometry,
      decoded: decoded,
    );
    final localScore = _scoreGeometry(geometry: localRefined, decoded: decoded);
    final nativeOutcome = await _refineCornersWithNativeOpenCv(
      decoded: decoded,
      initialGeometry: localRefined,
      imagePath: imagePath,
      imageBytes: imageBytes,
      sourceLabel: best.source,
    );
    final nativeScore = _scoreGeometry(
      geometry: nativeOutcome.geometry,
      decoded: decoded,
    );
    final roiBase = nativeScore > localScore
        ? nativeOutcome.geometry
        : localRefined;
    final roiBaseScore = math.max(localScore, nativeScore);
    final roiOutcome = _refineCornersInRoi(
      decoded: decoded,
      initialGeometry: roiBase,
      initialScore: roiBaseScore,
      imagePath: imagePath,
      sourceLabel: best.source,
    );
    final roiScore = _scoreGeometry(
      geometry: roiOutcome.geometry,
      decoded: decoded,
    );

    var finalGeometry = localRefined;
    var finalScore = localScore;
    var whichPathWon = 'local';
    if (nativeScore > finalScore) {
      finalScore = nativeScore;
      finalGeometry = nativeOutcome.geometry;
      whichPathWon = 'native';
    }
    if (roiScore > finalScore) {
      finalScore = roiScore;
      finalGeometry = roiOutcome.geometry;
      whichPathWon = 'roi';
    }

    final innerRefined = _refineInnerBoardWindow(
      decoded: decoded,
      geometry: finalGeometry,
    );
    final innerScore = _scoreGeometry(geometry: innerRefined, decoded: decoded);
    if (innerScore > finalScore + 0.004) {
      finalScore = innerScore;
      finalGeometry = innerRefined;
      whichPathWon = 'inner';
    }

    var forcedProfileTriggered = false;
    var forcedProfileAccepted = false;
    var forcedProfileSource = 'none';
    final finalQuality = _geometryQualitySummary(
      decoded: decoded,
      geometry: finalGeometry,
    );
    final forcedCandidates = <_ForcedCandidate>[
      _ForcedCandidate(
        label: 'final',
        geometry: finalGeometry,
        quality: finalQuality,
      ),
      _ForcedCandidate(
        label: 'roi',
        geometry: roiOutcome.geometry,
        quality: _geometryQualitySummary(
          decoded: decoded,
          geometry: roiOutcome.geometry,
        ),
      ),
      _ForcedCandidate(
        label: 'native',
        geometry: nativeOutcome.geometry,
        quality: _geometryQualitySummary(
          decoded: decoded,
          geometry: nativeOutcome.geometry,
        ),
      ),
      _ForcedCandidate(
        label: 'local',
        geometry: localRefined,
        quality: _geometryQualitySummary(
          decoded: decoded,
          geometry: localRefined,
        ),
      ),
    ];
    _ForcedCandidate? bestForced;
    double bestForcedScore = finalScore;
    _GeometryQualitySummary bestForcedQuality = finalQuality;
    final forcedAttempts = <String>[];

    for (final candidate in forcedCandidates) {
      if (!_shouldForceInnerProfile(
        decoded: decoded,
        geometry: candidate.geometry,
        quality: candidate.quality,
      )) {
        forcedAttempts.add('${candidate.label}:skip');
        continue;
      }
      forcedProfileTriggered = true;
      forcedAttempts.add('${candidate.label}:trigger');

      final forcedInner = _forceInnerBoardWindowFromProfiles(
        decoded: decoded,
        geometry: candidate.geometry,
      );
      final forcedOrdered = _orderCornersTlTrBrBl(forcedInner.corners);
      final forcedValid = _isReasonableQuad(
        forcedOrdered,
        width: decoded.width,
        height: decoded.height,
      );
      if (!forcedValid) {
        forcedAttempts.add('${candidate.label}:invalid');
        continue;
      }
      final forcedGeometry = BoardGeometry(corners: forcedOrdered);
      final forcedQuality = _geometryQualitySummary(
        decoded: decoded,
        geometry: forcedGeometry,
      );
      final forcedScore = _scoreGeometry(
        geometry: forcedGeometry,
        decoded: decoded,
      );
      final forcedImproves = _shouldAcceptForcedProfile(
        finalQuality: candidate.quality,
        forcedQuality: forcedQuality,
      );
      if (!forcedImproves) {
        forcedAttempts.add(
          '${candidate.label}:reject('
          'b=${candidate.quality.combined.toStringAsFixed(3)}>'
          '${forcedQuality.combined.toStringAsFixed(3)})',
        );
        continue;
      }
      forcedAttempts.add(
        '${candidate.label}:accept('
        'b=${candidate.quality.combined.toStringAsFixed(3)}>'
        '${forcedQuality.combined.toStringAsFixed(3)})',
      );

      if (bestForced == null ||
          _isBetterForcedCandidate(
            currentQuality: bestForcedQuality,
            currentScore: bestForcedScore,
            candidateQuality: forcedQuality,
            candidateScore: forcedScore,
          )) {
        bestForced = _ForcedCandidate(
          label: candidate.label,
          geometry: forcedGeometry,
          quality: forcedQuality,
        );
        bestForcedQuality = forcedQuality;
        bestForcedScore = forcedScore;
      }
    }

    if (bestForced != null) {
      forcedProfileAccepted = true;
      forcedProfileSource = bestForced.label;
      finalScore = bestForcedScore;
      finalGeometry = bestForced.geometry;
      whichPathWon = 'forced_profile';
    }

    var lineFallbackTriggered = false;
    var lineFallbackAccepted = false;
    var lineFallbackSource = 'none';
    var lineFallbackReason = 'none';
    final lineAttempts = <String>[];
    final lineCandidates = <_ForcedCandidate>[
      _ForcedCandidate(
        label: 'final',
        geometry: finalGeometry,
        quality: _geometryQualitySummary(
          decoded: decoded,
          geometry: finalGeometry,
        ),
      ),
      _ForcedCandidate(
        label: 'roi',
        geometry: roiOutcome.geometry,
        quality: _geometryQualitySummary(
          decoded: decoded,
          geometry: roiOutcome.geometry,
        ),
      ),
      _ForcedCandidate(
        label: 'native',
        geometry: nativeOutcome.geometry,
        quality: _geometryQualitySummary(
          decoded: decoded,
          geometry: nativeOutcome.geometry,
        ),
      ),
      _ForcedCandidate(
        label: 'local',
        geometry: localRefined,
        quality: _geometryQualitySummary(
          decoded: decoded,
          geometry: localRefined,
        ),
      ),
    ];
    _LineFallbackCandidate? bestLineCandidate;
    final globalLineTrigger = _shouldRunLineFallback(
      decoded: decoded,
      geometry: finalGeometry,
      quality: finalQuality,
    );
    if (!globalLineTrigger) {
      lineAttempts.add(
        'global:skip('
        'b=${finalQuality.combined.toStringAsFixed(3)} '
        'chk=${finalQuality.checker.toStringAsFixed(3)} '
        'reg=${finalQuality.regularity.toStringAsFixed(3)})',
      );
    } else {
      for (final candidate in lineCandidates) {
        if (!_shouldRunLineFallback(
          decoded: decoded,
          geometry: candidate.geometry,
          quality: candidate.quality,
        )) {
          lineAttempts.add('${candidate.label}:skip');
          continue;
        }
        lineFallbackTriggered = true;
        final lineOutcome = _refineLineBasedInWarp(
          decoded: decoded,
          geometry: candidate.geometry,
        );
        if (!lineOutcome.geometry.isValid) {
          lineAttempts.add('${candidate.label}:invalid(${lineOutcome.reason})');
          continue;
        }
        final ordered = _orderCornersTlTrBrBl(lineOutcome.geometry.corners);
        if (!_isReasonableQuad(
          ordered,
          width: decoded.width,
          height: decoded.height,
        )) {
          lineAttempts.add(
            '${candidate.label}:invalid_quad(${lineOutcome.reason})',
          );
          continue;
        }
        final refinedGeometry = BoardGeometry(corners: ordered);
        final refinedQuality = _geometryQualitySummary(
          decoded: decoded,
          geometry: refinedGeometry,
        );
        if (!_shouldAcceptLineFallback(
          decoded: decoded,
          baseGeometry: candidate.geometry,
          baseQuality: candidate.quality,
          refinedGeometry: refinedGeometry,
          refinedQuality: refinedQuality,
        )) {
          lineAttempts.add(
            '${candidate.label}:reject('
            'b=${candidate.quality.combined.toStringAsFixed(3)}->'
            '${refinedQuality.combined.toStringAsFixed(3)})',
          );
          continue;
        }
        final refinedScore = _scoreGeometry(
          geometry: refinedGeometry,
          decoded: decoded,
        );
        lineAttempts.add(
          '${candidate.label}:accept('
          'b=${candidate.quality.combined.toStringAsFixed(3)}->'
          '${refinedQuality.combined.toStringAsFixed(3)})',
        );
        final currentBest = bestLineCandidate;
        if (currentBest == null ||
            _isBetterLineFallbackCandidate(
              current: currentBest,
              candidateGeometry: refinedGeometry,
              candidateQuality: refinedQuality,
              candidateScore: refinedScore,
            )) {
          bestLineCandidate = _LineFallbackCandidate(
            label: candidate.label,
            geometry: refinedGeometry,
            quality: refinedQuality,
            score: refinedScore,
            reason: lineOutcome.reason,
          );
        }
      }
    }
    if (bestLineCandidate != null) {
      lineFallbackAccepted = true;
      lineFallbackSource = bestLineCandidate.label;
      lineFallbackReason = bestLineCandidate.reason;
      var bestLineGeometry = bestLineCandidate.geometry;
      var bestLineScore = bestLineCandidate.score;
      final lineNativeOutcome = await _refineCornersWithNativeOpenCv(
        decoded: decoded,
        initialGeometry: bestLineGeometry,
        imagePath: imagePath,
        imageBytes: imageBytes,
        sourceLabel: 'line_${bestLineCandidate.label}',
      );
      final lineNativeScore = _scoreGeometry(
        geometry: lineNativeOutcome.geometry,
        decoded: decoded,
      );
      if (lineNativeScore > bestLineScore + 0.001) {
        bestLineGeometry = lineNativeOutcome.geometry;
        bestLineScore = lineNativeScore;
        lineFallbackReason =
            '${bestLineCandidate.reason}|native=${lineNativeOutcome.reason}';
      } else {
        lineFallbackReason =
            '${bestLineCandidate.reason}|native_keep=${lineNativeOutcome.reason}';
      }
      final lineRoiOutcome = _refineCornersInRoi(
        decoded: decoded,
        initialGeometry: bestLineGeometry,
        initialScore: bestLineScore,
        imagePath: imagePath,
        sourceLabel: 'line_${bestLineCandidate.label}',
      );
      final lineRoiScore = _scoreGeometry(
        geometry: lineRoiOutcome.geometry,
        decoded: decoded,
      );
      if (lineRoiScore > bestLineScore + 0.001) {
        bestLineGeometry = lineRoiOutcome.geometry;
        bestLineScore = lineRoiScore;
        lineFallbackReason = '$lineFallbackReason|roi=${lineRoiOutcome.reason}';
      } else {
        lineFallbackReason =
            '$lineFallbackReason|roi_keep=${lineRoiOutcome.reason}';
      }
      finalGeometry = bestLineGeometry;
      finalScore = bestLineScore;
      whichPathWon = 'line_fallback';
    }

    var rejectedNoBoard = false;
    var boardRejectReason = 'none';
    final finalAreaRatioBeforeGate =
        _quadArea(finalGeometry.corners) /
        math.max(1.0, (decoded.width * decoded.height).toDouble());
    final finalQualityNow = _geometryQualitySummary(
      decoded: decoded,
      geometry: finalGeometry,
    );
    final finalConfidence = _boardDetectionConfidence(
      decoded: decoded,
      geometry: finalGeometry,
      quality: finalQualityNow,
      lineFallbackAccepted: lineFallbackAccepted,
      forcedProfileAccepted: forcedProfileAccepted,
    );
    final finalSideBalance = _edgeSideBalanceScore(
      decoded: decoded,
      geometry: finalGeometry,
    );
    final rejectReason = _boardRejectReason(
      decoded: decoded,
      geometry: finalGeometry,
      quality: finalQualityNow,
      confidence: finalConfidence,
      lineFallbackAccepted: lineFallbackAccepted,
      forcedProfileAccepted: forcedProfileAccepted,
      sideBalance: finalSideBalance,
    );
    if (rejectReason != null) {
      rejectedNoBoard = true;
      boardRejectReason = rejectReason;
      finalGeometry = const BoardGeometry(corners: <BoardCorner>[]);
      finalScore = 0.0;
      whichPathWon = 'rejected_no_board';
    }

    if (kDebugMode) {
      debugPrint(
        '[scan][hybrid] source=${best.source} score_init=${initialScore.toStringAsFixed(4)} '
        'score_local=${localScore.toStringAsFixed(4)} score_native=${nativeScore.toStringAsFixed(4)} '
        'score_roi=${roiScore.toStringAsFixed(4)} score_inner=${innerScore.toStringAsFixed(4)} '
        'score_final=${finalScore.toStringAsFixed(4)} '
        'quality_final=${finalQualityNow.combined.toStringAsFixed(4)} '
        'area_ratio=${finalAreaRatioBeforeGate.toStringAsFixed(4)} '
        'conf=${finalConfidence.toStringAsFixed(4)} '
        'side_balance=${finalSideBalance.toStringAsFixed(4)} '
        'rejected_no_board=$rejectedNoBoard reject=$boardRejectReason',
      );
      debugPrint(
        '[scan][hybrid] quad_initial=${_cornersLabel(best.geometry.corners)} '
        'quad_final=${_cornersLabel(finalGeometry.corners)} '
        'native_refine=${nativeOutcome.reason} roi_refine=${roiOutcome.reason} '
        'forced_profile_triggered=$forcedProfileTriggered '
        'forced_profile_accepted=$forcedProfileAccepted '
        'forced_profile_source=$forcedProfileSource '
        'forced_profile_attempts=${forcedAttempts.join(",")} '
        'line_fallback_triggered=$lineFallbackTriggered '
        'line_fallback_accepted=$lineFallbackAccepted '
        'line_fallback_source=$lineFallbackSource '
        'line_fallback_reason=$lineFallbackReason '
        'line_fallback_attempts=${lineAttempts.join(",")} '
        'board_area_ratio=${finalAreaRatioBeforeGate.toStringAsFixed(3)} '
        'board_confidence=${finalConfidence.toStringAsFixed(3)} '
        'board_checker=${finalQualityNow.checker.toStringAsFixed(3)} '
        'board_regularity=${finalQualityNow.regularity.toStringAsFixed(3)} '
        'board_edge_frame=${finalQualityNow.edgeFrame.toStringAsFixed(3)} '
        'board_quality=${finalQualityNow.combined.toStringAsFixed(3)} '
        'board_side_balance=${finalSideBalance.toStringAsFixed(3)} '
        'board_rejected=$rejectedNoBoard '
        'which_path_won=$whichPathWon',
      );
    }
    ScanDebugTrace.instance.record(
      'detector=hybrid source=${best.source} '
      'forced_profile_triggered=$forcedProfileTriggered '
      'forced_profile_accepted=$forcedProfileAccepted '
      'forced_profile_source=$forcedProfileSource '
      'forced_profile_attempts=${forcedAttempts.join(",")} '
      'line_fallback_triggered=$lineFallbackTriggered '
      'line_fallback_accepted=$lineFallbackAccepted '
      'line_fallback_source=$lineFallbackSource '
      'line_fallback_reason=$lineFallbackReason '
      'line_fallback_attempts=${lineAttempts.join(",")} '
      'board_area_ratio=${finalAreaRatioBeforeGate.toStringAsFixed(3)} '
      'board_confidence=${finalConfidence.toStringAsFixed(3)} '
      'board_checker=${finalQualityNow.checker.toStringAsFixed(3)} '
      'board_regularity=${finalQualityNow.regularity.toStringAsFixed(3)} '
      'board_edge_frame=${finalQualityNow.edgeFrame.toStringAsFixed(3)} '
      'board_quality=${finalQualityNow.combined.toStringAsFixed(3)} '
      'board_side_balance=${finalSideBalance.toStringAsFixed(3)} '
      'board_rejected=$rejectedNoBoard '
      'reject=$boardRejectReason '
      'which_path_won=$whichPathWon '
      'native_refine=${nativeOutcome.reason} '
      'roi_refine=${roiOutcome.reason}',
    );
    return finalGeometry;
  }

  bool _isLikelyAxisAlignedRectangle(BoardGeometry geometry) {
    if (!geometry.isValid || geometry.corners.length != 4) {
      return false;
    }
    final c = _orderCornersTlTrBrBl(geometry.corners);
    final topSlope = (c[1].y - c[0].y).abs();
    final bottomSlope = (c[2].y - c[3].y).abs();
    final leftSlope = (c[3].x - c[0].x).abs();
    final rightSlope = (c[2].x - c[1].x).abs();
    return topSlope < 3.0 &&
        bottomSlope < 3.0 &&
        leftSlope < 3.0 &&
        rightSlope < 3.0;
  }

  bool _shouldForceInnerProfile({
    required _DecodedLuma decoded,
    required BoardGeometry geometry,
    required _GeometryQualitySummary quality,
  }) {
    if (!geometry.isValid || geometry.corners.length != 4) {
      return false;
    }
    final axisAligned = _isLikelyAxisAlignedRectangle(geometry);
    final areaRatio =
        _quadArea(geometry.corners) /
        math.max(1.0, (decoded.width * decoded.height).toDouble());
    final marginX = math.max(2.0, decoded.width * 0.015);
    final marginY = math.max(2.0, decoded.height * 0.015);
    int nearBorderCorners = 0;
    for (final c in geometry.corners) {
      if (c.x < marginX ||
          c.y < marginY ||
          c.x > decoded.width - marginX ||
          c.y > decoded.height - marginY) {
        nearBorderCorners += 1;
      }
    }
    final lowQuality = quality.combined < 0.36 || quality.checker < 0.30;
    final veryLowQuality =
        quality.combined < 0.31 ||
        quality.checker < 0.24 ||
        quality.regularity < 0.12;
    final suspiciousCoverage = areaRatio > 0.23;
    final suspiciousBorder = nearBorderCorners >= 2;
    if (axisAligned) {
      return lowQuality || suspiciousCoverage || suspiciousBorder;
    }
    final suspiciousNonAxis =
        veryLowQuality && (suspiciousCoverage || suspiciousBorder);
    return suspiciousNonAxis;
  }

  bool _isBetterForcedCandidate({
    required _GeometryQualitySummary currentQuality,
    required double currentScore,
    required _GeometryQualitySummary candidateQuality,
    required double candidateScore,
  }) {
    final currentStrong =
        currentQuality.checker >= 0.20 &&
        currentQuality.regularity >= 0.09 &&
        currentQuality.edgeFrame >= 0.04;
    final candidateStrong =
        candidateQuality.checker >= 0.20 &&
        candidateQuality.regularity >= 0.09 &&
        candidateQuality.edgeFrame >= 0.04;

    if (candidateStrong && !currentStrong) {
      return true;
    }
    if (candidateQuality.combined > currentQuality.combined + 0.006) {
      return true;
    }
    if (candidateQuality.checker > currentQuality.checker + 0.020 &&
        candidateQuality.regularity >= currentQuality.regularity - 0.015) {
      return true;
    }
    return candidateScore > currentScore + 0.015;
  }

  bool _shouldAcceptForcedProfile({
    required _GeometryQualitySummary finalQuality,
    required _GeometryQualitySummary forcedQuality,
  }) {
    final minimumQuality =
        forcedQuality.combined >= 0.18 && forcedQuality.checker >= 0.10;
    if (!minimumQuality) {
      return false;
    }

    // Forced path is enabled specifically for difficult cases where the base
    // score can be biased by an oversized axis-aligned candidate.
    if (forcedQuality.combined >= finalQuality.combined - 0.020 &&
        forcedQuality.checker >= finalQuality.checker - 0.010) {
      return true;
    }
    return forcedQuality.checker > finalQuality.checker + 0.03 &&
        forcedQuality.regularity >= finalQuality.regularity - 0.03;
  }

  bool _shouldRunLineFallback({
    required _DecodedLuma decoded,
    required BoardGeometry geometry,
    required _GeometryQualitySummary quality,
  }) {
    if (!geometry.isValid || geometry.corners.length != 4) {
      return false;
    }
    final areaRatio =
        _quadArea(geometry.corners) /
        math.max(1.0, (decoded.width * decoded.height).toDouble());
    final marginX = math.max(2.0, decoded.width * 0.015);
    final marginY = math.max(2.0, decoded.height * 0.015);
    int nearBorderCorners = 0;
    for (final c in geometry.corners) {
      if (c.x < marginX ||
          c.y < marginY ||
          c.x > decoded.width - marginX ||
          c.y > decoded.height - marginY) {
        nearBorderCorners += 1;
      }
    }

    // Proxy for warp/orientation failure before validation stage.
    final likelyFunctionalFail =
        quality.combined < 0.30 ||
        (quality.checker < 0.20 && quality.regularity < 0.12);
    final currentBLow = quality.combined < 0.35;
    final suspiciousQuad =
        _isLikelyAxisAlignedRectangle(geometry) ||
        areaRatio > 0.22 ||
        nearBorderCorners >= 2;
    return (likelyFunctionalFail || currentBLow) && suspiciousQuad;
  }

  bool _shouldAcceptLineFallback({
    required _DecodedLuma decoded,
    required BoardGeometry baseGeometry,
    required _GeometryQualitySummary baseQuality,
    required BoardGeometry refinedGeometry,
    required _GeometryQualitySummary refinedQuality,
  }) {
    final combinedGain = refinedQuality.combined - baseQuality.combined;
    final regularityGain = refinedQuality.regularity - baseQuality.regularity;
    final checkerGain = refinedQuality.checker - baseQuality.checker;

    if (combinedGain > 0.010) {
      return true;
    }

    final likelyFail =
        baseQuality.combined < 0.34 ||
        baseQuality.regularity < 0.16 ||
        baseQuality.checker < 0.24;
    if (likelyFail && refinedQuality.combined >= baseQuality.combined - 0.005) {
      final baseSuspicion = _geometrySuspicionScore(
        decoded: decoded,
        geometry: baseGeometry,
        quality: baseQuality,
      );
      final refinedSuspicion = _geometrySuspicionScore(
        decoded: decoded,
        geometry: refinedGeometry,
        quality: refinedQuality,
      );
      if (refinedSuspicion + 0.08 <= baseSuspicion) {
        return true;
      }
    }

    // Hard-case override: allow strong geometric gain even if checker is noisy
    // because overlays/glow can suppress checker parity while lines improve.
    if (combinedGain >= 0.040 && regularityGain >= 0.015) {
      return true;
    }

    if (refinedQuality.combined < 0.18) {
      return false;
    }
    if (refinedQuality.checker < 0.08 && regularityGain < 0.030) {
      return false;
    }
    if (combinedGain >= 0.012) {
      return true;
    }
    return checkerGain > 0.028 &&
        refinedQuality.regularity >= baseQuality.regularity - 0.035;
  }

  double _geometrySuspicionScore({
    required _DecodedLuma decoded,
    required BoardGeometry geometry,
    required _GeometryQualitySummary quality,
  }) {
    if (!geometry.isValid || geometry.corners.length != 4) {
      return 1.0;
    }
    final areaRatio =
        _quadArea(geometry.corners) /
        math.max(1.0, (decoded.width * decoded.height).toDouble());
    final marginX = math.max(2.0, decoded.width * 0.015);
    final marginY = math.max(2.0, decoded.height * 0.015);
    int nearBorderCorners = 0;
    for (final c in geometry.corners) {
      if (c.x < marginX ||
          c.y < marginY ||
          c.x > decoded.width - marginX ||
          c.y > decoded.height - marginY) {
        nearBorderCorners += 1;
      }
    }
    final axisPenalty = _isLikelyAxisAlignedRectangle(geometry) ? 0.28 : 0.0;
    final borderPenalty = (nearBorderCorners / 4.0) * 0.34;
    final areaPenalty = (areaRatio > 0.22 ? ((areaRatio - 0.22) / 0.50) : 0.0)
        .clamp(0.0, 0.30)
        .toDouble();
    final qualityPenalty = (1.0 - quality.combined).clamp(0.0, 1.0) * 0.24;
    return (axisPenalty + borderPenalty + areaPenalty + qualityPenalty)
        .clamp(0.0, 1.0)
        .toDouble();
  }

  double _boardDetectionConfidence({
    required _DecodedLuma decoded,
    required BoardGeometry geometry,
    required _GeometryQualitySummary quality,
    required bool lineFallbackAccepted,
    required bool forcedProfileAccepted,
  }) {
    if (!geometry.isValid || geometry.corners.length != 4) {
      return 0.0;
    }
    final areaRatio =
        _quadArea(geometry.corners) /
        math.max(1.0, (decoded.width * decoded.height).toDouble());
    final coverage = (1.0 - ((areaRatio - 0.22).abs() / 0.28))
        .clamp(0.0, 1.0)
        .toDouble();
    final suspicion = _geometrySuspicionScore(
      decoded: decoded,
      geometry: geometry,
      quality: quality,
    );
    var confidence =
        (quality.checker * 0.58) +
        (quality.combined * 0.27) +
        (quality.edgeFrame * 0.10) +
        (coverage * 0.05);
    confidence *= (1.0 - (suspicion * 0.28)).clamp(0.65, 1.0).toDouble();
    if (lineFallbackAccepted) {
      confidence += 0.05;
    }
    if (forcedProfileAccepted) {
      confidence += 0.02;
    }
    return confidence.clamp(0.0, 1.0).toDouble();
  }

  String? _boardRejectReason({
    required _DecodedLuma decoded,
    required BoardGeometry geometry,
    required _GeometryQualitySummary quality,
    required double confidence,
    required bool lineFallbackAccepted,
    required bool forcedProfileAccepted,
    required double sideBalance,
  }) {
    if (!geometry.isValid || geometry.corners.length != 4) {
      return 'invalid_geometry';
    }
    final areaRatio =
        _quadArea(geometry.corners) /
        math.max(1.0, (decoded.width * decoded.height).toDouble());
    if (areaRatio < 0.05) {
      return 'area_too_small';
    }
    if (areaRatio > 0.95 &&
        quality.checker < 0.22 &&
        quality.edgeFrame < 0.55) {
      return 'area_too_large_weak_structure';
    }
    final minConfidence = lineFallbackAccepted
        ? minBoardConfidenceLineFallback
        : minBoardConfidence;
    if (confidence < minConfidence) {
      return 'confidence_below_min';
    }
    final minLineCombined = rescueMode ? 0.25 : 0.26;
    final veryStrongFrameRescue =
        rescueMode && areaRatio >= 0.20 && quality.edgeFrame >= 0.80;
    final strongFramePattern =
        quality.regularity >= 0.38 && quality.edgeFrame >= 0.70;
    if (!lineFallbackAccepted &&
        quality.checker < 0.19 &&
        quality.combined < 0.34 &&
        !strongFramePattern &&
        !veryStrongFrameRescue) {
      return 'no_line_low_checker_combined';
    }
    if (lineFallbackAccepted && quality.combined < minLineCombined) {
      return 'line_low_combined';
    }
    if (lineFallbackAccepted &&
        quality.checker < 0.08 &&
        quality.regularity < 0.36) {
      return 'line_low_checker_regularity';
    }
    if (lineFallbackAccepted && areaRatio < 0.08) {
      return 'line_area_too_small';
    }
    if (lineFallbackAccepted && quality.checker < 0.17 && areaRatio > 0.22) {
      return 'line_large_area_low_checker';
    }
    final minLineSmallArea = rescueMode ? 0.12 : 0.16;
    final minLineSmallEdge = rescueMode ? 0.50 : 0.60;
    if (lineFallbackAccepted &&
        areaRatio < minLineSmallArea &&
        quality.edgeFrame < minLineSmallEdge) {
      return 'line_small_area_low_edge';
    }
    if (forcedProfileAccepted &&
        confidence < 0.30 &&
        quality.edgeFrame < 0.60) {
      return 'forced_profile_low_conf_edge';
    }
    if (confidence < 0.33 && sideBalance < 0.16) {
      return 'low_side_balance';
    }
    if (confidence < 0.24 && sideBalance < 0.20 && quality.regularity < 0.38) {
      return 'low_conf_low_side_balance_regularity';
    }
    final strongFramePatternForFinal =
        quality.regularity >= 0.38 && quality.edgeFrame >= 0.70;
    if (!lineFallbackAccepted &&
        quality.checker < 0.30 &&
        confidence < 0.28 &&
        sideBalance < 0.22 &&
        !strongFramePatternForFinal) {
      return 'no_line_low_checker_conf_side_balance';
    }
    return null;
  }

  double _edgeSideBalanceScore({
    required _DecodedLuma decoded,
    required BoardGeometry geometry,
  }) {
    if (!geometry.isValid || geometry.corners.length != 4) {
      return 0.0;
    }
    final warped = _warpBoardLuma(
      decoded: decoded,
      corners: _orderCornersTlTrBrBl(geometry.corners),
      targetSize: checkerTargetSize.clamp(128, 192),
    );
    if (warped == null || warped.size < 24) {
      return 0.0;
    }
    final n = warped.size;
    final strip = math.max(2, (n * 0.08).round());
    double topSum = 0;
    int topCount = 0;
    double bottomSum = 0;
    int bottomCount = 0;
    double leftSum = 0;
    int leftCount = 0;
    double rightSum = 0;
    int rightCount = 0;

    for (int y = 1; y < n - 1; y++) {
      final rowOffset = y * n;
      for (int x = 1; x < n - 1; x++) {
        final idx = rowOffset + x;
        final gx = (warped.luma[idx + 1] - warped.luma[idx - 1]).abs();
        final gy = (warped.luma[idx + n] - warped.luma[idx - n]).abs();
        final g = gx + gy;
        if (y < strip) {
          topSum += g;
          topCount += 1;
        }
        if (y >= n - strip) {
          bottomSum += g;
          bottomCount += 1;
        }
        if (x < strip) {
          leftSum += g;
          leftCount += 1;
        }
        if (x >= n - strip) {
          rightSum += g;
          rightCount += 1;
        }
      }
    }

    if (topCount == 0 ||
        bottomCount == 0 ||
        leftCount == 0 ||
        rightCount == 0) {
      return 0.0;
    }
    final sides = <double>[
      topSum / topCount,
      bottomSum / bottomCount,
      leftSum / leftCount,
      rightSum / rightCount,
    ];
    final minSide = sides.reduce(math.min);
    final maxSide = sides.reduce(math.max);
    if (maxSide <= 1e-6) {
      return 0.0;
    }
    return (minSide / maxSide).clamp(0.0, 1.0).toDouble();
  }

  bool _isBetterLineFallbackCandidate({
    required _LineFallbackCandidate current,
    required BoardGeometry candidateGeometry,
    required _GeometryQualitySummary candidateQuality,
    required double candidateScore,
  }) {
    if (candidateQuality.combined > current.quality.combined + 0.010) {
      return true;
    }
    if (candidateQuality.checker > current.quality.checker + 0.022 &&
        candidateQuality.regularity >= current.quality.regularity - 0.020) {
      return true;
    }
    if (candidateScore > current.score + 0.020) {
      return true;
    }
    return _quadArea(candidateGeometry.corners) >
        _quadArea(current.geometry.corners);
  }

  _RefineOutcome _refineLineBasedInWarp({
    required _DecodedLuma decoded,
    required BoardGeometry geometry,
  }) {
    if (!geometry.isValid || geometry.corners.length != 4) {
      return _RefineOutcome(
        geometry: geometry,
        reason: 'fallback_line_invalid_geometry',
      );
    }
    final base = _orderCornersTlTrBrBl(geometry.corners);
    if (!_isReasonableQuad(
      base,
      width: decoded.width,
      height: decoded.height,
    )) {
      return _RefineOutcome(
        geometry: geometry,
        reason: 'fallback_line_invalid_quad',
      );
    }

    final warpSize = checkerTargetSize.clamp(128, 256);
    final warped = _warpBoardLuma(
      decoded: decoded,
      corners: base,
      targetSize: warpSize,
    );
    final homography = _homographyUnitToQuad(base);
    if (warped == null || homography == null) {
      return _RefineOutcome(
        geometry: geometry,
        reason: 'fallback_line_warp_failed',
      );
    }

    final n = warped.size;
    final cols = List<double>.filled(n, 0.0);
    final rows = List<double>.filled(n, 0.0);
    for (int y = 1; y < n - 1; y++) {
      final rowOffset = y * n;
      for (int x = 1; x < n - 1; x++) {
        final idx = rowOffset + x;
        final gx = (warped.luma[idx + 1] - warped.luma[idx - 1]).abs();
        final gy = (warped.luma[idx + n] - warped.luma[idx - n]).abs();
        final g = gx + gy;
        cols[x] += g;
        rows[y] += g;
      }
    }
    final smoothCols = _smoothProjection(cols, radius: 4);
    final smoothRows = _smoothProjection(rows, radius: 4);
    final leftCandidates = _topPeaksInRange(
      smoothCols,
      start: (n * 0.02).round(),
      end: (n * 0.45).round(),
      count: 4,
      minSpacing: math.max(6, (n * 0.06).round()),
    );
    final rightCandidates = _topPeaksInRange(
      smoothCols,
      start: (n * 0.55).round(),
      end: (n * 0.98).round(),
      count: 4,
      minSpacing: math.max(6, (n * 0.06).round()),
    );
    final topCandidates = _topPeaksInRange(
      smoothRows,
      start: (n * 0.02).round(),
      end: (n * 0.45).round(),
      count: 4,
      minSpacing: math.max(6, (n * 0.06).round()),
    );
    final bottomCandidates = _topPeaksInRange(
      smoothRows,
      start: (n * 0.55).round(),
      end: (n * 0.98).round(),
      count: 4,
      minSpacing: math.max(6, (n * 0.06).round()),
    );
    if (leftCandidates.isEmpty ||
        rightCandidates.isEmpty ||
        topCandidates.isEmpty ||
        bottomCandidates.isEmpty) {
      return _RefineOutcome(
        geometry: geometry,
        reason: 'fallback_line_no_peaks',
      );
    }

    double bestWindowScore = -double.infinity;
    int? bestLeft;
    int? bestRight;
    int? bestTop;
    int? bestBottom;
    for (final left in leftCandidates) {
      for (final right in rightCandidates) {
        if (right - left < (n * 0.30)) {
          continue;
        }
        for (final top in topCandidates) {
          for (final bottom in bottomCandidates) {
            if (bottom - top < (n * 0.30)) {
              continue;
            }
            final u0 = left / n;
            final v0 = top / n;
            final u1 = right / n;
            final v1 = bottom / n;
            final selectionScore = _windowSelectionScore(
              warped: warped,
              u0: u0,
              v0: v0,
              u1: u1,
              v1: v1,
            );
            final spanX = (right - left) / n;
            final spanY = (bottom - top) / n;
            final spanScore =
                ((1.0 - ((spanX - 0.80).abs() / 0.45)) * 0.5 +
                        (1.0 - ((spanY - 0.80).abs() / 0.45)) * 0.5)
                    .clamp(0.0, 1.0)
                    .toDouble();
            final totalScore = (selectionScore * 0.88) + (spanScore * 0.12);
            if (totalScore > bestWindowScore) {
              bestWindowScore = totalScore;
              bestLeft = left;
              bestRight = right;
              bestTop = top;
              bestBottom = bottom;
            }
          }
        }
      }
    }

    if (bestLeft == null ||
        bestRight == null ||
        bestTop == null ||
        bestBottom == null) {
      return _RefineOutcome(
        geometry: geometry,
        reason: 'fallback_line_no_valid_window',
      );
    }

    final p0 = _applyHomography(homography, bestLeft / n, bestTop / n);
    final p1 = _applyHomography(homography, bestRight / n, bestTop / n);
    final p2 = _applyHomography(homography, bestRight / n, bestBottom / n);
    final p3 = _applyHomography(homography, bestLeft / n, bestBottom / n);
    final candidate = _orderCornersTlTrBrBl(<BoardCorner>[
      BoardCorner(x: p0.x, y: p0.y),
      BoardCorner(x: p1.x, y: p1.y),
      BoardCorner(x: p2.x, y: p2.y),
      BoardCorner(x: p3.x, y: p3.y),
    ]);
    if (!_isReasonableQuad(
      candidate,
      width: decoded.width,
      height: decoded.height,
    )) {
      return _RefineOutcome(
        geometry: geometry,
        reason: 'fallback_line_invalid_candidate',
      );
    }
    return _RefineOutcome(
      geometry: BoardGeometry(corners: candidate),
      reason:
          'ok_line_profile(l=$bestLeft r=$bestRight t=$bestTop b=$bestBottom '
          'score=${bestWindowScore.toStringAsFixed(4)})',
    );
  }

  List<int> _topPeaksInRange(
    List<double> values, {
    required int start,
    required int end,
    required int count,
    required int minSpacing,
  }) {
    if (values.isEmpty) {
      return const <int>[];
    }
    final s = start.clamp(0, values.length - 1);
    final e = end.clamp(s, values.length - 1);
    final peaks = <int>[];
    for (
      int i = math.max(s + 1, 1);
      i <= math.min(e - 1, values.length - 2);
      i++
    ) {
      if (values[i] >= values[i - 1] && values[i] > values[i + 1]) {
        peaks.add(i);
      }
    }
    if (peaks.isEmpty) {
      final fallback = _peakIndexInRange(values, s, e);
      return fallback >= 0 ? <int>[fallback] : const <int>[];
    }
    peaks.sort((a, b) => values[b].compareTo(values[a]));
    final selected = <int>[];
    for (final p in peaks) {
      if (selected.every((q) => (q - p).abs() >= minSpacing)) {
        selected.add(p);
      }
      if (selected.length >= count) {
        break;
      }
    }
    selected.sort();
    return selected;
  }

  _GeometryQualitySummary _geometryQualitySummary({
    required _DecodedLuma decoded,
    required BoardGeometry geometry,
  }) {
    if (!geometry.isValid || geometry.corners.length != 4) {
      return const _GeometryQualitySummary(
        checker: 0,
        regularity: 0,
        edgeFrame: 0,
        combined: 0,
      );
    }
    final warped = _warpBoardLuma(
      decoded: decoded,
      corners: _orderCornersTlTrBrBl(geometry.corners),
      targetSize: checkerTargetSize.clamp(128, 256),
    );
    if (warped == null) {
      return const _GeometryQualitySummary(
        checker: 0,
        regularity: 0,
        edgeFrame: 0,
        combined: 0,
      );
    }
    final checker = _checkerboardWindowScore(
      warped: warped,
      u0: 0,
      v0: 0,
      u1: 1,
      v1: 1,
    );
    final regularity = _gridRegularityWindowScore(
      warped: warped,
      u0: 0,
      v0: 0,
      u1: 1,
      v1: 1,
    );
    final edgeFrame = _edgeFrameWindowScore(
      warped: warped,
      u0: 0,
      v0: 0,
      u1: 1,
      v1: 1,
    );
    final combined =
        ((checker * 0.45) + (regularity * 0.40) + (edgeFrame * 0.15))
            .clamp(0.0, 1.0)
            .toDouble();
    return _GeometryQualitySummary(
      checker: checker,
      regularity: regularity,
      edgeFrame: edgeFrame,
      combined: combined,
    );
  }

  BoardGeometry _refineGeometryForScore({
    required BoardGeometry geometry,
    required _DecodedLuma? decoded,
  }) {
    if (decoded == null || !geometry.isValid) {
      return geometry;
    }
    final width = decoded.width;
    final height = decoded.height;
    final current = List<BoardCorner>.from(geometry.corners);
    if (!_isReasonableQuad(current, width: width, height: height)) {
      return geometry;
    }

    double bestScore = _scoreGeometry(
      geometry: BoardGeometry(corners: current),
      decoded: decoded,
    );

    final directions = <List<int>>[
      <int>[-1, -1],
      <int>[-1, 0],
      <int>[-1, 1],
      <int>[0, -1],
      <int>[0, 1],
      <int>[1, -1],
      <int>[1, 0],
      <int>[1, 1],
    ];

    for (final step in <double>[40, 20, 10, 5, 2]) {
      int iter = 0;
      bool improved = true;
      while (improved && iter < 8) {
        improved = false;
        iter += 1;
        for (int i = 0; i < 4; i++) {
          final base = current[i];
          for (final dir in directions) {
            final nx = (base.x + (dir[0] * step)).clamp(
              0.0,
              (width - 1).toDouble(),
            );
            final ny = (base.y + (dir[1] * step)).clamp(
              0.0,
              (height - 1).toDouble(),
            );
            if ((nx - base.x).abs() < 1e-6 && (ny - base.y).abs() < 1e-6) {
              continue;
            }

            final candidate = List<BoardCorner>.from(current);
            candidate[i] = BoardCorner(x: nx, y: ny);
            if (!_isReasonableQuad(candidate, width: width, height: height)) {
              continue;
            }

            final score = _scoreGeometry(
              geometry: BoardGeometry(corners: candidate),
              decoded: decoded,
            );
            if (score > bestScore + 0.001) {
              bestScore = score;
              current
                ..clear()
                ..addAll(candidate);
              improved = true;
              break;
            }
          }
          if (improved) {
            break;
          }
        }
      }
    }

    return BoardGeometry(corners: current);
  }

  bool _isReasonableQuad(
    List<BoardCorner> c, {
    required int width,
    required int height,
  }) {
    if (c.length != 4) {
      return false;
    }
    for (final p in c) {
      if (p.x < 0 || p.y < 0 || p.x > width - 1 || p.y > height - 1) {
        return false;
      }
    }

    final area = _quadArea(c);
    final imageArea = math.max(1.0, (width * height).toDouble());
    final areaRatio = area / imageArea;
    if (areaRatio < 0.04 || areaRatio > 0.99) {
      return false;
    }

    if (!_isConvexQuad(c)) {
      return false;
    }

    final dTop = _distance(c[0], c[1]);
    final dRight = _distance(c[1], c[2]);
    final dBottom = _distance(c[2], c[3]);
    final dLeft = _distance(c[3], c[0]);
    final minSide = math.min(math.min(dTop, dRight), math.min(dBottom, dLeft));
    if (minSide < 16) {
      return false;
    }
    final avgHoriz = (dTop + dBottom) * 0.5;
    final avgVert = (dRight + dLeft) * 0.5;
    final aspect =
        math.max(avgHoriz, avgVert) /
        math.max(1e-6, math.min(avgHoriz, avgVert));
    if (aspect > 1.9) {
      return false;
    }
    return true;
  }

  bool _isConvexQuad(List<BoardCorner> c) {
    double? sign;
    for (int i = 0; i < 4; i++) {
      final a = c[i];
      final b = c[(i + 1) % 4];
      final d = c[(i + 2) % 4];
      final cross = (b.x - a.x) * (d.y - b.y) - (b.y - a.y) * (d.x - b.x);
      if (cross.abs() < 1e-6) {
        continue;
      }
      final s = cross > 0 ? 1.0 : -1.0;
      sign ??= s;
      if (sign != s) {
        return false;
      }
    }
    return true;
  }

  Future<_RefineOutcome> _refineCornersWithNativeOpenCv({
    required _DecodedLuma decoded,
    required BoardGeometry initialGeometry,
    required String imagePath,
    required Uint8List imageBytes,
    required String sourceLabel,
  }) async {
    if (!enableOpenCv || imageBytes.isEmpty) {
      return _RefineOutcome(
        geometry: initialGeometry,
        reason: 'fallback_native_disabled',
      );
    }

    final initialCorners = _orderCornersTlTrBrBl(initialGeometry.corners);
    if (!_isReasonableQuad(
      initialCorners,
      width: decoded.width,
      height: decoded.height,
    )) {
      return _RefineOutcome(
        geometry: initialGeometry,
        reason: 'fallback_native_invalid_initial_quad',
      );
    }

    try {
      final response = await Cv2.refineBoardCorners(
        pathFrom: _resolvePathFrom(imagePath),
        pathString: imagePath,
        data: imageBytes,
        corners: <double>[
          initialCorners[0].x,
          initialCorners[0].y,
          initialCorners[1].x,
          initialCorners[1].y,
          initialCorners[2].x,
          initialCorners[2].y,
          initialCorners[3].x,
          initialCorners[3].y,
        ],
        roiPaddingRatio: 0.30,
        minAreaRatio: 0.015,
        approxEpsilonRatio: 0.02,
        cannyLow: 35,
        cannyHigh: 130,
        subPixWinSize: 5,
        maxCandidates: 120,
      );
      if (response == null) {
        return _RefineOutcome(
          geometry: initialGeometry,
          reason: 'fallback_native_unavailable',
        );
      }
      final ok = response['ok'] == true;
      final reason = response['reason']?.toString() ?? 'native_no_reason';
      if (!ok) {
        return _RefineOutcome(
          geometry: initialGeometry,
          reason: 'fallback_native_not_ok($reason)',
        );
      }

      final parsedCorners = _parseNativeCorners(response['corners']);
      if (parsedCorners == null || parsedCorners.length != 4) {
        return _RefineOutcome(
          geometry: initialGeometry,
          reason: 'fallback_native_bad_payload($reason)',
        );
      }

      final ordered = _orderCornersTlTrBrBl(parsedCorners);
      if (!_isReasonableQuad(
        ordered,
        width: decoded.width,
        height: decoded.height,
      )) {
        return _RefineOutcome(
          geometry: initialGeometry,
          reason: 'fallback_native_invalid_quad($reason)',
        );
      }

      return _RefineOutcome(
        geometry: BoardGeometry(corners: ordered),
        reason: 'ok_native(source=$sourceLabel $reason)',
      );
    } catch (e) {
      return _RefineOutcome(
        geometry: initialGeometry,
        reason: 'fallback_native_exception($e)',
      );
    }
  }

  List<BoardCorner>? _parseNativeCorners(dynamic payload) {
    if (payload is! List || payload.length < 8) {
      return null;
    }
    final values = <double>[];
    for (final v in payload) {
      if (v is num) {
        values.add(v.toDouble());
      } else {
        return null;
      }
    }
    if (values.length < 8) {
      return null;
    }
    return <BoardCorner>[
      BoardCorner(x: values[0], y: values[1]),
      BoardCorner(x: values[2], y: values[3]),
      BoardCorner(x: values[4], y: values[5]),
      BoardCorner(x: values[6], y: values[7]),
    ];
  }

  _RefineOutcome _refineCornersInRoi({
    required _DecodedLuma decoded,
    required BoardGeometry initialGeometry,
    required double initialScore,
    required String imagePath,
    required String sourceLabel,
  }) {
    final initialCorners = _orderCornersTlTrBrBl(initialGeometry.corners);
    if (!_isReasonableQuad(
      initialCorners,
      width: decoded.width,
      height: decoded.height,
    )) {
      return _RefineOutcome(
        geometry: initialGeometry,
        reason: 'fallback_invalid_initial_quad',
      );
    }

    final roi = _buildRoiBounds(
      corners: initialCorners,
      width: decoded.width,
      height: decoded.height,
      padRatio: 0.22,
    );
    if (roi.width < 28 || roi.height < 28) {
      return _RefineOutcome(
        geometry: initialGeometry,
        reason: 'fallback_roi_too_small(${roi.width}x${roi.height})',
      );
    }

    final maps = _buildRoiBinaryMaps(decoded: decoded, roi: roi);
    _ScoredQuadCandidate best = _ScoredQuadCandidate(
      corners: initialCorners,
      score: initialScore,
      label: 'initial',
    );
    int candidatesTried = 0;
    int candidatesAccepted = 0;

    for (final map in maps) {
      final minArea = math.max(48, (roi.width * roi.height * 0.015).round());
      final components = _extractConnectedComponents(
        map: map,
        minArea: minArea,
      );
      for (final component in components) {
        final quadLocals = _quadCandidatesFromBoundary(component.boundary);
        for (
          int candidateIdx = 0;
          candidateIdx < quadLocals.length;
          candidateIdx++
        ) {
          final local = quadLocals[candidateIdx];
          candidatesTried += 1;
          final quadGlobal = local
              .map(
                (p) => BoardCorner(
                  x: (roi.left + p.x).toDouble(),
                  y: (roi.top + p.y).toDouble(),
                ),
              )
              .toList(growable: false);
          final ordered = _orderCornersTlTrBrBl(quadGlobal);
          if (!_isReasonableQuad(
            ordered,
            width: decoded.width,
            height: decoded.height,
          )) {
            continue;
          }

          final score = _scoreRoiCandidate(
            decoded: decoded,
            candidateCorners: ordered,
            initialCorners: initialCorners,
          );
          candidatesAccepted += 1;
          if (score > best.score + 0.001) {
            best = _ScoredQuadCandidate(
              corners: ordered,
              score: score,
              label: '${map.label}#${candidateIdx + 1}',
            );
          }
        }
      }
    }

    if (best.label == 'initial') {
      return _RefineOutcome(
        geometry: initialGeometry,
        reason:
            'fallback_no_better_quad(tried=$candidatesTried accepted=$candidatesAccepted)',
      );
    }

    final refinedSubPixel = _refineCornersSubPixelLike(
      decoded: decoded,
      corners: best.corners,
    );
    final subPixelGeometry = BoardGeometry(corners: refinedSubPixel);
    final subPixelScore = _scoreGeometry(
      geometry: subPixelGeometry,
      decoded: decoded,
    );
    if (subPixelScore + 0.0008 >= best.score) {
      return _RefineOutcome(
        geometry: subPixelGeometry,
        reason:
            'ok(source=$sourceLabel map=${best.label} score=${best.score.toStringAsFixed(4)} '
            'sub=${subPixelScore.toStringAsFixed(4)})',
      );
    }

    return _RefineOutcome(
      geometry: BoardGeometry(corners: best.corners),
      reason:
          'ok_no_subpixel_gain(source=$sourceLabel map=${best.label} '
          'score=${best.score.toStringAsFixed(4)} sub=${subPixelScore.toStringAsFixed(4)})',
    );
  }

  BoardGeometry _refineInnerBoardWindow({
    required _DecodedLuma decoded,
    required BoardGeometry geometry,
  }) {
    if (!geometry.isValid) {
      return geometry;
    }
    final baseCorners = _orderCornersTlTrBrBl(geometry.corners);
    if (!_isReasonableQuad(
      baseCorners,
      width: decoded.width,
      height: decoded.height,
    )) {
      return geometry;
    }

    final warpSize = checkerTargetSize.clamp(128, 256);
    final warped = _warpBoardLuma(
      decoded: decoded,
      corners: baseCorners,
      targetSize: warpSize,
    );
    final homography = _homographyUnitToQuad(baseCorners);
    if (warped == null || homography == null) {
      return geometry;
    }

    final fullScore = _windowSelectionScore(
      warped: warped,
      u0: 0.0,
      v0: 0.0,
      u1: 1.0,
      v1: 1.0,
    );
    var bestScore = fullScore;
    double bestU0 = 0.0;
    double bestV0 = 0.0;
    double bestU1 = 1.0;
    double bestV1 = 1.0;

    const sizes = <double>[
      0.50,
      0.55,
      0.60,
      0.65,
      0.70,
      0.75,
      0.80,
      0.85,
      0.90,
      0.95,
      1.0,
    ];
    for (final size in sizes) {
      final maxStart = (1.0 - size).clamp(0.0, 1.0);
      final steps = math.max(1, (maxStart / 0.05).floor());
      for (int xi = 0; xi <= steps; xi++) {
        final u0 = xi == steps ? maxStart : (xi * 0.05);
        final u1 = (u0 + size).clamp(0.0, 1.0);
        for (int yi = 0; yi <= steps; yi++) {
          final v0 = yi == steps ? maxStart : (yi * 0.05);
          final v1 = (v0 + size).clamp(0.0, 1.0);
          final score = _windowSelectionScore(
            warped: warped,
            u0: u0,
            v0: v0,
            u1: u1,
            v1: v1,
          );
          if (score > bestScore + 0.030) {
            bestScore = score;
            bestU0 = u0;
            bestV0 = v0;
            bestU1 = u1;
            bestV1 = v1;
          }
        }
      }
    }

    if ((bestU0 - 0.0).abs() < 1e-6 &&
        (bestV0 - 0.0).abs() < 1e-6 &&
        (bestU1 - 1.0).abs() < 1e-6 &&
        (bestV1 - 1.0).abs() < 1e-6) {
      return geometry;
    }

    final p0 = _applyHomography(homography, bestU0, bestV0);
    final p1 = _applyHomography(homography, bestU1, bestV0);
    final p2 = _applyHomography(homography, bestU1, bestV1);
    final p3 = _applyHomography(homography, bestU0, bestV1);
    final refined = <BoardCorner>[
      BoardCorner(x: p0.x, y: p0.y),
      BoardCorner(x: p1.x, y: p1.y),
      BoardCorner(x: p2.x, y: p2.y),
      BoardCorner(x: p3.x, y: p3.y),
    ];
    final ordered = _orderCornersTlTrBrBl(refined);
    if (!_isReasonableQuad(
      ordered,
      width: decoded.width,
      height: decoded.height,
    )) {
      return geometry;
    }

    final currentScore = _scoreGeometry(geometry: geometry, decoded: decoded);
    final refinedScore = _scoreGeometry(
      geometry: BoardGeometry(corners: ordered),
      decoded: decoded,
    );
    if (refinedScore <= currentScore + 0.010) {
      return geometry;
    }
    if (kDebugMode) {
      debugPrint(
        '[scan][hybrid] inner_window=(${bestU0.toStringAsFixed(2)},${bestV0.toStringAsFixed(2)} -> '
        '${bestU1.toStringAsFixed(2)},${bestV1.toStringAsFixed(2)}) '
        'full=${fullScore.toStringAsFixed(4)} best=${bestScore.toStringAsFixed(4)}',
      );
    }
    return BoardGeometry(corners: ordered);
  }

  BoardGeometry _forceInnerBoardWindowFromProfiles({
    required _DecodedLuma decoded,
    required BoardGeometry geometry,
  }) {
    if (!geometry.isValid) {
      return geometry;
    }
    final baseCorners = _orderCornersTlTrBrBl(geometry.corners);
    final warpSize = checkerTargetSize.clamp(128, 256);
    final warped = _warpBoardLuma(
      decoded: decoded,
      corners: baseCorners,
      targetSize: warpSize,
    );
    final homography = _homographyUnitToQuad(baseCorners);
    if (warped == null || homography == null) {
      return geometry;
    }

    final cols = List<double>.filled(warped.size, 0.0);
    final rows = List<double>.filled(warped.size, 0.0);
    for (int y = 1; y < warped.size - 1; y++) {
      final rowOffset = y * warped.size;
      for (int x = 1; x < warped.size - 1; x++) {
        final idx = rowOffset + x;
        final gx = (warped.luma[idx + 1] - warped.luma[idx - 1]).abs();
        final gy =
            (warped.luma[idx + warped.size] - warped.luma[idx - warped.size])
                .abs();
        final g = gx + gy;
        cols[x] += g;
        rows[y] += g;
      }
    }
    final smoothCols = _smoothProjection(cols, radius: 4);
    final smoothRows = _smoothProjection(rows, radius: 4);
    final n = warped.size;

    final left = _peakIndexInRange(
      smoothCols,
      (n * 0.08).round(),
      (n * 0.60).round(),
    );
    final right = _peakIndexInRange(
      smoothCols,
      (n * 0.40).round(),
      (n * 0.96).round(),
    );
    final top = _peakIndexInRange(
      smoothRows,
      (n * 0.08).round(),
      (n * 0.60).round(),
    );
    final bottom = _peakIndexInRange(
      smoothRows,
      (n * 0.40).round(),
      (n * 0.96).round(),
    );

    if (left < 0 || right < 0 || top < 0 || bottom < 0) {
      return geometry;
    }
    if (right - left < (n * 0.28) || bottom - top < (n * 0.28)) {
      return geometry;
    }

    final p0 = _applyHomography(homography, left / n, top / n);
    final p1 = _applyHomography(homography, right / n, top / n);
    final p2 = _applyHomography(homography, right / n, bottom / n);
    final p3 = _applyHomography(homography, left / n, bottom / n);
    final candidate = _orderCornersTlTrBrBl(<BoardCorner>[
      BoardCorner(x: p0.x, y: p0.y),
      BoardCorner(x: p1.x, y: p1.y),
      BoardCorner(x: p2.x, y: p2.y),
      BoardCorner(x: p3.x, y: p3.y),
    ]);
    if (!_isReasonableQuad(
      candidate,
      width: decoded.width,
      height: decoded.height,
    )) {
      return geometry;
    }
    return BoardGeometry(corners: candidate);
  }

  List<double> _smoothProjection(List<double> input, {required int radius}) {
    final out = List<double>.filled(input.length, 0.0);
    for (int i = 0; i < input.length; i++) {
      double sum = 0;
      int count = 0;
      final a = math.max(0, i - radius);
      final b = math.min(input.length - 1, i + radius);
      for (int j = a; j <= b; j++) {
        sum += input[j];
        count += 1;
      }
      out[i] = sum / math.max(1, count);
    }
    return out;
  }

  int _peakIndexInRange(List<double> values, int start, int end) {
    if (values.isEmpty) {
      return -1;
    }
    final s = start.clamp(0, values.length - 1);
    final e = end.clamp(s, values.length - 1);
    int best = -1;
    double bestVal = -double.infinity;
    for (int i = s; i <= e; i++) {
      final v = values[i];
      if (v > bestVal) {
        bestVal = v;
        best = i;
      }
    }
    return best;
  }

  double _windowSelectionScore({
    required _WarpedBoard warped,
    required double u0,
    required double v0,
    required double u1,
    required double v1,
  }) {
    final checker = _checkerboardWindowScore(
      warped: warped,
      u0: u0,
      v0: v0,
      u1: u1,
      v1: v1,
    );
    final regularity = _gridRegularityWindowScore(
      warped: warped,
      u0: u0,
      v0: v0,
      u1: u1,
      v1: v1,
    );
    final edgeFrame = _edgeFrameWindowScore(
      warped: warped,
      u0: u0,
      v0: v0,
      u1: u1,
      v1: v1,
    );
    return ((checker * 0.45) + (regularity * 0.40) + (edgeFrame * 0.15))
        .clamp(0.0, 1.0)
        .toDouble();
  }

  double _gridRegularityWindowScore({
    required _WarpedBoard warped,
    required double u0,
    required double v0,
    required double u1,
    required double v1,
  }) {
    final left = (u0 * warped.size).floor().clamp(0, warped.size - 1);
    final top = (v0 * warped.size).floor().clamp(0, warped.size - 1);
    final right = (u1 * warped.size).ceil().clamp(left + 1, warped.size);
    final bottom = (v1 * warped.size).ceil().clamp(top + 1, warped.size);
    final width = right - left;
    final height = bottom - top;
    if (width < 28 || height < 28) {
      return 0.0;
    }

    final col = List<double>.filled(width, 0.0);
    final row = List<double>.filled(height, 0.0);
    for (int y = top + 1; y < bottom - 1; y++) {
      final rowOffset = y * warped.size;
      for (int x = left + 1; x < right - 1; x++) {
        final idx = rowOffset + x;
        final gx = (warped.luma[idx + 1] - warped.luma[idx - 1]).abs();
        final gy =
            (warped.luma[idx + warped.size] - warped.luma[idx - warped.size])
                .abs();
        final g = gx + gy;
        col[x - left] += g;
        row[y - top] += g;
      }
    }
    final colScore = _projectionRegularityScore(col);
    final rowScore = _projectionRegularityScore(row);
    return math.sqrt(colScore * rowScore).clamp(0.0, 1.0).toDouble();
  }

  double _projectionRegularityScore(List<double> projection) {
    if (projection.length < 8) {
      return 0.0;
    }
    final smooth = List<double>.filled(projection.length, 0.0);
    for (int i = 0; i < projection.length; i++) {
      double sum = 0;
      int count = 0;
      final a = math.max(0, i - 2);
      final b = math.min(projection.length - 1, i + 2);
      for (int j = a; j <= b; j++) {
        sum += projection[j];
        count += 1;
      }
      smooth[i] = sum / math.max(1, count);
    }

    final mean = smooth.fold<double>(0.0, (acc, v) => acc + v) / smooth.length;
    double varSum = 0;
    for (final v in smooth) {
      final d = v - mean;
      varSum += d * d;
    }
    final std = math.sqrt(varSum / smooth.length);
    final threshold = mean + std * 0.65;

    final peaks = <int>[];
    for (int i = 1; i < smooth.length - 1; i++) {
      if (smooth[i] >= threshold &&
          smooth[i] >= smooth[i - 1] &&
          smooth[i] > smooth[i + 1]) {
        peaks.add(i);
      }
    }
    if (peaks.length < 4) {
      return 0.0;
    }

    final countScore = (1.0 - ((peaks.length - 9).abs() / 9.0))
        .clamp(0.0, 1.0)
        .toDouble();
    final deltas = <double>[];
    for (int i = 1; i < peaks.length; i++) {
      deltas.add((peaks[i] - peaks[i - 1]).toDouble());
    }
    if (deltas.isEmpty) {
      return countScore * 0.5;
    }
    final meanDelta =
        deltas.fold<double>(0.0, (acc, v) => acc + v) / deltas.length;
    double varDelta = 0;
    for (final d in deltas) {
      final dv = d - meanDelta;
      varDelta += dv * dv;
    }
    final stdDelta = math.sqrt(varDelta / deltas.length);
    final cv = stdDelta / math.max(1e-6, meanDelta);
    final periodicity = (1.0 - (cv / 0.60)).clamp(0.0, 1.0).toDouble();
    return ((countScore * 0.45) + (periodicity * 0.55))
        .clamp(0.0, 1.0)
        .toDouble();
  }

  double _edgeFrameWindowScore({
    required _WarpedBoard warped,
    required double u0,
    required double v0,
    required double u1,
    required double v1,
  }) {
    final left = (u0 * warped.size).floor().clamp(0, warped.size - 1);
    final top = (v0 * warped.size).floor().clamp(0, warped.size - 1);
    final right = (u1 * warped.size).ceil().clamp(left + 1, warped.size);
    final bottom = (v1 * warped.size).ceil().clamp(top + 1, warped.size);
    final width = right - left;
    final height = bottom - top;
    if (width < 24 || height < 24) {
      return 0.0;
    }
    final border = math.max(2, (math.min(width, height) * 0.06).round());
    double borderSum = 0;
    int borderCount = 0;
    double innerSum = 0;
    int innerCount = 0;
    for (int y = top + 1; y < bottom - 1; y++) {
      final rowOffset = y * warped.size;
      for (int x = left + 1; x < right - 1; x++) {
        final idx = rowOffset + x;
        final gx = (warped.luma[idx + 1] - warped.luma[idx - 1]).abs();
        final gy =
            (warped.luma[idx + warped.size] - warped.luma[idx - warped.size])
                .abs();
        final g = gx + gy;
        final isBorder =
            x < left + border ||
            x >= right - border ||
            y < top + border ||
            y >= bottom - border;
        if (isBorder) {
          borderSum += g;
          borderCount += 1;
        } else {
          innerSum += g;
          innerCount += 1;
        }
      }
    }
    if (borderCount == 0 || innerCount == 0) {
      return 0.0;
    }
    final ratio =
        (borderSum / borderCount) / math.max(1e-6, innerSum / innerCount);
    return (ratio / 1.8).clamp(0.0, 1.0).toDouble();
  }

  _RoiBounds _buildRoiBounds({
    required List<BoardCorner> corners,
    required int width,
    required int height,
    required double padRatio,
  }) {
    final minX = corners.map((c) => c.x).reduce(math.min);
    final maxX = corners.map((c) => c.x).reduce(math.max);
    final minY = corners.map((c) => c.y).reduce(math.min);
    final maxY = corners.map((c) => c.y).reduce(math.max);
    final w = math.max(1.0, maxX - minX);
    final h = math.max(1.0, maxY - minY);
    final pad = math.max(6.0, math.max(w, h) * padRatio);
    final left = (minX - pad).floor().clamp(0, width - 1).toInt();
    final top = (minY - pad).floor().clamp(0, height - 1).toInt();
    final right = (maxX + pad).ceil().clamp(0, width - 1).toInt();
    final bottom = (maxY + pad).ceil().clamp(0, height - 1).toInt();
    return _RoiBounds(left: left, top: top, right: right, bottom: bottom);
  }

  List<_BinaryRoiMap> _buildRoiBinaryMaps({
    required _DecodedLuma decoded,
    required _RoiBounds roi,
  }) {
    final w = roi.width;
    final h = roi.height;
    final size = w * h;
    if (size <= 0) {
      return const <_BinaryRoiMap>[];
    }

    final roiLuma = Uint8List(size);
    final grad = Float32List(size);
    double sumL = 0;
    double sumG = 0;
    for (int y = 0; y < h; y++) {
      final gy = roi.top + y;
      final rowOut = y * w;
      for (int x = 0; x < w; x++) {
        final gx = roi.left + x;
        final idxGlobal = (gy * decoded.width) + gx;
        final l = decoded.luma[idxGlobal].toDouble();
        roiLuma[rowOut + x] = l.round().clamp(0, 255).toInt();
        sumL += l;

        final gxL = math.max(0, gx - 1).toInt();
        final gxR = math.min(decoded.width - 1, gx + 1).toInt();
        final gyT = math.max(0, gy - 1).toInt();
        final gyB = math.min(decoded.height - 1, gy + 1).toInt();
        final leftV = decoded.luma[(gy * decoded.width) + gxL].toDouble();
        final rightV = decoded.luma[(gy * decoded.width) + gxR].toDouble();
        final topV = decoded.luma[(gyT * decoded.width) + gx].toDouble();
        final botV = decoded.luma[(gyB * decoded.width) + gx].toDouble();
        final g = (rightV - leftV).abs() + (botV - topV).abs();
        grad[rowOut + x] = g.toDouble();
        sumG += g;
      }
    }

    final meanL = sumL / size;
    final meanG = sumG / size;
    double varL = 0;
    double varG = 0;
    for (int i = 0; i < size; i++) {
      final dl = roiLuma[i] - meanL;
      final dg = grad[i] - meanG;
      varL += dl * dl;
      varG += dg * dg;
    }
    final stdL = math.sqrt(varL / size);
    final stdG = math.sqrt(varG / size);

    final strongEdgeThreshold = meanG + stdG * 0.85;
    final midEdgeThreshold = meanG + stdG * 0.55;

    final strongEdge = Uint8List(size);
    final midEdge = Uint8List(size);
    for (int i = 0; i < size; i++) {
      if (grad[i] >= strongEdgeThreshold) {
        strongEdge[i] = 1;
      }
      if (grad[i] >= midEdgeThreshold) {
        midEdge[i] = 1;
      }
    }

    final localContrast = Uint8List(size);
    final integral = _buildIntegralFromRoiLuma(roiLuma, w, h);
    final radius = math.max(2, (math.min(w, h) * 0.08).round()).toInt();
    final contrastThreshold = math.max(10.0, stdL * 0.33);
    for (int y = 0; y < h; y++) {
      final rowOffset = y * w;
      final y0 = math.max(0, y - radius).toInt();
      final y1 = math.min(h - 1, y + radius).toInt();
      for (int x = 0; x < w; x++) {
        final x0 = math.max(0, x - radius).toInt();
        final x1 = math.min(w - 1, x + radius).toInt();
        final localMean = _rectMeanIntegral(
          integral: integral,
          width: w,
          left: x0,
          top: y0,
          right: x1,
          bottom: y1,
        );
        final l = roiLuma[rowOffset + x].toDouble();
        if ((l - localMean).abs() >= contrastThreshold) {
          localContrast[rowOffset + x] = 1;
        }
      }
    }

    final maps = <_BinaryRoiMap>[
      _BinaryRoiMap(label: 'edge_strong', width: w, height: h, map: strongEdge),
      _BinaryRoiMap(label: 'edge_mid', width: w, height: h, map: midEdge),
      _BinaryRoiMap(
        label: 'local_contrast',
        width: w,
        height: h,
        map: localContrast,
      ),
    ];
    return maps;
  }

  Float64List _buildIntegralFromRoiLuma(Uint8List luma, int width, int height) {
    final stride = width + 1;
    final integral = Float64List((width + 1) * (height + 1));
    for (int y = 1; y <= height; y++) {
      double rowSum = 0;
      final srcRow = (y - 1) * width;
      final rowIndex = y * stride;
      final prevRow = (y - 1) * stride;
      for (int x = 1; x <= width; x++) {
        rowSum += luma[srcRow + x - 1];
        integral[rowIndex + x] = integral[prevRow + x] + rowSum;
      }
    }
    return integral;
  }

  double _rectMeanIntegral({
    required Float64List integral,
    required int width,
    required int left,
    required int top,
    required int right,
    required int bottom,
  }) {
    if (right < left || bottom < top) {
      return 0.0;
    }
    final stride = width + 1;
    final a = integral[top * stride + left];
    final b = integral[top * stride + (right + 1)];
    final c = integral[(bottom + 1) * stride + left];
    final d = integral[(bottom + 1) * stride + (right + 1)];
    final area = ((right - left + 1) * (bottom - top + 1)).toDouble();
    if (area <= 0) {
      return 0.0;
    }
    return (d - b - c + a) / area;
  }

  List<_RoiComponent> _extractConnectedComponents({
    required _BinaryRoiMap map,
    required int minArea,
  }) {
    final visited = Uint8List(map.width * map.height);
    final components = <_RoiComponent>[];
    const neighbors = <List<int>>[
      <int>[1, 0],
      <int>[-1, 0],
      <int>[0, 1],
      <int>[0, -1],
      <int>[1, 1],
      <int>[1, -1],
      <int>[-1, 1],
      <int>[-1, -1],
    ];

    for (int y = 0; y < map.height; y++) {
      final rowOffset = y * map.width;
      for (int x = 0; x < map.width; x++) {
        final startIdx = rowOffset + x;
        if (visited[startIdx] == 1 || map.map[startIdx] == 0) {
          continue;
        }

        final queue = <int>[startIdx];
        visited[startIdx] = 1;
        int head = 0;
        int area = 0;
        final boundary = <_PointI>[];

        while (head < queue.length) {
          final idx = queue[head++];
          final cx = idx % map.width;
          final cy = idx ~/ map.width;
          area += 1;
          bool isBoundary = false;

          for (final n in neighbors) {
            final nx = cx + n[0];
            final ny = cy + n[1];
            if (nx < 0 || ny < 0 || nx >= map.width || ny >= map.height) {
              isBoundary = true;
              continue;
            }
            final nIdx = ny * map.width + nx;
            if (map.map[nIdx] == 0) {
              isBoundary = true;
              continue;
            }
            if (visited[nIdx] == 0) {
              visited[nIdx] = 1;
              queue.add(nIdx);
            }
          }

          if (isBoundary) {
            boundary.add(_PointI(cx, cy));
          }
        }

        if (area < minArea || boundary.length < 20) {
          continue;
        }
        components.add(_RoiComponent(area: area, boundary: boundary));
      }
    }
    components.sort((a, b) => b.area.compareTo(a.area));
    return components.take(12).toList(growable: false);
  }

  List<_PointI>? _quadFromBoundary(List<_PointI> boundary) {
    if (boundary.length < 20) {
      return null;
    }

    _PointI? tl;
    _PointI? tr;
    _PointI? br;
    _PointI? bl;
    int bestTl = -1 << 30;
    int bestTr = -1 << 30;
    int bestBr = -1 << 30;
    int bestBl = -1 << 30;

    for (final p in boundary) {
      final sum = p.x + p.y;
      final diff = p.x - p.y;
      final invDiff = p.y - p.x;
      final negSum = -(p.x + p.y);
      if (negSum > bestTl) {
        bestTl = negSum;
        tl = p;
      }
      if (diff > bestTr) {
        bestTr = diff;
        tr = p;
      }
      if (sum > bestBr) {
        bestBr = sum;
        br = p;
      }
      if (invDiff > bestBl) {
        bestBl = invDiff;
        bl = p;
      }
    }

    if (tl == null || tr == null || br == null || bl == null) {
      return null;
    }

    final radius = math.max(4.0, math.sqrt(boundary.length) * 0.9);
    final rtl = _refinePointFromBoundary(tl, boundary, radius);
    final rtr = _refinePointFromBoundary(tr, boundary, radius);
    final rbr = _refinePointFromBoundary(br, boundary, radius);
    final rbl = _refinePointFromBoundary(bl, boundary, radius);

    return <_PointI>[
      _PointI(rtl.x.round(), rtl.y.round()),
      _PointI(rtr.x.round(), rtr.y.round()),
      _PointI(rbr.x.round(), rbr.y.round()),
      _PointI(rbl.x.round(), rbl.y.round()),
    ];
  }

  List<List<_PointI>> _quadCandidatesFromBoundary(List<_PointI> boundary) {
    final out = <List<_PointI>>[];
    final axis = _quadFromBoundary(boundary);
    if (axis != null) {
      out.add(axis);
    }
    final oriented = _orientedQuadFromBoundary(boundary);
    if (oriented != null) {
      out.add(oriented);
    }
    return out;
  }

  List<_PointI>? _orientedQuadFromBoundary(List<_PointI> boundary) {
    if (boundary.length < 20) {
      return null;
    }
    double cx = 0;
    double cy = 0;
    for (final p in boundary) {
      cx += p.x;
      cy += p.y;
    }
    cx /= boundary.length;
    cy /= boundary.length;

    double xx = 0;
    double xy = 0;
    double yy = 0;
    int minX = boundary.first.x;
    int minY = boundary.first.y;
    int maxX = boundary.first.x;
    int maxY = boundary.first.y;
    for (final p in boundary) {
      final dx = p.x - cx;
      final dy = p.y - cy;
      xx += dx * dx;
      xy += dx * dy;
      yy += dy * dy;
      if (p.x < minX) {
        minX = p.x;
      }
      if (p.y < minY) {
        minY = p.y;
      }
      if (p.x > maxX) {
        maxX = p.x;
      }
      if (p.y > maxY) {
        maxY = p.y;
      }
    }
    final denom = (xx - yy).abs() + (2 * xy).abs();
    if (denom < 1e-6) {
      return null;
    }

    final theta = 0.5 * math.atan2(2 * xy, xx - yy);
    final cosT = math.cos(theta);
    final sinT = math.sin(theta);
    double minU = double.infinity;
    double maxU = double.negativeInfinity;
    double minV = double.infinity;
    double maxV = double.negativeInfinity;
    for (final p in boundary) {
      final dx = p.x - cx;
      final dy = p.y - cy;
      final u = (dx * cosT) + (dy * sinT);
      final v = (-dx * sinT) + (dy * cosT);
      if (u < minU) {
        minU = u;
      }
      if (u > maxU) {
        maxU = u;
      }
      if (v < minV) {
        minV = v;
      }
      if (v > maxV) {
        maxV = v;
      }
    }

    final sideU = (maxU - minU).abs();
    final sideV = (maxV - minV).abs();
    if (math.min(sideU, sideV) < 8) {
      return null;
    }

    _PointD toGlobal(double u, double v) {
      final x = cx + (u * cosT) - (v * sinT);
      final y = cy + (u * sinT) + (v * cosT);
      return _PointD(x, y);
    }

    final seeds = <_PointD>[
      toGlobal(minU, minV),
      toGlobal(maxU, minV),
      toGlobal(maxU, maxV),
      toGlobal(minU, maxV),
    ];
    final radius = math.max(4.0, math.sqrt(boundary.length) * 0.9);
    final out = <_PointI>[];
    for (final s in seeds) {
      final seed = _PointI(
        s.x.round().clamp(minX, maxX),
        s.y.round().clamp(minY, maxY),
      );
      final refined = _refinePointFromBoundary(seed, boundary, radius);
      out.add(
        _PointI(
          refined.x.round().clamp(minX, maxX),
          refined.y.round().clamp(minY, maxY),
        ),
      );
    }
    return out;
  }

  _PointD _refinePointFromBoundary(
    _PointI seed,
    List<_PointI> boundary,
    double radius,
  ) {
    double cx = seed.x.toDouble();
    double cy = seed.y.toDouble();
    double r = radius;
    for (int iter = 0; iter < 2; iter++) {
      final r2 = r * r;
      double sumW = 0;
      double sumX = 0;
      double sumY = 0;
      for (final p in boundary) {
        final dx = p.x - cx;
        final dy = p.y - cy;
        final d2 = dx * dx + dy * dy;
        if (d2 > r2) {
          continue;
        }
        final w = 1.0 / (1.0 + d2 * 0.05);
        sumW += w;
        sumX += p.x * w;
        sumY += p.y * w;
      }
      if (sumW > 1e-6) {
        cx = sumX / sumW;
        cy = sumY / sumW;
      }
      r *= 0.65;
    }
    return _PointD(cx, cy);
  }

  double _scoreRoiCandidate({
    required _DecodedLuma decoded,
    required List<BoardCorner> candidateCorners,
    required List<BoardCorner> initialCorners,
  }) {
    final geomScore = _scoreGeometry(
      geometry: BoardGeometry(corners: candidateCorners),
      decoded: decoded,
    );

    final initArea = math.max(1e-6, _quadArea(initialCorners));
    final candArea = math.max(1e-6, _quadArea(candidateCorners));
    final areaRatio = candArea / initArea;
    final areaScore = (1.0 - (math.log(areaRatio).abs() / 1.4))
        .clamp(0.0, 1.0)
        .toDouble();

    final initSideRef = _averageSideLength(initialCorners);
    double dist = 0;
    for (int i = 0; i < 4; i++) {
      dist += _distance(initialCorners[i], candidateCorners[i]);
    }
    dist /= 4.0;
    final proximity = (1.0 - (dist / math.max(1e-6, initSideRef * 0.60)))
        .clamp(0.0, 1.0)
        .toDouble();

    return (geomScore * 0.78) + (proximity * 0.14) + (areaScore * 0.08);
  }

  double _averageSideLength(List<BoardCorner> corners) {
    if (corners.length != 4) {
      return 1.0;
    }
    final a = _distance(corners[0], corners[1]);
    final b = _distance(corners[1], corners[2]);
    final c = _distance(corners[2], corners[3]);
    final d = _distance(corners[3], corners[0]);
    return (a + b + c + d) * 0.25;
  }

  List<BoardCorner> _refineCornersSubPixelLike({
    required _DecodedLuma decoded,
    required List<BoardCorner> corners,
  }) {
    final refined = <BoardCorner>[];
    final baseRadius = math.max(4.0, _averageSideLength(corners) * 0.04);
    for (final c in corners) {
      double cx = c.x;
      double cy = c.y;
      double radius = baseRadius;
      for (int iter = 0; iter < 2; iter++) {
        final r = radius.round().clamp(2, 24).toInt();
        final x0 = (cx - r).floor().clamp(1, decoded.width - 2).toInt();
        final x1 = (cx + r).ceil().clamp(1, decoded.width - 2).toInt();
        final y0 = (cy - r).floor().clamp(1, decoded.height - 2).toInt();
        final y1 = (cy + r).ceil().clamp(1, decoded.height - 2).toInt();
        double sumW = 0;
        double sumX = 0;
        double sumY = 0;
        for (int y = y0; y <= y1; y++) {
          final rowOffset = y * decoded.width;
          for (int x = x0; x <= x1; x++) {
            final idx = rowOffset + x;
            final dx =
                decoded.luma[idx + 1].toDouble() -
                decoded.luma[idx - 1].toDouble();
            final dy =
                decoded.luma[idx + decoded.width].toDouble() -
                decoded.luma[idx - decoded.width].toDouble();
            final g = dx.abs() + dy.abs();
            if (g < 6) {
              continue;
            }
            final ddx = x - cx;
            final ddy = y - cy;
            final dist2 = ddx * ddx + ddy * ddy;
            final w = g / (1.0 + dist2 * 0.045);
            sumW += w;
            sumX += x * w;
            sumY += y * w;
          }
        }
        if (sumW > 1e-6) {
          cx = sumX / sumW;
          cy = sumY / sumW;
        }
        radius *= 0.60;
      }
      refined.add(
        BoardCorner(
          x: cx.clamp(0.0, (decoded.width - 1).toDouble()).toDouble(),
          y: cy.clamp(0.0, (decoded.height - 1).toDouble()).toDouble(),
        ),
      );
    }
    return _orderCornersTlTrBrBl(refined);
  }

  List<BoardCorner> _orderCornersTlTrBrBl(List<BoardCorner> corners) {
    if (corners.length != 4) {
      return corners;
    }
    int idxTl = 0;
    int idxBr = 0;
    int idxTr = 0;
    int idxBl = 0;
    for (int i = 1; i < 4; i++) {
      final si = corners[i].x + corners[i].y;
      final stl = corners[idxTl].x + corners[idxTl].y;
      final sbr = corners[idxBr].x + corners[idxBr].y;
      if (si < stl) {
        idxTl = i;
      }
      if (si > sbr) {
        idxBr = i;
      }
      final di = corners[i].x - corners[i].y;
      final dtr = corners[idxTr].x - corners[idxTr].y;
      final dbl = corners[idxBl].x - corners[idxBl].y;
      if (di > dtr) {
        idxTr = i;
      }
      if (di < dbl) {
        idxBl = i;
      }
    }
    final unique = <int>{idxTl, idxTr, idxBr, idxBl};
    if (unique.length == 4) {
      return <BoardCorner>[
        corners[idxTl],
        corners[idxTr],
        corners[idxBr],
        corners[idxBl],
      ];
    }
    return corners;
  }

  String _cornersLabel(List<BoardCorner> corners) {
    return corners
        .map((c) => '(${c.x.toStringAsFixed(1)},${c.y.toStringAsFixed(1)})')
        .join(' ');
  }

  CVPathFrom _resolvePathFrom(String path) {
    final normalized = path.toLowerCase();
    if (normalized.startsWith('assets/')) {
      return CVPathFrom.ASSETS;
    }
    if (normalized.startsWith('http://') || normalized.startsWith('https://')) {
      return CVPathFrom.URL;
    }
    return CVPathFrom.GALLERY_CAMERA;
  }

  Future<List<_OpenCvVariant>> _buildOpenCvVariants({
    required CVPathFrom pathFrom,
    required String path,
  }) async {
    final out = <_OpenCvVariant>[];

    Future<void> add(String label, Future<dynamic> Function() run) async {
      try {
        final value = await run();
        final bytes = _asBytes(value);
        if (bytes != null && bytes.isNotEmpty) {
          out.add(_OpenCvVariant(label: label, bytes: bytes));
        }
      } catch (_) {
        // Ignore unavailable operation / plugin failures.
      }
    }

    await add(
      'opencv_gray',
      () => Cv2.cvtColor(
        pathFrom: pathFrom,
        pathString: path,
        outputType: Cv2.COLOR_BGR2GRAY,
      ),
    );

    await add(
      'opencv_blur',
      () => Cv2.gaussianBlur(
        pathFrom: pathFrom,
        pathString: path,
        kernelSize: <double>[5, 5],
        sigmaX: 0,
      ),
    );

    await add(
      'opencv_threshold_otsu',
      () => Cv2.threshold(
        pathFrom: pathFrom,
        pathString: path,
        thresholdValue: 0,
        maxThresholdValue: 255,
        thresholdType: Cv2.THRESH_BINARY | Cv2.THRESH_OTSU,
      ),
    );

    if (!useLightPreprocessSet) {
      await add(
        'opencv_adaptive_threshold',
        () => Cv2.adaptiveThreshold(
          pathFrom: pathFrom,
          pathString: path,
          maxValue: 255,
          adaptiveMethod: Cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
          thresholdType: Cv2.THRESH_BINARY,
          blockSize: 17,
          constantValue: 4,
        ),
      );

      await add(
        'opencv_laplacian',
        () => Cv2.laplacian(pathFrom: pathFrom, pathString: path, depth: -1),
      );

      await add(
        'opencv_sobel_xy',
        () => Cv2.sobel(
          pathFrom: pathFrom,
          pathString: path,
          depth: -1,
          dx: 1,
          dy: 1,
        ),
      );
    }

    final limited = maxOpenCvVariants <= 0
        ? out
        : out.take(maxOpenCvVariants).toList(growable: false);
    return limited;
  }

  Uint8List? _asBytes(dynamic value) {
    if (value is Uint8List) {
      return value;
    }
    if (value is ByteData) {
      return value.buffer.asUint8List();
    }
    return null;
  }

  Future<_DecodedLuma?> _decodeLuma(Uint8List bytes) async {
    try {
      final codec = await ui.instantiateImageCodec(bytes);
      final frame = await codec.getNextFrame();
      final image = frame.image;
      final byteData = await image.toByteData(
        format: ui.ImageByteFormat.rawRgba,
      );
      if (byteData == null) {
        return null;
      }
      final rgba = byteData.buffer.asUint8List();
      final pixelCount = rgba.length ~/ 4;
      final luma = Uint8List(pixelCount);
      int src = 0;
      for (int i = 0; i < pixelCount; i++) {
        final r = rgba[src];
        final g = rgba[src + 1];
        final b = rgba[src + 2];
        luma[i] = ((r * 77) + (g * 150) + (b * 29)) >> 8;
        src += 4;
      }
      return _DecodedLuma(width: image.width, height: image.height, luma: luma);
    } catch (_) {
      return null;
    }
  }

  double _scoreGeometry({
    required BoardGeometry geometry,
    required _DecodedLuma? decoded,
  }) {
    final corners = geometry.corners;
    if (corners.length != 4) {
      return 0.0;
    }
    if (decoded == null) {
      return 0.5;
    }

    final tl = corners[0];
    final tr = corners[1];
    final br = corners[2];
    final bl = corners[3];

    final width = decoded.width.toDouble();
    final height = decoded.height.toDouble();
    final imageArea = math.max(1.0, width * height);

    final area = _quadArea(corners);
    final areaRatio = area / imageArea;
    double areaScore = 1.0;
    if (areaRatio < 0.10) {
      areaScore = 0.1;
    } else if (areaRatio < 0.20) {
      areaScore = 0.45;
    } else if (areaRatio < 0.30) {
      areaScore = 0.75;
    } else if (areaRatio > 0.98) {
      areaScore = 0.35;
    } else if (areaRatio > 0.92) {
      areaScore = 0.70;
    }

    final sideTop = _distance(tl, tr);
    final sideRight = _distance(tr, br);
    final sideBottom = _distance(br, bl);
    final sideLeft = _distance(bl, tl);
    final avgHoriz = (sideTop + sideBottom) * 0.5;
    final avgVert = (sideRight + sideLeft) * 0.5;
    final aspect =
        math.max(avgHoriz, avgVert) /
        math.max(1e-6, math.min(avgHoriz, avgVert));
    final aspectScore = (1.0 - ((aspect - 1.0) / 0.8))
        .clamp(0.0, 1.0)
        .toDouble();

    final cx = (tl.x + tr.x + br.x + bl.x) * 0.25;
    final cy = (tl.y + tr.y + br.y + bl.y) * 0.25;
    final dx = (cx - (width * 0.5)).abs() / math.max(1.0, width * 0.5);
    final dy = (cy - (height * 0.5)).abs() / math.max(1.0, height * 0.5);
    final centerScore = (1.0 - math.sqrt((dx * dx) + (dy * dy))).clamp(
      0.0,
      1.0,
    );

    final checker = _checkerboardScore(decoded: decoded, corners: corners);

    final borderPenalty = _borderPenaltyForCorners(
      corners: corners,
      width: width,
      height: height,
    );

    return ((areaScore * 0.25) +
            (aspectScore * 0.20) +
            (centerScore * 0.15) +
            (checker * 0.40)) *
        borderPenalty;
  }

  double _borderPenaltyForCorners({
    required List<BoardCorner> corners,
    required double width,
    required double height,
  }) {
    if (corners.isEmpty) {
      return 1.0;
    }

    final shortSide = math.max(1.0, math.min(width, height));
    final margin = math.max(2.0, shortSide * 0.015); // ~1.5% image margin.
    final nearMargin = margin * 2.0;

    int hardTouchCount = 0;
    int touchCount = 0;
    int nearCount = 0;
    double weightedPressure = 0.0;
    for (final c in corners) {
      final distanceX = math.min(c.x, (width - 1) - c.x);
      final distanceY = math.min(c.y, (height - 1) - c.y);
      final borderDistance = math.min(distanceX, distanceY);

      if (borderDistance <= margin * 0.35) {
        hardTouchCount += 1;
        weightedPressure += 1.0;
      } else if (borderDistance <= margin) {
        touchCount += 1;
        weightedPressure += 0.75;
      } else if (borderDistance <= nearMargin) {
        nearCount += 1;
        weightedPressure += 0.35;
      }
    }

    double penalty = 1.0 - ((weightedPressure / corners.length) * 0.55);
    final hardOrTouch = hardTouchCount + touchCount;
    if (hardOrTouch >= 3) {
      penalty *= 0.78;
    } else if (hardOrTouch == 2) {
      penalty *= 0.88;
    }
    if (hardTouchCount >= 1 && nearCount >= 2) {
      penalty *= 0.92;
    }
    return penalty.clamp(0.50, 1.0).toDouble();
  }

  double _checkerboardScore({
    required _DecodedLuma decoded,
    required List<BoardCorner> corners,
  }) {
    final sideTop = _distance(corners[0], corners[1]);
    final sideRight = _distance(corners[1], corners[2]);
    final sideBottom = _distance(corners[2], corners[3]);
    final sideLeft = _distance(corners[3], corners[0]);
    final minSide = math.min(
      math.min(sideTop, sideBottom),
      math.min(sideLeft, sideRight),
    );
    if (minSide < 20) {
      return 0.0;
    }

    final warped = _warpBoardLuma(
      decoded: decoded,
      corners: corners,
      targetSize: checkerTargetSize.clamp(96, 256),
    );
    if (warped == null) {
      return 0.0;
    }

    final cellWidth = warped.size / 8.0;
    final cellHeight = warped.size / 8.0;
    final cellMeans = List<double>.filled(64, 0.0);
    double even = 0;
    double odd = 0;
    int evenCount = 0;
    int oddCount = 0;

    for (int row = 0; row < 8; row++) {
      for (int col = 0; col < 8; col++) {
        final x0 = (col * cellWidth + cellWidth * 0.2).floor().clamp(
          0,
          warped.size - 1,
        );
        final y0 = (row * cellHeight + cellHeight * 0.2).floor().clamp(
          0,
          warped.size - 1,
        );
        final x1 = (((col + 1) * cellWidth) - cellWidth * 0.2).floor().clamp(
          x0,
          warped.size - 1,
        );
        final y1 = (((row + 1) * cellHeight) - cellHeight * 0.2).floor().clamp(
          y0,
          warped.size - 1,
        );
        final mean = _meanRectLuma(
          luma: warped.luma,
          width: warped.size,
          left: x0,
          top: y0,
          right: x1,
          bottom: y1,
        );
        final idx = row * 8 + col;
        cellMeans[idx] = mean;
        if (((row + col) & 1) == 0) {
          even += mean;
          evenCount += 1;
        } else {
          odd += mean;
          oddCount += 1;
        }
      }
    }

    if (evenCount == 0 || oddCount == 0) {
      return 0.0;
    }
    final evenMean = even / evenCount;
    final oddMean = odd / oddCount;
    final parityContrast = ((evenMean - oddMean).abs() / 96.0).clamp(0.0, 1.0);

    double adjacency = 0;
    int adjacencyCount = 0;
    for (int row = 0; row < 8; row++) {
      for (int col = 0; col < 7; col++) {
        final a = cellMeans[row * 8 + col];
        final b = cellMeans[row * 8 + col + 1];
        adjacency += (a - b).abs();
        adjacencyCount += 1;
      }
    }
    for (int col = 0; col < 8; col++) {
      for (int row = 0; row < 7; row++) {
        final a = cellMeans[row * 8 + col];
        final b = cellMeans[(row + 1) * 8 + col];
        adjacency += (a - b).abs();
        adjacencyCount += 1;
      }
    }
    final adjacencyScore = adjacencyCount == 0
        ? 0.0
        : (adjacency / adjacencyCount / 96.0).clamp(0.0, 1.0);

    return ((parityContrast * 0.70) + (adjacencyScore * 0.30))
        .clamp(0.0, 1.0)
        .toDouble();
  }

  double _checkerboardWindowScore({
    required _WarpedBoard warped,
    required double u0,
    required double v0,
    required double u1,
    required double v1,
  }) {
    final left = (u0 * warped.size).floor().clamp(0, warped.size - 1);
    final top = (v0 * warped.size).floor().clamp(0, warped.size - 1);
    final right = (u1 * warped.size).ceil().clamp(left + 1, warped.size);
    final bottom = (v1 * warped.size).ceil().clamp(top + 1, warped.size);
    final width = right - left;
    final height = bottom - top;
    if (width < 32 || height < 32) {
      return 0.0;
    }

    final cellWidth = width / 8.0;
    final cellHeight = height / 8.0;
    final cellMeans = List<double>.filled(64, 0.0);
    double even = 0;
    double odd = 0;
    int evenCount = 0;
    int oddCount = 0;

    for (int row = 0; row < 8; row++) {
      for (int col = 0; col < 8; col++) {
        final x0 = (left + (col * cellWidth) + cellWidth * 0.2).floor().clamp(
          0,
          warped.size - 1,
        );
        final y0 = (top + (row * cellHeight) + cellHeight * 0.2).floor().clamp(
          0,
          warped.size - 1,
        );
        final x1 = (left + ((col + 1) * cellWidth) - cellWidth * 0.2)
            .floor()
            .clamp(x0, warped.size - 1);
        final y1 = (top + ((row + 1) * cellHeight) - cellHeight * 0.2)
            .floor()
            .clamp(y0, warped.size - 1);
        final mean = _meanRectLuma(
          luma: warped.luma,
          width: warped.size,
          left: x0,
          top: y0,
          right: x1,
          bottom: y1,
        );
        final idx = row * 8 + col;
        cellMeans[idx] = mean;
        if (((row + col) & 1) == 0) {
          even += mean;
          evenCount += 1;
        } else {
          odd += mean;
          oddCount += 1;
        }
      }
    }
    if (evenCount == 0 || oddCount == 0) {
      return 0.0;
    }

    final parityContrast = ((even / evenCount - odd / oddCount).abs() / 96.0)
        .clamp(0.0, 1.0);
    double adjacency = 0;
    int adjacencyCount = 0;
    for (int row = 0; row < 8; row++) {
      for (int col = 0; col < 7; col++) {
        final a = cellMeans[row * 8 + col];
        final b = cellMeans[row * 8 + col + 1];
        adjacency += (a - b).abs();
        adjacencyCount += 1;
      }
    }
    for (int col = 0; col < 8; col++) {
      for (int row = 0; row < 7; row++) {
        final a = cellMeans[row * 8 + col];
        final b = cellMeans[(row + 1) * 8 + col];
        adjacency += (a - b).abs();
        adjacencyCount += 1;
      }
    }
    final adjacencyScore = adjacencyCount == 0
        ? 0.0
        : (adjacency / adjacencyCount / 96.0).clamp(0.0, 1.0);
    return ((parityContrast * 0.70) + (adjacencyScore * 0.30))
        .clamp(0.0, 1.0)
        .toDouble();
  }

  _WarpedBoard? _warpBoardLuma({
    required _DecodedLuma decoded,
    required List<BoardCorner> corners,
    required int targetSize,
  }) {
    final homography = _homographyUnitToQuad(corners);
    if (homography == null) {
      return null;
    }

    final out = Uint8List(targetSize * targetSize);
    for (int y = 0; y < targetSize; y++) {
      final v = (y + 0.5) / targetSize;
      final rowOffset = y * targetSize;
      for (int x = 0; x < targetSize; x++) {
        final u = (x + 0.5) / targetSize;
        final p = _applyHomography(homography, u, v);
        final sample = _sampleBilinear(
          luma: decoded.luma,
          width: decoded.width,
          height: decoded.height,
          x: p.x,
          y: p.y,
        );
        out[rowOffset + x] = sample.round().clamp(0, 255);
      }
    }
    return _WarpedBoard(size: targetSize, luma: out);
  }

  List<double>? _homographyUnitToQuad(List<BoardCorner> corners) {
    if (corners.length != 4) {
      return null;
    }
    final points = <_PointPair>[
      _PointPair(u: 0, v: 0, x: corners[0].x, y: corners[0].y),
      _PointPair(u: 1, v: 0, x: corners[1].x, y: corners[1].y),
      _PointPair(u: 1, v: 1, x: corners[2].x, y: corners[2].y),
      _PointPair(u: 0, v: 1, x: corners[3].x, y: corners[3].y),
    ];

    final a = List<List<double>>.generate(
      8,
      (_) => List<double>.filled(8, 0),
      growable: false,
    );
    final b = List<double>.filled(8, 0);

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

    final x = List<double>.filled(n, 0);
    for (int i = 0; i < n; i++) {
      x[i] = aug[i][n];
    }
    return x;
  }

  _PointD _applyHomography(List<double> h, double u, double v) {
    final den = (h[6] * u) + (h[7] * v) + h[8];
    if (den.abs() < 1e-9) {
      return const _PointD(0, 0);
    }
    final x = ((h[0] * u) + (h[1] * v) + h[2]) / den;
    final y = ((h[3] * u) + (h[4] * v) + h[5]) / den;
    return _PointD(x, y);
  }

  double _sampleBilinear({
    required Uint8List luma,
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

    final p00 = luma[y0 * width + x0].toDouble();
    final p10 = luma[y0 * width + x1].toDouble();
    final p01 = luma[y1 * width + x0].toDouble();
    final p11 = luma[y1 * width + x1].toDouble();

    final top = p00 + ((p10 - p00) * tx);
    final bottom = p01 + ((p11 - p01) * tx);
    return top + ((bottom - top) * ty);
  }

  double _meanRectLuma({
    required Uint8List luma,
    required int width,
    required int left,
    required int top,
    required int right,
    required int bottom,
  }) {
    double sum = 0;
    int count = 0;
    for (int y = top; y <= bottom; y++) {
      final rowOffset = y * width;
      for (int x = left; x <= right; x++) {
        sum += luma[rowOffset + x];
        count += 1;
      }
    }
    if (count == 0) {
      return 0;
    }
    return sum / count;
  }

  double _quadArea(List<BoardCorner> c) {
    double sum = 0;
    for (int i = 0; i < 4; i++) {
      final p = c[i];
      final n = c[(i + 1) % 4];
      sum += p.x * n.y - n.x * p.y;
    }
    return sum.abs() * 0.5;
  }

  double _distance(BoardCorner a, BoardCorner b) {
    final dx = a.x - b.x;
    final dy = a.y - b.y;
    return math.sqrt((dx * dx) + (dy * dy));
  }
}

class _OpenCvVariant {
  const _OpenCvVariant({required this.label, required this.bytes});

  final String label;
  final Uint8List bytes;
}

class _ScoredGeometry {
  const _ScoredGeometry({
    required this.geometry,
    required this.score,
    required this.source,
    required this.decodeBytes,
  });

  final BoardGeometry geometry;
  final double score;
  final String source;
  final Uint8List decodeBytes;
}

class _DecodedLuma {
  const _DecodedLuma({
    required this.width,
    required this.height,
    required this.luma,
  });

  final int width;
  final int height;
  final Uint8List luma;
}

class _WarpedBoard {
  const _WarpedBoard({required this.size, required this.luma});

  final int size;
  final Uint8List luma;
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

class _RefineOutcome {
  const _RefineOutcome({required this.geometry, required this.reason});

  final BoardGeometry geometry;
  final String reason;
}

class _ScoredQuadCandidate {
  const _ScoredQuadCandidate({
    required this.corners,
    required this.score,
    required this.label,
  });

  final List<BoardCorner> corners;
  final double score;
  final String label;
}

class _RoiBounds {
  const _RoiBounds({
    required this.left,
    required this.top,
    required this.right,
    required this.bottom,
  });

  final int left;
  final int top;
  final int right;
  final int bottom;

  int get width => math.max(0, (right - left) + 1);
  int get height => math.max(0, (bottom - top) + 1);
}

class _BinaryRoiMap {
  const _BinaryRoiMap({
    required this.label,
    required this.width,
    required this.height,
    required this.map,
  });

  final String label;
  final int width;
  final int height;
  final Uint8List map;
}

class _RoiComponent {
  const _RoiComponent({required this.area, required this.boundary});

  final int area;
  final List<_PointI> boundary;
}

class _PointI {
  const _PointI(this.x, this.y);

  final int x;
  final int y;
}

class _PointD {
  const _PointD(this.x, this.y);

  final double x;
  final double y;
}

class _GeometryQualitySummary {
  const _GeometryQualitySummary({
    required this.checker,
    required this.regularity,
    required this.edgeFrame,
    required this.combined,
  });

  final double checker;
  final double regularity;
  final double edgeFrame;
  final double combined;
}

class _ForcedCandidate {
  const _ForcedCandidate({
    required this.label,
    required this.geometry,
    required this.quality,
  });

  final String label;
  final BoardGeometry geometry;
  final _GeometryQualitySummary quality;
}

class _LineFallbackCandidate {
  const _LineFallbackCandidate({
    required this.label,
    required this.geometry,
    required this.quality,
    required this.score,
    required this.reason,
  });

  final String label;
  final BoardGeometry geometry;
  final _GeometryQualitySummary quality;
  final double score;
  final String reason;
}
