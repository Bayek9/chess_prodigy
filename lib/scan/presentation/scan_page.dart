import 'dart:convert';
import 'dart:ui' as ui;

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image_picker/image_picker.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:tflite_flutter/tflite_flutter.dart' as tfl;

import '../data/services/asset_scan_validation_dataset_loader.dart';
import '../data/services/basic_fen_builder.dart';
import '../data/services/basic_position_validator.dart';
import '../data/services/default_scan_pipeline.dart';
import '../domain/entities/board_geometry.dart';
import '../domain/entities/board_scan_position.dart';
import '../domain/entities/scan_image.dart';
import '../domain/entities/scan_validation_dataset.dart';
import '../domain/services/fen_builder.dart';
import '../domain/services/grid_square_mapper.dart';
import '../domain/services/position_validator.dart';
import '../domain/services/board_presence_classifier.dart';
import '../domain/usecases/run_scan_dataset_validation_use_case.dart';
import '../domain/usecases/scan_position_use_case.dart';
import 'widgets/board_correction_editor.dart';
import 'widgets/piece_chooser_sheet.dart';

enum _ScanCaptureDomain { photoReal, photoPrint, screen }

class _RoutingResolution {
  const _RoutingResolution({
    required this.domain,
    required this.isAuto,
    required this.reason,
    this.alternateDomain,
    this.selectedScore,
    this.alternateScore,
    this.ambiguous = false,
  });

  final _ScanCaptureDomain domain;
  final bool isAuto;
  final String reason;
  final _ScanCaptureDomain? alternateDomain;
  final double? selectedScore;
  final double? alternateScore;
  final bool ambiguous;
}

enum _FieldExpectedDomain { screen, photoReal }

enum _FieldExpectedClass { board, noBoard }

class _FieldTestEntry {
  const _FieldTestEntry({
    required this.index,
    required this.timestamp,
    required this.expectedDomain,
    required this.expectedClass,
    required this.tPrimaryMs,
    required this.tAltMs,
    required this.acquisitionSource,
    required this.chosenDomain,
    required this.decision,
    required this.gateDecisionRaw,
    required this.finalDecisionRaw,
    required this.boardDetected,
    required this.postWarpGridness,
    required this.minPostWarpGridness,
    required this.rejectedPostWarpGridness,
    this.bypassReason,
    required this.routingDebug,
    required this.detectorDebug,
    required this.imagePath,
  });

  final int index;
  final DateTime timestamp;
  final _FieldExpectedDomain expectedDomain;
  final _FieldExpectedClass expectedClass;
  final int tPrimaryMs;
  final int tAltMs;
  final String acquisitionSource;
  final _ScanCaptureDomain chosenDomain;
  final String decision; // reject | gray | accept
  final String gateDecisionRaw; // reject_strong_no_board | allow_* | unknown
  final String finalDecisionRaw; // latest decision=* token in detector debug
  final bool boardDetected;
  final double? postWarpGridness;
  final double? minPostWarpGridness;
  final bool rejectedPostWarpGridness;
  final String? bypassReason;
  final String routingDebug;
  final String detectorDebug;
  final String imagePath;

  Map<String, Object?> toJson() {
    return <String, Object?>{
      'index': index,
      'timestamp': timestamp.toIso8601String(),
      'expectedDomain': expectedDomain.name,
      'expectedClass': expectedClass.name,
      'tPrimaryMs': tPrimaryMs,
      'tAltMs': tAltMs,
      'acquisitionSource': acquisitionSource,
      'chosenDomain': chosenDomain.name,
      'decision': decision,
      'gateDecisionRaw': gateDecisionRaw,
      'finalDecisionRaw': finalDecisionRaw,
      'boardDetected': boardDetected,
      'postWarpGridness': postWarpGridness,
      'minPostWarpGridness': minPostWarpGridness,
      'rejectedPostWarpGridness': rejectedPostWarpGridness,
      'bypassReason': bypassReason,
      'routingDebug': routingDebug,
      'detectorDebug': detectorDebug,
      'imagePath': imagePath,
    };
  }

  static _FieldTestEntry? fromJson(Map<String, dynamic> json) {
    final expectedDomain = switch (json['expectedDomain']) {
      'screen' => _FieldExpectedDomain.screen,
      'photoReal' => _FieldExpectedDomain.photoReal,
      _ => null,
    };
    final expectedClass = switch (json['expectedClass']) {
      'board' => _FieldExpectedClass.board,
      'noBoard' => _FieldExpectedClass.noBoard,
      _ => null,
    };
    final chosenDomain = switch (json['chosenDomain']) {
      'photoReal' => _ScanCaptureDomain.photoReal,
      'photoPrint' => _ScanCaptureDomain.photoPrint,
      'screen' => _ScanCaptureDomain.screen,
      _ => null,
    };
    final timestampRaw = json['timestamp']?.toString();
    final timestamp = timestampRaw == null
        ? null
        : DateTime.tryParse(timestampRaw);

    if (expectedDomain == null ||
        expectedClass == null ||
        chosenDomain == null ||
        timestamp == null) {
      return null;
    }

    return _FieldTestEntry(
      index: (json['index'] as num?)?.toInt() ?? 0,
      timestamp: timestamp,
      expectedDomain: expectedDomain,
      expectedClass: expectedClass,
      tPrimaryMs: (json['tPrimaryMs'] as num?)?.toInt() ?? 0,
      tAltMs: (json['tAltMs'] as num?)?.toInt() ?? 0,
      acquisitionSource: json['acquisitionSource']?.toString() ?? 'unknown',
      chosenDomain: chosenDomain,
      decision: json['decision']?.toString() ?? 'gray',
      gateDecisionRaw: json['gateDecisionRaw']?.toString() ?? 'unknown',
      finalDecisionRaw: json['finalDecisionRaw']?.toString() ?? 'unknown',
      boardDetected: json['boardDetected'] == true,
      postWarpGridness: (json['postWarpGridness'] as num?)?.toDouble(),
      minPostWarpGridness: (json['minPostWarpGridness'] as num?)?.toDouble(),
      rejectedPostWarpGridness: json['rejectedPostWarpGridness'] == true,
      bypassReason: json['bypassReason']?.toString(),
      routingDebug: json['routingDebug']?.toString() ?? '',
      detectorDebug: json['detectorDebug']?.toString() ?? '',
      imagePath: json['imagePath']?.toString() ?? '',
    );
  }
}

class ScanPage extends StatefulWidget {
  const ScanPage({super.key, required this.onAnalyzeFen});

  final ValueChanged<String> onAnalyzeFen;

  @override
  State<ScanPage> createState() => _ScanPageState();
}

class _ScanPageState extends State<ScanPage> {
  static const String _scanCoreRevision = 'scan-core-r2026-02-27-01';
  static const double _photoRealAcceptThreshold = 0.81;
  static const double _photoRealRejectThreshold = 0.50;
  static const double _screenAcceptThreshold = 0.89;
  static const double _screenRejectThreshold = 0.57;
  static const double _screenOpenCvMinBoardConfidence = 0.30;
  static const double _screenOpenCvMinBoardConfidenceLineFallback = 0.34;
  static const double _screenLooseRetryStrongProbability = 0.55;
  static const double _screenLooseOpenCvMinBoardConfidence = 0.20;
  static const double _screenLooseOpenCvMinBoardConfidenceLineFallback = 0.24;
  static const double _screenMinPostWarpGridness = 0.11;
  static const double _screenGridnessRescueMinPostWarpGridness = 0.08;
  static const double _autoRoutingAmbiguousScoreDelta = 0.05;
  static const double _autoRoutingAlternateRetryMinScore = 0.35;
  static const double _alternateBypassMinBoardQuality = 0.40;
  static const double _alternateBypassMinBoardConfidence = 0.35;
  static const double _alternateBypassMinBoardAreaRatio = 0.16;
  static const double _gridnessRescueMinBoardQuality = 0.30;
  static const double _gridnessRescueMinBoardConfidence = 0.35;
  static const double _gridnessRescueMinBoardAreaRatio = 0.12;
  static const int _fieldProtocolBucketTarget = 10;
  static const int _fieldProtocolTotalTarget = 40;
  static const String _photoBoardModelAssetPath =
      'assets/scan_models/board_binary_photo.tflite';
  static const String _screenBoardModelAssetPath =
      'assets/scan_models/board_binary_screen.tflite';
  static const String _fieldProtocolPrefsKey = 'scan.field_protocol.entries';
  static const String _fieldProtocolDomainPrefsKey =
      'scan.field_protocol.expected_domain';
  static const String _fieldProtocolClassPrefsKey =
      'scan.field_protocol.expected_class';

  final ImagePicker _picker = ImagePicker();
  final GridSquareMapper _squareMapper = const GridSquareMapper();
  final FenBuilder _fenBuilder = const BasicFenBuilder();
  final PositionValidator _validator = const BasicPositionValidator();
  late final ScanPositionUseCase _scanUseCasePhotoReal;
  late final ScanPositionUseCase _scanUseCaseScreen;
  late final ScanPositionUseCase _scanUseCaseScreenLoose;
  late final ScanPositionUseCase _scanUseCaseScreenGridnessRescue;
  late final ScanPositionUseCase _scanUseCasePhotoRealNoGate;
  late final ScanPositionUseCase _scanUseCaseScreenNoGate;
  late final ScanPositionUseCase _datasetScanUseCase;
  late final ScanPositionUseCase _datasetScanUseCaseFast;
  late final RunScanDatasetValidationUseCase _datasetValidationUseCase;
  late final RunScanDatasetValidationUseCase _datasetValidationUseCaseFast;
  late final BoardPresenceClassifier _photoGateClassifier;
  late final BoardPresenceClassifier _screenGateClassifier;

  ScanInputImage? _selectedImage;
  ScanPipelineResult? _scanResult;
  BoardScanPosition? _editablePosition;
  ScanDatasetValidationReport? _datasetReport;
  ScanCaseEvaluation? _selectedEvaluation;
  PositionValidationResult _validation = const PositionValidationResult();
  String? _finalFen;
  bool _flipped = false;
  bool _isScanning = false;
  bool _isRunningDatasetValidation = false;
  bool _isLoadingSelectedEvaluation = false;
  bool _isAutoFillingAllExpected = false;
  bool _useFastDatasetValidation = true;
  String? _datasetProgressLabel;
  String? _errorMessage;
  Size? _selectedImageSize;
  Object? _selectedImageSizeToken;
  List<BoardCorner> _manualCorners = const <BoardCorner>[];
  ImageSource? _selectedImageSource;
  _ScanCaptureDomain _selectedCaptureDomain = _ScanCaptureDomain.screen;
  _ScanCaptureDomain? _captureDomainOverride;
  String _routingDebugLabel = 'mode=auto';
  int _autoRouteScanCount = 0;
  int _autoRouteRetryCount = 0;
  int _lastScanPrimaryMs = 0;
  int _lastScanAltMs = 0;
  String _lastReportedGateDecisionRaw = 'unknown';
  String _lastReportedFinalDecisionRaw = 'unknown';
  String? _lastBypassReason;
  bool _selectedImageHasExif = false;
  bool _selectedImageLooksScreenshot = false;
  final Map<String, int> _gateDecisionCounters = <String, int>{
    'strong_reject': 0,
    'gray': 0,
    'strong_accept': 0,
  };
  _FieldExpectedDomain _fieldExpectedDomain = _FieldExpectedDomain.screen;
  _FieldExpectedClass _fieldExpectedClass = _FieldExpectedClass.board;
  final List<_FieldTestEntry> _fieldTestEntries = <_FieldTestEntry>[];

  @override
  void initState() {
    super.initState();
    if (kDebugMode) {
      debugPrint('[scan][tflite] runtime=${tfl.version}');
    }
    _loadFieldProtocolState();
    _photoGateClassifier =
        DefaultScanPipelineFactory.boardPresenceClassifierForAsset(
          modelAssetPath: _photoBoardModelAssetPath,
        );
    _screenGateClassifier =
        DefaultScanPipelineFactory.boardPresenceClassifierForAsset(
          modelAssetPath: _screenBoardModelAssetPath,
        );
    _scanUseCasePhotoReal = DefaultScanPipelineFactory.create(
      validator: _validator,
      fenBuilder: _fenBuilder,
      useBoardPresenceGate: true,
      boardPresenceThreshold: _photoRealAcceptThreshold,
      boardPresenceRejectThreshold: _photoRealRejectThreshold,
      boardPresenceModelAssetPath: _photoBoardModelAssetPath,
      useFallbackForReject: true,
    );
    _scanUseCaseScreen = DefaultScanPipelineFactory.create(
      validator: _validator,
      fenBuilder: _fenBuilder,
      useBoardPresenceGate: true,
      boardPresenceThreshold: _screenAcceptThreshold,
      boardPresenceRejectThreshold: _screenRejectThreshold,
      boardPresenceModelAssetPath: _screenBoardModelAssetPath,
      useFallbackForReject: false,
      openCvMinBoardConfidence: _screenOpenCvMinBoardConfidence,
      openCvMinBoardConfidenceLineFallback:
          _screenOpenCvMinBoardConfidenceLineFallback,
      minPostWarpGridness: _screenMinPostWarpGridness,
    );
    _scanUseCaseScreenLoose = DefaultScanPipelineFactory.create(
      validator: _validator,
      fenBuilder: _fenBuilder,
      useBoardPresenceGate: true,
      boardPresenceThreshold: _screenAcceptThreshold,
      boardPresenceRejectThreshold: _screenRejectThreshold,
      boardPresenceModelAssetPath: _screenBoardModelAssetPath,
      useFallbackForReject: false,
      openCvMinBoardConfidence: _screenLooseOpenCvMinBoardConfidence,
      openCvMinBoardConfidenceLineFallback:
          _screenLooseOpenCvMinBoardConfidenceLineFallback,
      minPostWarpGridness: _screenMinPostWarpGridness,
    );
    _scanUseCaseScreenGridnessRescue = DefaultScanPipelineFactory.create(
      validator: _validator,
      fenBuilder: _fenBuilder,
      useBoardPresenceGate: true,
      boardPresenceThreshold: _screenAcceptThreshold,
      boardPresenceRejectThreshold: _screenRejectThreshold,
      boardPresenceModelAssetPath: _screenBoardModelAssetPath,
      useFallbackForReject: false,
      openCvMinBoardConfidence: _screenOpenCvMinBoardConfidence,
      openCvMinBoardConfidenceLineFallback:
          _screenOpenCvMinBoardConfidenceLineFallback,
      minPostWarpGridness: _screenGridnessRescueMinPostWarpGridness,
    );
    _scanUseCasePhotoRealNoGate = DefaultScanPipelineFactory.create(
      validator: _validator,
      fenBuilder: _fenBuilder,
      useBoardPresenceGate: false,
    );
    _scanUseCaseScreenNoGate = DefaultScanPipelineFactory.create(
      validator: _validator,
      fenBuilder: _fenBuilder,
      useBoardPresenceGate: false,
      openCvMinBoardConfidence: _screenOpenCvMinBoardConfidence,
      openCvMinBoardConfidenceLineFallback:
          _screenOpenCvMinBoardConfidenceLineFallback,
      minPostWarpGridness: _screenMinPostWarpGridness,
    );
    _datasetScanUseCase = DefaultScanPipelineFactory.create(
      validator: _validator,
      fenBuilder: _fenBuilder,
      lowLatencyDetector: false,
      useBoardPresenceGate: true,
    );
    _datasetScanUseCaseFast = DefaultScanPipelineFactory.create(
      validator: _validator,
      fenBuilder: _fenBuilder,
      lowLatencyDetector: true,
      useBoardPresenceGate: true,
    );
    _datasetValidationUseCase = RunScanDatasetValidationUseCase(
      scanPipeline: _datasetScanUseCase,
      datasetLoader: const AssetScanValidationDatasetLoader(),
      compareFen: false,
    );
    _datasetValidationUseCaseFast = RunScanDatasetValidationUseCase(
      scanPipeline: _datasetScanUseCaseFast,
      datasetLoader: const AssetScanValidationDatasetLoader(),
      compareFen: false,
    );
  }

  Future<void> _pickImage(ImageSource source) async {
    try {
      final file = await _picker.pickImage(
        source: source,
        imageQuality: 90,
        maxWidth: 1920,
        maxHeight: 1920,
      );
      if (file == null) {
        return;
      }
      final bytes = await file.readAsBytes();
      final hasExif = _hasExifMetadata(bytes);
      final looksScreenshot = _looksLikeScreenshotPath(file.path);
      final captureDomain = _resolveCaptureDomain(
        hasExif: hasExif,
        looksScreenshot: looksScreenshot,
      );
      if (!mounted) {
        return;
      }
      setState(() {
        _selectedImage = ScanInputImage(path: file.path, bytes: bytes);
        _selectedImageSource = source;
        _selectedCaptureDomain = captureDomain;
        _captureDomainOverride = null;
        _routingDebugLabel =
            'mode=auto hint=${_captureDomainLabel(captureDomain)} pending_scan';
        _selectedImageHasExif = hasExif;
        _selectedImageLooksScreenshot = looksScreenshot;
        _scanResult = null;
        _editablePosition = null;
        _finalFen = null;
        _errorMessage = null;
        _lastScanPrimaryMs = 0;
        _lastScanAltMs = 0;
        _lastReportedGateDecisionRaw = 'unknown';
        _lastReportedFinalDecisionRaw = 'unknown';
        _lastBypassReason = null;
        _selectedImageSize = null;
        _manualCorners = const <BoardCorner>[];
      });
      await _resolveSelectedImageSize(bytes);
    } catch (e) {
      if (!mounted) {
        return;
      }
      setState(() => _errorMessage = 'Image loading failed: $e');
    }
  }

  Future<void> _runScan() async {
    final image = _selectedImage;
    if (image == null || _isScanning) {
      return;
    }

    setState(() {
      _isScanning = true;
      _errorMessage = null;
    });

    try {
      final routing = await _resolveRoutingForScan(image);
      var resolvedDomain = routing.domain;
      final allowScreenLooseRetry = !routing.isAuto;
      final primaryStopwatch = Stopwatch()..start();
      var result = await _scanWithDomain(
        domain: resolvedDomain,
        image: image,
        allowScreenLooseRetry: allowScreenLooseRetry,
      );
      primaryStopwatch.stop();
      final primaryScanMs = primaryStopwatch.elapsedMilliseconds;
      final primaryGateRaw = _gateDecisionRawFromDetectorDebug(
        result.detectorDebug,
      );
      var alternateScanMs = 0;
      var routingDebug = routing.reason;

      var retries = 0;
      const maxRetries = 2;
      final shouldRetryAlternate =
          routing.isAuto &&
          !result.boardDetected &&
          routing.alternateDomain != null &&
          (routing.ambiguous ||
              ((routing.alternateScore ?? double.negativeInfinity) >=
                  _autoRoutingAlternateRetryMinScore) ||
              primaryGateRaw == 'allow_strong_accept');
      var finalUsedBypassNoGate = false;
      var usedGridnessRescue = false;
      String? bypassReason;
      if (shouldRetryAlternate && retries < maxRetries) {
        retries += 1;
        final alternateDomain = routing.alternateDomain!;
        final retryAlternateWithoutGate =
            primaryGateRaw == 'allow_strong_accept';
        final alternateStopwatch = Stopwatch()..start();
        final alternateResult = await _scanWithDomain(
          domain: alternateDomain,
          image: image,
          bypassGate: retryAlternateWithoutGate,
          allowScreenLooseRetry: false,
        );
        alternateStopwatch.stop();
        alternateScanMs = alternateStopwatch.elapsedMilliseconds;
        final alternateBypassQualityPass =
            !retryAlternateWithoutGate ||
            _passesAlternateBypassQualityGate(alternateResult);
        final switched = _shouldSwitchToAlternateResult(
          primary: result,
          alternate: alternateResult,
          requireBypassQualityGate: retryAlternateWithoutGate,
        );
        if (switched) {
          result = alternateResult;
          resolvedDomain = alternateDomain;
          if (retryAlternateWithoutGate) {
            finalUsedBypassNoGate = true;
            bypassReason = 'primary_allow_strong_accept_detector_failed';
          }
        }
        routingDebug =
            '$routingDebug retry_alt=${_captureDomainLabel(alternateDomain)} '
            'alt_score=${_fmtRoutingNumber(routing.alternateScore)} '
            'retry_threshold=${_autoRoutingAlternateRetryMinScore.toStringAsFixed(3)} '
            'primary_gate_raw=$primaryGateRaw '
            'retry_count=$retries/$maxRetries '
            'alt_bypass_gate=$retryAlternateWithoutGate '
            'alt_quality_gate_pass=$alternateBypassQualityPass '
            'switched=$switched alt_board=${alternateResult.boardDetected} '
            't_primary_ms=$primaryScanMs t_alt_ms=$alternateScanMs';
      }

      final currentGateDecisionRaw = finalUsedBypassNoGate
          ? 'bypass_no_gate'
          : _gateDecisionRawFromDetectorDebug(result.detectorDebug);
      final currentFinalDecisionRaw = _finalDecisionRawFromDetectorDebug(
        result.detectorDebug,
      );
      final shouldRetryGridnessRescue =
          retries < maxRetries &&
          resolvedDomain == _ScanCaptureDomain.screen &&
          currentFinalDecisionRaw == 'reject_post_warp_gridness' &&
          (currentGateDecisionRaw == 'allow_strong_accept' ||
              currentGateDecisionRaw == 'bypass_no_gate') &&
          _passesGridnessRescuePrecheck(result);
      if (shouldRetryGridnessRescue) {
        retries += 1;
        final rescueStopwatch = Stopwatch()..start();
        final rescueResult = await _scanUseCaseScreenGridnessRescue.execute(
          image,
        );
        rescueStopwatch.stop();
        final rescueMs = rescueStopwatch.elapsedMilliseconds;
        alternateScanMs += rescueMs;
        final rescueQualityPass = _passesGridnessRescuePrecheck(rescueResult);
        final rescueSwitched = rescueResult.boardDetected && rescueQualityPass;
        if (rescueSwitched) {
          result = rescueResult;
          usedGridnessRescue = true;
          finalUsedBypassNoGate = false;
          bypassReason = 'gridness_rescue_post_warp_gridness';
        }
        routingDebug =
            '$routingDebug gridness_rescue=true '
            'gridness_min=${_screenGridnessRescueMinPostWarpGridness.toStringAsFixed(3)} '
            'rescue_switched=$rescueSwitched '
            'rescue_quality_pass=$rescueQualityPass '
            'rescue_board=${rescueResult.boardDetected} '
            'rescue_ms=$rescueMs '
            'retry_count=$retries/$maxRetries';
      }

      var reportedGateDecisionRaw = _gateDecisionRawFromDetectorDebug(
        result.detectorDebug,
      );
      var reportedFinalDecisionRaw = _finalDecisionRawFromDetectorDebug(
        result.detectorDebug,
      );
      if (finalUsedBypassNoGate && !usedGridnessRescue) {
        reportedGateDecisionRaw = 'bypass_no_gate';
        reportedFinalDecisionRaw = 'bypass_no_gate';
      }

      if (kDebugMode) {
        debugPrint(
          '[scan][auto-route] chosen=${_captureDomainLabel(resolvedDomain)} '
          '$routingDebug t_primary_ms=$primaryScanMs t_alt_ms=$alternateScanMs',
        );
      }

      if (!mounted) {
        return;
      }
      setState(() {
        _selectedCaptureDomain = resolvedDomain;
        _routingDebugLabel = routingDebug;
        _scanResult = result;
        _lastScanPrimaryMs = primaryScanMs;
        _lastScanAltMs = alternateScanMs;
        _lastReportedGateDecisionRaw = reportedGateDecisionRaw;
        _lastReportedFinalDecisionRaw = reportedFinalDecisionRaw;
        _lastBypassReason = bypassReason;
        _editablePosition = result.detectedPosition;
        _recordGateDecision(result.detectorDebug);
        if (routing.isAuto) {
          _autoRouteScanCount += 1;
          if (retries > 0) {
            _autoRouteRetryCount += 1;
          }
        }
        _refreshFenAndValidation();
      });
    } catch (e) {
      if (!mounted) {
        return;
      }
      setState(() => _errorMessage = 'Scan failed: $e');
    } finally {
      if (mounted) {
        setState(() => _isScanning = false);
      }
    }
  }

  Future<_RoutingResolution> _resolveRoutingForScan(
    ScanInputImage image,
  ) async {
    final overrideDomain = _captureDomainOverride;
    if (overrideDomain != null) {
      return _RoutingResolution(
        domain: overrideDomain,
        isAuto: false,
        reason: 'mode=override forced=${_captureDomainLabel(overrideDomain)}',
      );
    }
    return _resolveAutoRoutingForScan(image);
  }

  Future<_RoutingResolution> _resolveAutoRoutingForScan(
    ScanInputImage image,
  ) async {
    final hintedDomain = _resolveCaptureDomain(
      hasExif: _selectedImageHasExif,
      looksScreenshot: _selectedImageLooksScreenshot,
    );

    try {
      // Run gate predictions sequentially to avoid potential interpreter
      // concurrency issues on some TFLite bindings/delegates.
      final screenPrediction = await _screenGateClassifier.predict(image);
      final photoPrediction = await _photoGateClassifier.predict(image);
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
        rejectThreshold: _photoRealRejectThreshold,
      );

      if (screenScore == null && photoScore == null) {
        return _RoutingResolution(
          domain: hintedDomain,
          isAuto: true,
          reason:
              'mode=auto fallback=hint both_unavailable '
              'hint=${_captureDomainLabel(hintedDomain)}',
        );
      }

      final chooseScreen =
          photoScore == null ||
          (screenScore != null && screenScore >= photoScore);
      final domain = chooseScreen
          ? _ScanCaptureDomain.screen
          : _ScanCaptureDomain.photoReal;
      final alternate = chooseScreen
          ? _ScanCaptureDomain.photoReal
          : _ScanCaptureDomain.screen;
      final selectedScore = chooseScreen ? screenScore : photoScore;
      final alternateScore = chooseScreen ? photoScore : screenScore;
      final delta = (screenScore != null && photoScore != null)
          ? (screenScore - photoScore).abs()
          : null;
      final ambiguous =
          delta != null && delta < _autoRoutingAmbiguousScoreDelta;

      return _RoutingResolution(
        domain: domain,
        alternateDomain: alternate,
        selectedScore: selectedScore,
        alternateScore: alternateScore,
        ambiguous: ambiguous,
        isAuto: true,
        reason:
            'mode=auto '
            'screen_strong=${_fmtRoutingNumber(screenStrong)} '
            'screen_reject=${_screenRejectThreshold.toStringAsFixed(3)} '
            'screen_score=${_fmtRoutingNumber(screenScore)} '
            'photo_strong=${_fmtRoutingNumber(photoStrong)} '
            'photo_reject=${_photoRealRejectThreshold.toStringAsFixed(3)} '
            'photo_score=${_fmtRoutingNumber(photoScore)} '
            'delta=${_fmtRoutingNumber(delta)} '
            'ambiguity_rule=delta<${_autoRoutingAmbiguousScoreDelta.toStringAsFixed(3)} '
            'ambiguous=$ambiguous '
            'selected=${_captureDomainLabel(domain)}',
      );
    } catch (e) {
      return _RoutingResolution(
        domain: hintedDomain,
        isAuto: true,
        reason:
            'mode=auto fallback=hint predict_error=$e '
            'hint=${_captureDomainLabel(hintedDomain)}',
      );
    }
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

  String _fmtRoutingNumber(double? value) {
    return value == null ? 'na' : value.toStringAsFixed(3);
  }

  String _routingModeLabel() {
    return _captureDomainOverride == null ? 'auto' : 'override';
  }

  String _autoRoutingRetrySummary() {
    final total = _autoRouteScanCount;
    final retries = _autoRouteRetryCount;
    final percent = total == 0 ? 0.0 : (retries * 100.0 / total);
    return 'Auto retries: $retries/$total (${percent.toStringAsFixed(1)}%)';
  }

  void _setCaptureDomainOverride(_ScanCaptureDomain? domain) {
    setState(() {
      _captureDomainOverride = domain;
      if (domain != null) {
        _selectedCaptureDomain = domain;
        _routingDebugLabel =
            'mode=override forced=${_captureDomainLabel(domain)}';
      } else {
        _routingDebugLabel =
            'mode=auto hint=${_captureDomainLabel(_selectedCaptureDomain)}';
      }
    });
  }

  Future<ScanPipelineResult> _scanWithDomain({
    required _ScanCaptureDomain domain,
    required ScanInputImage image,
    bool bypassGate = false,
    bool allowScreenLooseRetry = false,
  }) async {
    final useCase = _scanUseCaseForDomain(domain, bypassGate: bypassGate);
    var result = await useCase.execute(image);
    if (domain == _ScanCaptureDomain.screen && allowScreenLooseRetry) {
      result = await _maybeRetryScreenLoose(image: image, result: result);
    }
    return result;
  }

  Future<ScanPipelineResult> _maybeRetryScreenLoose({
    required ScanInputImage image,
    required ScanPipelineResult result,
  }) async {
    if (result.boardDetected) {
      return result;
    }

    final detectorDebug = result.detectorDebug;
    final isStrongReject = detectorDebug.contains(
      'decision=reject_strong_no_board',
    );
    if (isStrongReject) {
      return result;
    }

    final gateAllowed = _extractGateAllowed(detectorDebug) ?? false;
    final strongProbability = _extractStrongProbability(detectorDebug);
    final shouldRetryLoose =
        gateAllowed ||
        (strongProbability != null &&
            strongProbability >= _screenLooseRetryStrongProbability);
    if (!shouldRetryLoose) {
      return result;
    }

    final looseResult = await _scanUseCaseScreenLoose.execute(image);
    if (kDebugMode) {
      debugPrint(
        '[scan][screen-retry] gate_allowed=$gateAllowed '
        'strong_prob=${strongProbability?.toStringAsFixed(3) ?? "na"} '
        'threshold=${_screenLooseRetryStrongProbability.toStringAsFixed(3)} '
        'loose_detected=${looseResult.boardDetected}',
      );
    }
    return looseResult;
  }

  ScanPositionUseCase _scanUseCaseForDomain(
    _ScanCaptureDomain domain, {
    bool bypassGate = false,
  }) {
    switch (domain) {
      case _ScanCaptureDomain.photoReal:
      case _ScanCaptureDomain.photoPrint:
        return bypassGate ? _scanUseCasePhotoRealNoGate : _scanUseCasePhotoReal;
      case _ScanCaptureDomain.screen:
        return bypassGate ? _scanUseCaseScreenNoGate : _scanUseCaseScreen;
    }
  }

  double? _extractStrongProbability(String detectorDebug) {
    final match = RegExp(
      r'strong_prob=([01](?:\.\d+)?)',
    ).firstMatch(detectorDebug);
    if (match == null || match.groupCount < 1) {
      return null;
    }
    return double.tryParse(match.group(1)!);
  }

  bool? _extractGateAllowed(String detectorDebug) {
    final match = RegExp(r'allowed=(true|false)').firstMatch(detectorDebug);
    if (match == null || match.groupCount < 1) {
      return null;
    }
    final value = match.group(1);
    if (value == 'true') {
      return true;
    }
    if (value == 'false') {
      return false;
    }
    return null;
  }

  String _gateAllowedLabel(ScanPipelineResult result) {
    final allowed = _extractGateAllowed(result.detectorDebug);
    return allowed == null ? 'unknown' : allowed.toString();
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
    final primaryQuality = _extractBoardQuality(primary.detectorDebug);
    final alternateQuality = _extractBoardQuality(alternate.detectorDebug);
    if (primaryQuality != null && alternateQuality != null) {
      if (alternateQuality > primaryQuality) {
        return true;
      }
      if (alternateQuality < primaryQuality) {
        return false;
      }
    }
    final primaryConfidence = _extractBoardConfidence(primary.detectorDebug);
    final alternateConfidence = _extractBoardConfidence(
      alternate.detectorDebug,
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
    final quality = _extractBoardQuality(result.detectorDebug);
    final confidence = _extractBoardConfidence(result.detectorDebug);
    final areaRatio = _extractDetectorMetric(
      result.detectorDebug,
      'board_area_ratio',
    );
    if (quality == null || confidence == null || areaRatio == null) {
      return false;
    }
    return quality >= _alternateBypassMinBoardQuality &&
        confidence >= _alternateBypassMinBoardConfidence &&
        areaRatio >= _alternateBypassMinBoardAreaRatio;
  }

  bool _passesGridnessRescuePrecheck(ScanPipelineResult result) {
    final quality = _extractBoardQuality(result.detectorDebug);
    final confidence = _extractBoardConfidence(result.detectorDebug);
    final areaRatio = _extractDetectorMetric(
      result.detectorDebug,
      'board_area_ratio',
    );
    if (quality == null || confidence == null || areaRatio == null) {
      return false;
    }
    return quality >= _gridnessRescueMinBoardQuality &&
        confidence >= _gridnessRescueMinBoardConfidence &&
        areaRatio >= _gridnessRescueMinBoardAreaRatio;
  }

  double? _extractBoardQuality(String detectorDebug) {
    return _extractDetectorMetric(detectorDebug, 'board_quality');
  }

  double? _extractBoardConfidence(String detectorDebug) {
    return _extractDetectorMetric(detectorDebug, 'board_confidence');
  }

  double? _extractDetectorMetric(String detectorDebug, String key) {
    final match = RegExp(
      '$key=([-+]?\\d+(?:\\.\\d+)?)',
    ).firstMatch(detectorDebug);
    if (match == null || match.groupCount < 1) {
      return null;
    }
    return double.tryParse(match.group(1)!);
  }

  bool _isGateFinalMismatch(ScanPipelineResult result) {
    final allowed = _extractGateAllowed(result.detectorDebug);
    return allowed != null && allowed != result.boardDetected;
  }

  _ScanCaptureDomain _resolveCaptureDomain({
    required bool hasExif,
    required bool looksScreenshot,
  }) {
    if (looksScreenshot) {
      return _ScanCaptureDomain.screen;
    }
    if (hasExif) {
      return _ScanCaptureDomain.photoReal;
    }
    return _ScanCaptureDomain.screen;
  }

  String _captureDomainLabel(_ScanCaptureDomain domain) {
    switch (domain) {
      case _ScanCaptureDomain.photoReal:
        return 'photo_real';
      case _ScanCaptureDomain.photoPrint:
        return 'photo_print';
      case _ScanCaptureDomain.screen:
        return 'screen';
    }
  }

  String _selectedImageSourceLabel() {
    final source = _selectedImageSource;
    if (source == null) {
      return 'unknown';
    }
    return source == ImageSource.camera ? 'camera' : 'gallery';
  }

  String _gateDecisionBucket(String detectorDebug) {
    // Final detector verdict takes precedence over the gate-only decision.
    if (detectorDebug.contains('board_rejected=true') ||
        detectorDebug.contains('which_path_won=rejected_')) {
      return 'strong_reject';
    }
    if (detectorDebug.contains('decision=reject_strong_no_board')) {
      return 'strong_reject';
    }
    if (detectorDebug.contains('decision=allow_strong_accept')) {
      return 'strong_accept';
    }
    if (detectorDebug.contains('decision=allow_gray_zone_fallback')) {
      return 'gray';
    }
    return 'gray';
  }

  void _recordGateDecision(String detectorDebug) {
    final bucket = _gateDecisionBucket(detectorDebug);
    _gateDecisionCounters[bucket] = (_gateDecisionCounters[bucket] ?? 0) + 1;
    final rejectCount = _gateDecisionCounters['strong_reject'] ?? 0;
    final grayCount = _gateDecisionCounters['gray'] ?? 0;
    final acceptCount = _gateDecisionCounters['strong_accept'] ?? 0;
    if (kDebugMode) {
      debugPrint(
        '[gate-metrics] reject=$rejectCount gray=$grayCount accept=$acceptCount',
      );
    }
  }

  String _decisionLabelFromDetectorDebug(String detectorDebug) {
    final bucket = _gateDecisionBucket(detectorDebug);
    switch (bucket) {
      case 'strong_reject':
        return 'reject';
      case 'strong_accept':
        return 'accept';
      default:
        return 'gray';
    }
  }

  String _gateDecisionRawFromDetectorDebug(String detectorDebug) {
    return _firstDecisionRawFromDetectorDebug(detectorDebug);
  }

  String _finalDecisionRawFromDetectorDebug(String detectorDebug) {
    final allMatches = RegExp(
      r'decision=([a-z_]+)',
    ).allMatches(detectorDebug).toList(growable: false);
    if (allMatches.isEmpty || allMatches.last.groupCount < 1) {
      return 'unknown';
    }
    return allMatches.last.group(1)!;
  }

  String _firstDecisionRawFromDetectorDebug(String detectorDebug) {
    final match = RegExp(r'decision=([a-z_]+)').firstMatch(detectorDebug);
    if (match == null || match.groupCount < 1) {
      return 'unknown';
    }
    return match.group(1)!;
  }

  bool _isRejectedPostWarpGridness(String detectorDebug) {
    return detectorDebug.contains('decision=reject_post_warp_gridness');
  }

  _FieldExpectedDomain _normalizedFieldDomain(_ScanCaptureDomain domain) {
    if (domain == _ScanCaptureDomain.screen) {
      return _FieldExpectedDomain.screen;
    }
    return _FieldExpectedDomain.photoReal;
  }

  String _fieldExpectedDomainLabel(_FieldExpectedDomain domain) {
    switch (domain) {
      case _FieldExpectedDomain.screen:
        return 'screen';
      case _FieldExpectedDomain.photoReal:
        return 'photo_real';
    }
  }

  String _fieldExpectedClassLabel(_FieldExpectedClass expectedClass) {
    switch (expectedClass) {
      case _FieldExpectedClass.board:
        return 'board';
      case _FieldExpectedClass.noBoard:
        return 'no_board';
    }
  }

  String _fieldBucketKey(
    _FieldExpectedDomain domain,
    _FieldExpectedClass expectedClass,
  ) {
    return '${_fieldExpectedDomainLabel(domain)}_${_fieldExpectedClassLabel(expectedClass)}';
  }

  Map<String, int> _fieldProtocolCounts() {
    final counts = <String, int>{
      'screen_board': 0,
      'screen_no_board': 0,
      'photo_real_board': 0,
      'photo_real_no_board': 0,
    };
    for (final entry in _fieldTestEntries) {
      final key = _fieldBucketKey(entry.expectedDomain, entry.expectedClass);
      counts[key] = (counts[key] ?? 0) + 1;
    }
    return counts;
  }

  bool _isFalsePositiveEntry(_FieldTestEntry entry) {
    return entry.expectedClass == _FieldExpectedClass.noBoard &&
        entry.boardDetected;
  }

  bool _isFalseNegativeEntry(_FieldTestEntry entry) {
    return entry.expectedClass == _FieldExpectedClass.board &&
        !entry.boardDetected;
  }

  String _fieldOutcomeLabel(_FieldTestEntry entry) {
    if (entry.expectedClass == _FieldExpectedClass.board) {
      return entry.boardDetected ? 'TP' : 'FN';
    }
    return entry.boardDetected ? 'FP' : 'TN';
  }

  String _fieldEntryShort(_FieldTestEntry entry) {
    final expected =
        '${_fieldExpectedDomainLabel(entry.expectedDomain)}/${_fieldExpectedClassLabel(entry.expectedClass)}';
    return '#${entry.index} '
        'exp=$expected '
        'src=${entry.acquisitionSource} '
        'chosen=${_captureDomainLabel(entry.chosenDomain)} '
        'gate_raw=${entry.gateDecisionRaw} '
        'final_raw=${entry.finalDecisionRaw} '
        'bypass_reason=${entry.bypassReason ?? "na"} '
        't_ms=${entry.tPrimaryMs}+${entry.tAltMs} '
        'decision=${entry.decision} '
        'board=${entry.boardDetected} '
        'outcome=${_fieldOutcomeLabel(entry)}';
  }

  void _recordCurrentScanToFieldProtocol() {
    final result = _scanResult;
    if (result == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Run a scan before logging protocol data.'),
        ),
      );
      return;
    }

    setState(() {
      _fieldTestEntries.add(
        _FieldTestEntry(
          index: _fieldTestEntries.length + 1,
          timestamp: DateTime.now(),
          expectedDomain: _fieldExpectedDomain,
          expectedClass: _fieldExpectedClass,
          tPrimaryMs: _lastScanPrimaryMs,
          tAltMs: _lastScanAltMs,
          acquisitionSource: _selectedImageSourceLabel(),
          chosenDomain: _selectedCaptureDomain,
          decision: _decisionLabelFromDetectorDebug(result.detectorDebug),
          gateDecisionRaw: _lastReportedGateDecisionRaw,
          finalDecisionRaw: _lastReportedFinalDecisionRaw,
          bypassReason: _lastBypassReason,
          boardDetected: result.boardDetected,
          postWarpGridness: _extractDetectorMetric(
            result.detectorDebug,
            'post_warp_gridness',
          ),
          minPostWarpGridness: _extractDetectorMetric(
            result.detectorDebug,
            'min_post_warp_gridness',
          ),
          rejectedPostWarpGridness: _isRejectedPostWarpGridness(
            result.detectorDebug,
          ),
          routingDebug: _routingDebugLabel,
          detectorDebug: result.detectorDebug,
          imagePath: _selectedImage?.path ?? '',
        ),
      );
    });
    _saveFieldProtocolState();

    if (!mounted) {
      return;
    }
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(
          'Logged protocol sample ${_fieldTestEntries.length}/$_fieldProtocolTotalTarget.',
        ),
      ),
    );
  }

  void _clearFieldProtocolEntries() {
    if (_fieldTestEntries.isEmpty) {
      return;
    }
    setState(() => _fieldTestEntries.clear());
    _saveFieldProtocolState();
    ScaffoldMessenger.of(
      context,
    ).showSnackBar(const SnackBar(content: Text('Protocol samples cleared.')));
  }

  Future<void> _saveFieldProtocolState() async {
    try {
      final prefs = await SharedPreferences.getInstance();
      final payload = <String, Object?>{
        'entries': _fieldTestEntries
            .map((entry) => entry.toJson())
            .toList(growable: false),
      };
      await prefs.setString(_fieldProtocolPrefsKey, jsonEncode(payload));
      await prefs.setString(
        _fieldProtocolDomainPrefsKey,
        _fieldExpectedDomain.name,
      );
      await prefs.setString(
        _fieldProtocolClassPrefsKey,
        _fieldExpectedClass.name,
      );
    } catch (e) {
      if (kDebugMode) {
        debugPrint('[scan][field-protocol] save_failed=$e');
      }
    }
  }

  Future<void> _loadFieldProtocolState() async {
    try {
      final prefs = await SharedPreferences.getInstance();
      final raw = prefs.getString(_fieldProtocolPrefsKey);
      final storedDomain = prefs.getString(_fieldProtocolDomainPrefsKey);
      final storedClass = prefs.getString(_fieldProtocolClassPrefsKey);

      final parsedEntries = <_FieldTestEntry>[];
      if (raw != null && raw.isNotEmpty) {
        final decoded = jsonDecode(raw);
        if (decoded is Map<String, dynamic>) {
          final entriesRaw = decoded['entries'];
          if (entriesRaw is List) {
            for (final item in entriesRaw) {
              if (item is Map) {
                final parsed = _FieldTestEntry.fromJson(
                  Map<String, dynamic>.from(item),
                );
                if (parsed != null) {
                  parsedEntries.add(parsed);
                }
              }
            }
          }
        }
      }

      if (!mounted) {
        return;
      }

      setState(() {
        _fieldTestEntries
          ..clear()
          ..addAll(parsedEntries);

        if (storedDomain == _FieldExpectedDomain.photoReal.name) {
          _fieldExpectedDomain = _FieldExpectedDomain.photoReal;
        } else {
          _fieldExpectedDomain = _FieldExpectedDomain.screen;
        }

        if (storedClass == _FieldExpectedClass.noBoard.name) {
          _fieldExpectedClass = _FieldExpectedClass.noBoard;
        } else {
          _fieldExpectedClass = _FieldExpectedClass.board;
        }
      });
    } catch (e) {
      if (kDebugMode) {
        debugPrint('[scan][field-protocol] load_failed=$e');
      }
    }
  }

  String _buildFieldProtocolReportText() {
    final counts = _fieldProtocolCounts();
    final falsePositives = _fieldTestEntries
        .where(_isFalsePositiveEntry)
        .length;
    final falseNegatives = _fieldTestEntries
        .where(_isFalseNegativeEntry)
        .length;
    final total = _fieldTestEntries.length;

    final lines = <String>[
      'Field protocol report',
      'revision=$_scanCoreRevision',
      'total=$total/$_fieldProtocolTotalTarget',
      'screen_board=${counts['screen_board'] ?? 0}/$_fieldProtocolBucketTarget',
      'screen_no_board=${counts['screen_no_board'] ?? 0}/$_fieldProtocolBucketTarget',
      'photo_real_board=${counts['photo_real_board'] ?? 0}/$_fieldProtocolBucketTarget',
      'photo_real_no_board=${counts['photo_real_no_board'] ?? 0}/$_fieldProtocolBucketTarget',
      'false_positive=$falsePositives',
      'false_negative=$falseNegatives',
      '',
      'entries:',
    ];

    for (final entry in _fieldTestEntries) {
      final expectedDomain = _fieldExpectedDomainLabel(entry.expectedDomain);
      final expectedClass = _fieldExpectedClassLabel(entry.expectedClass);
      final chosenDomain = _captureDomainLabel(entry.chosenDomain);
      final routeMatch =
          _normalizedFieldDomain(entry.chosenDomain) == entry.expectedDomain;
      lines.add(
        '#${entry.index} t=${entry.timestamp.toIso8601String()} '
        'expected=$expectedDomain/$expectedClass '
        'source=${entry.acquisitionSource} '
        'chosen=$chosenDomain route_match=$routeMatch '
        'gate_decision_raw=${entry.gateDecisionRaw} '
        'final_decision_raw=${entry.finalDecisionRaw} '
        'bypass_reason=${entry.bypassReason ?? "na"} '
        'post_warp_gridness=${entry.postWarpGridness?.toStringAsFixed(3) ?? "na"} '
        'min_post_warp_gridness=${entry.minPostWarpGridness?.toStringAsFixed(3) ?? "na"} '
        'rejected_post_warp_gridness=${entry.rejectedPostWarpGridness} '
        't_primary_ms=${entry.tPrimaryMs} t_alt_ms=${entry.tAltMs} '
        'decision=${entry.decision} boardDetected=${entry.boardDetected} '
        'outcome=${_fieldOutcomeLabel(entry)}',
      );
    }

    return lines.join('\n');
  }

  Future<void> _copyFieldProtocolReport() async {
    if (_fieldTestEntries.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('No protocol samples to copy.')),
      );
      return;
    }

    final text = _buildFieldProtocolReportText();
    await Clipboard.setData(ClipboardData(text: text));
    if (!mounted) {
      return;
    }
    ScaffoldMessenger.of(
      context,
    ).showSnackBar(const SnackBar(content: Text('Protocol report copied.')));
  }

  _FieldTestEntry? _latestFieldEntryWhere(
    bool Function(_FieldTestEntry entry) predicate,
  ) {
    for (var i = _fieldTestEntries.length - 1; i >= 0; i--) {
      final entry = _fieldTestEntries[i];
      if (predicate(entry)) {
        return entry;
      }
    }
    return null;
  }

  String _buildDetailedCaseLog(_FieldTestEntry entry, {required String kind}) {
    final expectedDomain = _fieldExpectedDomainLabel(entry.expectedDomain);
    final expectedClass = _fieldExpectedClassLabel(entry.expectedClass);
    return [
      'case_type=$kind',
      'index=${entry.index}',
      'timestamp=${entry.timestamp.toIso8601String()}',
      'expected_domain=$expectedDomain',
      'expected_class=$expectedClass',
      'acquisition_source=${entry.acquisitionSource}',
      'chosen_domain=${_captureDomainLabel(entry.chosenDomain)}',
      'gate_decision_raw=${entry.gateDecisionRaw}',
      'final_decision_raw=${entry.finalDecisionRaw}',
      'bypass_reason=${entry.bypassReason ?? "na"}',
      'post_warp_gridness=${entry.postWarpGridness?.toStringAsFixed(3) ?? "na"}',
      'min_post_warp_gridness=${entry.minPostWarpGridness?.toStringAsFixed(3) ?? "na"}',
      'rejected_post_warp_gridness=${entry.rejectedPostWarpGridness}',
      't_primary_ms=${entry.tPrimaryMs}',
      't_alt_ms=${entry.tAltMs}',
      'decision=${entry.decision}',
      'boardDetected=${entry.boardDetected}',
      'routing_debug=${entry.routingDebug}',
      'detector_debug=${entry.detectorDebug}',
      'image_path=${entry.imagePath}',
    ].join('\n');
  }

  Future<void> _copyLatestFalsePositiveLog() async {
    final entry = _latestFieldEntryWhere(_isFalsePositiveEntry);
    if (entry == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('No false positive logged yet.')),
      );
      return;
    }
    await Clipboard.setData(
      ClipboardData(text: _buildDetailedCaseLog(entry, kind: 'false_positive')),
    );
    if (!mounted) {
      return;
    }
    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(content: Text('Latest false positive log copied.')),
    );
  }

  Future<void> _copyLatestFalseNegativeLog() async {
    final entry = _latestFieldEntryWhere(_isFalseNegativeEntry);
    if (entry == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('No false negative logged yet.')),
      );
      return;
    }
    await Clipboard.setData(
      ClipboardData(text: _buildDetailedCaseLog(entry, kind: 'false_negative')),
    );
    if (!mounted) {
      return;
    }
    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(content: Text('Latest false negative log copied.')),
    );
  }

  bool _looksLikeScreenshotPath(String path) {
    final lower = path.toLowerCase();
    return lower.contains('screenshot') ||
        lower.contains('screen_shot') ||
        lower.contains('screen-shot') ||
        lower.contains('/screenshots/') ||
        lower.contains('\\screenshots\\') ||
        lower.contains('capture');
  }

  bool _hasExifMetadata(Uint8List bytes) {
    return _hasJpegExif(bytes) || _hasPngExif(bytes);
  }

  bool _hasJpegExif(Uint8List bytes) {
    if (bytes.length < 4) {
      return false;
    }
    if (bytes[0] != 0xFF || bytes[1] != 0xD8) {
      return false;
    }

    var offset = 2;
    while (offset + 4 <= bytes.length) {
      if (bytes[offset] != 0xFF) {
        offset += 1;
        continue;
      }

      final marker = bytes[offset + 1];
      if (marker == 0xD9 || marker == 0xDA) {
        break;
      }
      final segmentLength = (bytes[offset + 2] << 8) | bytes[offset + 3];
      if (segmentLength < 2) {
        break;
      }

      final nextOffset = offset + 2 + segmentLength;
      if (nextOffset > bytes.length) {
        break;
      }

      if (marker == 0xE1) {
        final header = offset + 4;
        if (header + 6 <= bytes.length &&
            bytes[header] == 0x45 &&
            bytes[header + 1] == 0x78 &&
            bytes[header + 2] == 0x69 &&
            bytes[header + 3] == 0x66 &&
            bytes[header + 4] == 0x00 &&
            bytes[header + 5] == 0x00) {
          return true;
        }
      }

      offset = nextOffset;
    }

    return false;
  }

  bool _hasPngExif(Uint8List bytes) {
    if (bytes.length < 8) {
      return false;
    }
    const signature = <int>[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
    for (var i = 0; i < signature.length; i++) {
      if (bytes[i] != signature[i]) {
        return false;
      }
    }

    var offset = 8;
    while (offset + 12 <= bytes.length) {
      final length =
          (bytes[offset] << 24) |
          (bytes[offset + 1] << 16) |
          (bytes[offset + 2] << 8) |
          bytes[offset + 3];
      if (length < 0) {
        return false;
      }

      final nextOffset = offset + 12 + length;
      if (nextOffset > bytes.length) {
        return false;
      }

      final isExifChunk =
          bytes[offset + 4] == 0x65 &&
          bytes[offset + 5] == 0x58 &&
          bytes[offset + 6] == 0x49 &&
          bytes[offset + 7] == 0x66;
      if (isExifChunk) {
        return true;
      }

      offset = nextOffset;
    }

    return false;
  }

  Future<void> _editSquare(String square) async {
    final position = _editablePosition;
    if (position == null) {
      return;
    }

    final choice = await PieceChooserSheet.show(
      context,
      currentPiece: position.pieceAt(square),
    );
    if (choice == null || !mounted) {
      return;
    }

    setState(() {
      if (choice.clear) {
        _editablePosition = position.removePiece(square);
      } else if (choice.piece != null) {
        _editablePosition = position.setPiece(square, choice.piece!);
      }
      _refreshFenAndValidation();
    });
  }

  void _refreshFenAndValidation() {
    final position = _editablePosition;
    if (position == null) {
      _validation = const PositionValidationResult();
      _finalFen = null;
      return;
    }
    _validation = _validator.validate(position);
    _finalFen = _fenBuilder.build(position);
  }

  void _openAnalysis() {
    final fen = _finalFen;
    if (fen == null) {
      return;
    }
    Navigator.of(context).pop();
    widget.onAnalyzeFen(fen);
  }

  Future<void> _runDatasetValidation() async {
    if (_isRunningDatasetValidation) {
      return;
    }
    final useFast = _useFastDatasetValidation;
    final validationUseCase = useFast
        ? _datasetValidationUseCaseFast
        : _datasetValidationUseCase;
    setState(() {
      _isRunningDatasetValidation = true;
      _datasetProgressLabel = useFast
          ? 'Validation rapide 0/...'
          : 'Validation complete 0/...';
      _errorMessage = null;
    });

    try {
      final report = await validationUseCase.run(
        datasetAssetPath: 'assets/scan_samples/scan_test_cases.json',
        includePayloads: false,
        onProgress: (done, total, caseId) {
          if (!mounted) {
            return;
          }
          if (done != total && done != 1 && done % 2 != 0) {
            return;
          }
          setState(() {
            _datasetProgressLabel = useFast
                ? 'Validation rapide $done/$total... ($caseId)'
                : 'Validation complete $done/$total... ($caseId)';
          });
        },
      );
      if (!mounted) {
        return;
      }

      final selected = report.evaluations.isNotEmpty
          ? report.evaluations.first
          : null;
      setState(() {
        _datasetReport = report;
        _selectedEvaluation = selected;
      });
      if (selected != null && !useFast) {
        await _hydrateSelectedEvaluation(selected.testCase.id);
      } else if (mounted) {
        setState(() => _datasetProgressLabel = null);
      }
    } catch (e) {
      if (!mounted) {
        return;
      }
      setState(() => _errorMessage = 'Dataset validation failed: $e');
    } finally {
      if (mounted) {
        setState(() {
          _isRunningDatasetValidation = false;
          if (_isLoadingSelectedEvaluation == false) {
            _datasetProgressLabel = null;
          }
        });
      }
    }
  }

  Future<void> _hydrateSelectedEvaluation(String caseId) async {
    final report = _datasetReport;
    if (report == null ||
        report.evaluations.isEmpty ||
        _isLoadingSelectedEvaluation) {
      return;
    }
    final summary = report.evaluations.firstWhere(
      (e) => e.testCase.id == caseId,
      orElse: () => report.evaluations.first,
    );

    setState(() {
      _isLoadingSelectedEvaluation = true;
      _datasetProgressLabel = 'Loading case ${summary.testCase.id}...';
      _selectedEvaluation = summary;
    });
    _applyEvaluationToDebugView(summary);

    try {
      final detailed = await _datasetValidationUseCase.runSingleCase(
        testCase: summary.testCase,
        includePayload: true,
      );
      if (!mounted) {
        return;
      }

      final refreshedReport = _datasetReport;
      if (refreshedReport == null) {
        return;
      }

      final updatedEvaluations = refreshedReport.evaluations
          .map((e) => e.testCase.id == detailed.testCase.id ? detailed : e)
          .toList(growable: false);
      setState(() {
        _datasetReport = ScanDatasetValidationReport(
          dataset: refreshedReport.dataset,
          evaluations: updatedEvaluations,
        );
        _selectedEvaluation = detailed;
      });
      _applyEvaluationToDebugView(detailed);
    } catch (e) {
      if (!mounted) {
        return;
      }
      setState(() => _errorMessage = 'Case validation failed: $e');
    } finally {
      if (mounted) {
        setState(() {
          _isLoadingSelectedEvaluation = false;
          _datasetProgressLabel = null;
        });
      }
    }
  }

  void _applyEvaluationToDebugView(ScanCaseEvaluation? evaluation) {
    if (evaluation == null || !mounted) {
      return;
    }
    final image = evaluation.sourceImage;
    final inferredHasExif = image == null
        ? false
        : _hasExifMetadata(image.bytes);
    final inferredLooksScreenshot = image == null
        ? false
        : _looksLikeScreenshotPath(image.path);

    setState(() {
      _selectedImage = image;
      _selectedImageSource = null;
      _selectedCaptureDomain = _resolveCaptureDomain(
        hasExif: inferredHasExif,
        looksScreenshot: inferredLooksScreenshot,
      );
      _captureDomainOverride = null;
      _routingDebugLabel =
          'mode=auto hint=${_captureDomainLabel(_selectedCaptureDomain)} dataset_eval';
      _selectedImageHasExif = inferredHasExif;
      _selectedImageLooksScreenshot = inferredLooksScreenshot;
      _scanResult = evaluation.result;
      _lastScanPrimaryMs = 0;
      _lastScanAltMs = 0;
      _editablePosition = evaluation.result?.detectedPosition;
      _selectedImageSize = null;
      _manualCorners = const <BoardCorner>[];
      _refreshFenAndValidation();
    });
    if (image != null) {
      _resolveSelectedImageSize(image.bytes);
    }
  }

  String _formatCorners(ScanPipelineResult? result) {
    if (result == null || result.geometry.corners.isEmpty) {
      return '-';
    }
    return result.geometry.corners
        .map((c) => '(${c.x.toStringAsFixed(1)}, ${c.y.toStringAsFixed(1)})')
        .join('  ');
  }

  String _cornerQualityLabel(double meanPercent, double maxPercent) {
    if (meanPercent < 4.0 && maxPercent < 8.0) {
      return 'excellent';
    }
    if (meanPercent < 8.0 && maxPercent < 15.0) {
      return 'acceptable';
    }
    return 'to_fix';
  }

  Future<void> _copyValidationTextReport() async {
    final report = _datasetReport;
    if (report == null || report.evaluations.isEmpty) {
      return;
    }

    final lines = <String>[];
    lines.add('Revision: $_scanCoreRevision');
    lines.add('');
    for (int i = 0; i < report.evaluations.length; i++) {
      final evaluation = report.evaluations[i];
      lines.add('${evaluation.testCase.id} ${evaluation.statusLabel}');
      lines.add('Image: ${evaluation.testCase.image}');
      lines.add('Compared fields: ${evaluation.comparisons.length}');

      final metrics = evaluation.cornerErrorMetrics;
      if (metrics != null) {
        final level = _cornerQualityLabel(
          metrics.meanPercent,
          metrics.maxPercent,
        );
        lines.add(
          'Corner error: mean ${metrics.meanPx.toStringAsFixed(1)} px '
          '(${metrics.meanPercent.toStringAsFixed(2)}%), '
          'max ${metrics.maxPx.toStringAsFixed(1)} px '
          '(${metrics.maxPercent.toStringAsFixed(2)}%) [$level]',
        );
      }

      final result = evaluation.result;
      if (result != null) {
        lines.add('Detector debug: ${result.detectorDebug}');
      }

      if (evaluation.error != null) {
        lines.add('Error: ${evaluation.error}');
      }

      for (final comparison in evaluation.comparisons) {
        lines.add(
          '${comparison.field}: expected=${comparison.expected} '
          'detected=${comparison.detected} ${comparison.matched ? "OK" : "KO"}',
        );
      }

      if (i < report.evaluations.length - 1) {
        lines.add('');
      }
    }

    final reportText = lines.join('\n');
    await Clipboard.setData(ClipboardData(text: reportText));
    if (!mounted) {
      return;
    }
    ScaffoldMessenger.of(
      context,
    ).showSnackBar(const SnackBar(content: Text('Validation report copied.')));
  }

  Future<void> _autoFillExpectedForSelectedCase() async {
    final report = _datasetReport;
    final selected = _selectedEvaluation;
    final result = _scanResult;
    if (report == null || selected == null || result == null) {
      return;
    }

    final updatedExpected = ScanValidationExpected(
      boardDetected: result.boardDetected,
      corners: result.geometry.corners.isEmpty ? null : result.geometry.corners,
      warpOk: result.warpOk,
      orientationOk: result.orientationOk,
      whiteAtBottom: selected.testCase.expected.whiteAtBottom,
      fen: _finalFen,
    );

    final updatedCases = report.dataset.cases
        .map((scanCase) {
          if (scanCase.id != selected.testCase.id) {
            return scanCase;
          }
          return ScanValidationCase(
            id: scanCase.id,
            image: scanCase.image,
            type: scanCase.type,
            difficulty: scanCase.difficulty,
            expected: updatedExpected,
            notes: scanCase.notes,
          );
        })
        .toList(growable: false);

    final updatedDataset = ScanValidationDataset(
      version: report.dataset.version,
      boardSize: report.dataset.boardSize,
      pointOrder: report.dataset.pointOrder,
      warpTargetSize: report.dataset.warpTargetSize,
      cases: updatedCases,
    );

    final updatedCaseById = <String, ScanValidationCase>{
      for (final c in updatedDataset.cases) c.id: c,
    };
    final updatedEvaluations = report.evaluations
        .map((evaluation) {
          return ScanCaseEvaluation(
            testCase:
                updatedCaseById[evaluation.testCase.id] ?? evaluation.testCase,
            comparisons: evaluation.comparisons,
            sourceImage: evaluation.sourceImage,
            result: evaluation.result,
            cornerErrorMetrics: evaluation.cornerErrorMetrics,
            error: evaluation.error,
          );
        })
        .toList(growable: false);
    final updatedSelected = updatedEvaluations.firstWhere(
      (e) => e.testCase.id == selected.testCase.id,
      orElse: () => selected,
    );

    final formattedJson = const JsonEncoder.withIndent(
      '  ',
    ).convert(updatedDataset.toJson());
    await Clipboard.setData(ClipboardData(text: formattedJson));

    if (!mounted) {
      return;
    }

    setState(() {
      _datasetReport = ScanDatasetValidationReport(
        dataset: updatedDataset,
        evaluations: updatedEvaluations,
      );
      _selectedEvaluation = updatedSelected;
    });

    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(
          'Expected auto-filled for ${selected.testCase.id}. JSON copied.',
        ),
      ),
    );
  }

  Future<void> _autoFillExpectedForAllCases() async {
    final report = _datasetReport;
    if (report == null || _isAutoFillingAllExpected) {
      return;
    }

    final total = report.dataset.cases.length;
    if (total == 0) {
      return;
    }

    setState(() {
      _isAutoFillingAllExpected = true;
      _errorMessage = null;
      _datasetProgressLabel = 'Auto-fill all 0/$total...';
    });

    try {
      final updatedCases = <ScanValidationCase>[];
      final compactEvaluations = <ScanCaseEvaluation>[];
      int updatedCount = 0;
      int keptCount = 0;

      for (int i = 0; i < report.dataset.cases.length; i++) {
        final scanCase = report.dataset.cases[i];
        if (!mounted) {
          return;
        }
        setState(() {
          _datasetProgressLabel =
              'Auto-fill all ${i + 1}/$total... (${scanCase.id})';
        });

        final evaluation = await _datasetValidationUseCase.runSingleCase(
          testCase: scanCase,
          includePayload: true,
        );
        compactEvaluations.add(
          ScanCaseEvaluation(
            testCase: evaluation.testCase,
            comparisons: evaluation.comparisons,
            cornerErrorMetrics: evaluation.cornerErrorMetrics,
            error: evaluation.error,
          ),
        );

        final result = evaluation.result;
        if (evaluation.error != null || result == null) {
          keptCount += 1;
          updatedCases.add(scanCase);
          continue;
        }

        updatedCount += 1;
        updatedCases.add(
          ScanValidationCase(
            id: scanCase.id,
            image: scanCase.image,
            type: scanCase.type,
            difficulty: scanCase.difficulty,
            expected: ScanValidationExpected(
              boardDetected: result.boardDetected,
              corners: result.geometry.corners.isEmpty
                  ? null
                  : result.geometry.corners,
              warpOk: result.warpOk,
              orientationOk: result.orientationOk,
              whiteAtBottom: scanCase.expected.whiteAtBottom,
              fen: result.detectedFen,
            ),
            notes: scanCase.notes,
          ),
        );

        await Future<void>.delayed(const Duration(milliseconds: 1));
      }

      final updatedDataset = ScanValidationDataset(
        version: report.dataset.version,
        boardSize: report.dataset.boardSize,
        pointOrder: report.dataset.pointOrder,
        warpTargetSize: report.dataset.warpTargetSize,
        cases: updatedCases,
      );
      final formattedJson = const JsonEncoder.withIndent(
        '  ',
      ).convert(updatedDataset.toJson());
      await Clipboard.setData(ClipboardData(text: formattedJson));

      if (!mounted) {
        return;
      }

      final updatedReport = ScanDatasetValidationReport(
        dataset: updatedDataset,
        evaluations: compactEvaluations,
      );
      final selectedId = _selectedEvaluation?.testCase.id;
      ScanCaseEvaluation? nextSelected;
      if (selectedId != null) {
        for (final e in compactEvaluations) {
          if (e.testCase.id == selectedId) {
            nextSelected = e;
            break;
          }
        }
      }
      nextSelected ??= compactEvaluations.isNotEmpty
          ? compactEvaluations.first
          : null;

      setState(() {
        _datasetReport = updatedReport;
        _selectedEvaluation = nextSelected;
      });

      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text(
            'Auto-fill all done: $updatedCount/$total updated, $keptCount kept. JSON copied.',
          ),
        ),
      );
    } catch (e) {
      if (!mounted) {
        return;
      }
      setState(() => _errorMessage = 'Auto-fill all failed: $e');
    } finally {
      if (mounted) {
        setState(() {
          _isAutoFillingAllExpected = false;
          if (!_isRunningDatasetValidation && !_isLoadingSelectedEvaluation) {
            _datasetProgressLabel = null;
          }
        });
      }
    }
  }

  Future<void> _resolveSelectedImageSize(Uint8List bytes) async {
    final token = Object();
    _selectedImageSizeToken = token;

    try {
      final codec = await ui.instantiateImageCodec(bytes);
      final frame = await codec.getNextFrame();
      if (!mounted || _selectedImageSizeToken != token) {
        return;
      }
      setState(() {
        _selectedImageSize = Size(
          frame.image.width.toDouble(),
          frame.image.height.toDouble(),
        );
      });
    } catch (_) {
      if (!mounted || _selectedImageSizeToken != token) {
        return;
      }
      setState(() => _selectedImageSize = null);
    }
  }

  void _onManualCornerTapped(Offset imagePoint) {
    if (_manualCorners.length >= 4) {
      return;
    }
    setState(() {
      _manualCorners = <BoardCorner>[
        ..._manualCorners,
        BoardCorner(x: imagePoint.dx, y: imagePoint.dy),
      ];
    });
  }

  void _undoManualCorner() {
    if (_manualCorners.isEmpty) {
      return;
    }
    setState(() {
      _manualCorners = _manualCorners
          .sublist(0, _manualCorners.length - 1)
          .toList(growable: false);
    });
  }

  void _clearManualCorners() {
    if (_manualCorners.isEmpty) {
      return;
    }
    setState(() => _manualCorners = const <BoardCorner>[]);
  }

  String _manualCornersJson() {
    final list = _manualCorners
        .map((c) => <String, int>{'x': c.x.round(), 'y': c.y.round()})
        .toList(growable: false);
    return const JsonEncoder.withIndent('  ').convert(list);
  }

  Future<void> _copyManualCornersJson() async {
    if (_manualCorners.length != 4) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Tap 4 corners before copying JSON.')),
      );
      return;
    }

    await Clipboard.setData(ClipboardData(text: _manualCornersJson()));
    if (!mounted) {
      return;
    }
    ScaffoldMessenger.of(
      context,
    ).showSnackBar(const SnackBar(content: Text('Corners JSON copied.')));
  }

  @override
  Widget build(BuildContext context) {
    final imageBytes = _selectedImage?.bytes;
    final rectifiedBytes = _scanResult?.rectifiedBoard.bytes;
    final protocolCounts = _fieldProtocolCounts();
    final protocolFalsePositives = _fieldTestEntries
        .where(_isFalsePositiveEntry)
        .length;
    final protocolFalseNegatives = _fieldTestEntries
        .where(_isFalseNegativeEntry)
        .length;
    final protocolLatestLines = _fieldTestEntries.reversed
        .take(5)
        .map(_fieldEntryShort)
        .toList(growable: false);

    return Scaffold(
      appBar: AppBar(title: const Text('Scan position')),
      backgroundColor: const Color(0xFF282725),
      body: SafeArea(
        child: Stack(
          children: [
            ListView(
              padding: const EdgeInsets.fromLTRB(16, 8, 16, 24),
              children: [
                _SectionCard(
                  title: 'Capture image',
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Wrap(
                        spacing: 10,
                        runSpacing: 10,
                        children: [
                          FilledButton.icon(
                            onPressed: _isScanning
                                ? null
                                : () => _pickImage(ImageSource.camera),
                            icon: const Icon(Icons.photo_camera_outlined),
                            label: const Text('Camera'),
                          ),
                          FilledButton.tonalIcon(
                            onPressed: _isScanning
                                ? null
                                : () => _pickImage(ImageSource.gallery),
                            icon: const Icon(Icons.photo_library_outlined),
                            label: const Text('Gallery'),
                          ),
                          FilledButton.icon(
                            onPressed: imageBytes == null ? null : _runScan,
                            icon: _isScanning
                                ? const SizedBox.square(
                                    dimension: 16,
                                    child: CircularProgressIndicator(
                                      strokeWidth: 2,
                                    ),
                                  )
                                : const Icon(Icons.center_focus_strong),
                            label: Text(
                              _isScanning ? 'Scanning...' : 'Scanner',
                            ),
                          ),
                          OutlinedButton.icon(
                            onPressed:
                                (_isScanning ||
                                    _isRunningDatasetValidation ||
                                    _isAutoFillingAllExpected)
                                ? null
                                : _runDatasetValidation,
                            icon: _isRunningDatasetValidation
                                ? const SizedBox.square(
                                    dimension: 16,
                                    child: CircularProgressIndicator(
                                      strokeWidth: 2,
                                    ),
                                  )
                                : const Icon(Icons.science_outlined),
                            label: const Text('Validate dataset'),
                          ),
                        ],
                      ),
                      const SizedBox(height: 8),
                      Row(
                        children: [
                          Icon(
                            Icons.speed_outlined,
                            size: 16,
                            color: Colors.white.withValues(alpha: 0.75),
                          ),
                          const SizedBox(width: 6),
                          Expanded(
                            child: Text(
                              'Validation rapide du dataset',
                              style: TextStyle(
                                fontSize: 12,
                                color: Colors.white.withValues(alpha: 0.75),
                              ),
                            ),
                          ),
                          Switch.adaptive(
                            value: _useFastDatasetValidation,
                            onChanged:
                                (_isScanning ||
                                    _isRunningDatasetValidation ||
                                    _isLoadingSelectedEvaluation ||
                                    _isAutoFillingAllExpected)
                                ? null
                                : (v) => setState(
                                    () => _useFastDatasetValidation = v,
                                  ),
                          ),
                        ],
                      ),
                      if (_datasetProgressLabel != null) ...[
                        const SizedBox(height: 8),
                        Text(
                          _datasetProgressLabel!,
                          style: TextStyle(
                            fontSize: 12,
                            color: Colors.white.withValues(alpha: 0.75),
                          ),
                        ),
                      ],
                      if (_selectedImage != null) ...[
                        const SizedBox(height: 8),
                        Text(
                          'Routing: ${_captureDomainLabel(_selectedCaptureDomain)} '
                          '(mode=${_routingModeLabel()}, '
                          'source=${_selectedImageSourceLabel()}, '
                          'screenshot_hint=${_selectedImageLooksScreenshot ? "yes" : "no"}, '
                          'exif=${_selectedImageHasExif ? "present" : "absent"})',
                          style: TextStyle(
                            fontSize: 12,
                            color: Colors.white.withValues(alpha: 0.75),
                          ),
                        ),
                        const SizedBox(height: 2),
                        Text(
                          _routingDebugLabel,
                          style: TextStyle(
                            fontSize: 11,
                            color: Colors.white.withValues(alpha: 0.65),
                          ),
                        ),
                        const SizedBox(height: 4),
                        Text(
                          'Gate counters: '
                          'reject=${_gateDecisionCounters['strong_reject'] ?? 0} '
                          'gray=${_gateDecisionCounters['gray'] ?? 0} '
                          'accept=${_gateDecisionCounters['strong_accept'] ?? 0}',
                          style: TextStyle(
                            fontSize: 11,
                            color: Colors.white.withValues(alpha: 0.70),
                          ),
                        ),
                        const SizedBox(height: 2),
                        Text(
                          _autoRoutingRetrySummary(),
                          style: TextStyle(
                            fontSize: 11,
                            color: Colors.white.withValues(alpha: 0.65),
                          ),
                        ),
                      ],
                      if (_selectedImage != null) ...[
                        const SizedBox(height: 8),
                        Text(
                          'Routing mode (AUTO par contenu ou override)',
                          style: TextStyle(
                            fontSize: 12,
                            color: Colors.white.withValues(alpha: 0.75),
                          ),
                        ),
                        const SizedBox(height: 6),
                        Wrap(
                          spacing: 8,
                          runSpacing: 8,
                          children: [
                            ChoiceChip(
                              label: const Text('AUTO'),
                              selected: _captureDomainOverride == null,
                              onSelected: (selected) {
                                if (!selected) {
                                  return;
                                }
                                _setCaptureDomainOverride(null);
                              },
                            ),
                            ChoiceChip(
                              label: const Text('Photo reelle'),
                              selected:
                                  _captureDomainOverride ==
                                  _ScanCaptureDomain.photoReal,
                              onSelected: (selected) {
                                if (!selected) {
                                  return;
                                }
                                _setCaptureDomainOverride(
                                  _ScanCaptureDomain.photoReal,
                                );
                              },
                            ),
                            ChoiceChip(
                              label: const Text('Photo imprimee (livre)'),
                              selected:
                                  _captureDomainOverride ==
                                  _ScanCaptureDomain.photoPrint,
                              onSelected: (selected) {
                                if (!selected) {
                                  return;
                                }
                                _setCaptureDomainOverride(
                                  _ScanCaptureDomain.photoPrint,
                                );
                              },
                            ),
                            ChoiceChip(
                              label: const Text('Screenshot / ecran'),
                              selected:
                                  _captureDomainOverride ==
                                  _ScanCaptureDomain.screen,
                              onSelected: (selected) {
                                if (!selected) {
                                  return;
                                }
                                _setCaptureDomainOverride(
                                  _ScanCaptureDomain.screen,
                                );
                              },
                            ),
                          ],
                        ),
                        const SizedBox(height: 4),
                        Text(
                          'EXIF et chemin screenshot sont des hints seulement.',
                          style: TextStyle(
                            fontSize: 11,
                            color: Colors.white.withValues(alpha: 0.65),
                          ),
                        ),
                      ],
                    ],
                  ),
                ),
                const SizedBox(height: 12),
                _SectionCard(
                  title: 'Protocole terrain (40 tests)',
                  subtitle:
                      'Log terrain pour ChatGPT: domaine choisi, decision, boardDetected.',
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        'Progression: ${_fieldTestEntries.length}/$_fieldProtocolTotalTarget',
                        style: TextStyle(
                          fontSize: 12,
                          color: Colors.white.withValues(alpha: 0.80),
                        ),
                      ),
                      const SizedBox(height: 4),
                      Text(
                        'screen_board ${protocolCounts['screen_board'] ?? 0}/$_fieldProtocolBucketTarget | '
                        'screen_no_board ${protocolCounts['screen_no_board'] ?? 0}/$_fieldProtocolBucketTarget',
                        style: TextStyle(
                          fontSize: 11,
                          color: Colors.white.withValues(alpha: 0.70),
                        ),
                      ),
                      Text(
                        'photo_real_board ${protocolCounts['photo_real_board'] ?? 0}/$_fieldProtocolBucketTarget | '
                        'photo_real_no_board ${protocolCounts['photo_real_no_board'] ?? 0}/$_fieldProtocolBucketTarget',
                        style: TextStyle(
                          fontSize: 11,
                          color: Colors.white.withValues(alpha: 0.70),
                        ),
                      ),
                      const SizedBox(height: 8),
                      Text(
                        'Attendu: domaine',
                        style: TextStyle(
                          fontSize: 12,
                          color: Colors.white.withValues(alpha: 0.75),
                        ),
                      ),
                      const SizedBox(height: 6),
                      Wrap(
                        spacing: 8,
                        runSpacing: 8,
                        children: [
                          ChoiceChip(
                            label: const Text('screen'),
                            selected:
                                _fieldExpectedDomain ==
                                _FieldExpectedDomain.screen,
                            onSelected: (selected) {
                              if (!selected) {
                                return;
                              }
                              setState(
                                () => _fieldExpectedDomain =
                                    _FieldExpectedDomain.screen,
                              );
                            },
                          ),
                          ChoiceChip(
                            label: const Text('photo_real'),
                            selected:
                                _fieldExpectedDomain ==
                                _FieldExpectedDomain.photoReal,
                            onSelected: (selected) {
                              if (!selected) {
                                return;
                              }
                              setState(
                                () => _fieldExpectedDomain =
                                    _FieldExpectedDomain.photoReal,
                              );
                            },
                          ),
                        ],
                      ),
                      const SizedBox(height: 8),
                      Text(
                        'Attendu: classe',
                        style: TextStyle(
                          fontSize: 12,
                          color: Colors.white.withValues(alpha: 0.75),
                        ),
                      ),
                      const SizedBox(height: 6),
                      Wrap(
                        spacing: 8,
                        runSpacing: 8,
                        children: [
                          ChoiceChip(
                            label: const Text('board'),
                            selected:
                                _fieldExpectedClass ==
                                _FieldExpectedClass.board,
                            onSelected: (selected) {
                              if (!selected) {
                                return;
                              }
                              setState(
                                () => _fieldExpectedClass =
                                    _FieldExpectedClass.board,
                              );
                            },
                          ),
                          ChoiceChip(
                            label: const Text('no_board'),
                            selected:
                                _fieldExpectedClass ==
                                _FieldExpectedClass.noBoard,
                            onSelected: (selected) {
                              if (!selected) {
                                return;
                              }
                              setState(
                                () => _fieldExpectedClass =
                                    _FieldExpectedClass.noBoard,
                              );
                            },
                          ),
                        ],
                      ),
                      const SizedBox(height: 10),
                      Wrap(
                        spacing: 8,
                        runSpacing: 8,
                        children: [
                          FilledButton.icon(
                            onPressed: _scanResult == null
                                ? null
                                : _recordCurrentScanToFieldProtocol,
                            icon: const Icon(
                              Icons.playlist_add_check_circle_outlined,
                            ),
                            label: const Text('Log scan courant'),
                          ),
                          OutlinedButton.icon(
                            onPressed: _fieldTestEntries.isEmpty
                                ? null
                                : _copyFieldProtocolReport,
                            icon: const Icon(Icons.copy_all_outlined),
                            label: const Text('Copier report'),
                          ),
                          OutlinedButton.icon(
                            onPressed: _fieldTestEntries.isEmpty
                                ? null
                                : _copyLatestFalsePositiveLog,
                            icon: const Icon(Icons.warning_amber_outlined),
                            label: const Text('Copier FP'),
                          ),
                          OutlinedButton.icon(
                            onPressed: _fieldTestEntries.isEmpty
                                ? null
                                : _copyLatestFalseNegativeLog,
                            icon: const Icon(Icons.error_outline),
                            label: const Text('Copier FN'),
                          ),
                          OutlinedButton.icon(
                            onPressed: _fieldTestEntries.isEmpty
                                ? null
                                : _clearFieldProtocolEntries,
                            icon: const Icon(Icons.delete_outline),
                            label: const Text('Vider'),
                          ),
                        ],
                      ),
                      const SizedBox(height: 8),
                      Text(
                        'FP=$protocolFalsePositives | FN=$protocolFalseNegatives',
                        style: TextStyle(
                          fontSize: 11,
                          color: Colors.white.withValues(alpha: 0.70),
                        ),
                      ),
                      if (protocolLatestLines.isNotEmpty) ...[
                        const SizedBox(height: 8),
                        SelectableText(
                          protocolLatestLines.join('\n'),
                          style: TextStyle(
                            fontSize: 11,
                            color: Colors.white.withValues(alpha: 0.8),
                          ),
                        ),
                      ],
                    ],
                  ),
                ),
                if (_datasetReport != null) ...[
                  const SizedBox(height: 12),
                  _SectionCard(
                    title: 'Dataset validation',
                    subtitle:
                        'Excellent ${_datasetReport!.excellentPassedCount}/${_datasetReport!.total} - '
                        'Quality ${_datasetReport!.qualityPassedCount}/${_datasetReport!.total} - '
                        'Functional ${_datasetReport!.functionalPassedCount}/${_datasetReport!.total}',
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        DropdownButton<String>(
                          isExpanded: true,
                          value: _selectedEvaluation?.testCase.id,
                          items: _datasetReport!.evaluations
                              .map(
                                (e) => DropdownMenuItem<String>(
                                  value: e.testCase.id,
                                  child: Text(
                                    '${e.testCase.id}  ${e.statusLabel}',
                                  ),
                                ),
                              )
                              .toList(growable: false),
                          onChanged: (id) {
                            if (id == null) {
                              return;
                            }
                            _hydrateSelectedEvaluation(id);
                          },
                        ),
                        const SizedBox(height: 8),
                        OutlinedButton.icon(
                          onPressed:
                              (_selectedEvaluation == null ||
                                  _isLoadingSelectedEvaluation ||
                                  _isRunningDatasetValidation ||
                                  _isAutoFillingAllExpected)
                              ? null
                              : () => _hydrateSelectedEvaluation(
                                  _selectedEvaluation!.testCase.id,
                                ),
                          icon: _isLoadingSelectedEvaluation
                              ? const SizedBox.square(
                                  dimension: 14,
                                  child: CircularProgressIndicator(
                                    strokeWidth: 2,
                                  ),
                                )
                              : const Icon(Icons.replay_outlined),
                          label: const Text('Validate selected case'),
                        ),
                        const SizedBox(height: 8),
                        OutlinedButton.icon(
                          onPressed:
                              (_isRunningDatasetValidation ||
                                  _isLoadingSelectedEvaluation ||
                                  _isAutoFillingAllExpected)
                              ? null
                              : _autoFillExpectedForAllCases,
                          icon: _isAutoFillingAllExpected
                              ? const SizedBox.square(
                                  dimension: 14,
                                  child: CircularProgressIndicator(
                                    strokeWidth: 2,
                                  ),
                                )
                              : const Icon(Icons.copy_all_outlined),
                          label: const Text('Auto-fill all + copy JSON'),
                        ),
                        const SizedBox(height: 8),
                        OutlinedButton.icon(
                          onPressed:
                              (_isRunningDatasetValidation ||
                                  _isLoadingSelectedEvaluation)
                              ? null
                              : _copyValidationTextReport,
                          icon: const Icon(Icons.text_snippet_outlined),
                          label: const Text('Copy validation text report'),
                        ),
                        if (_selectedEvaluation != null) ...[
                          Text('Status: ${_selectedEvaluation!.statusLabel}'),
                          Text('Image: ${_selectedEvaluation!.testCase.image}'),
                          if (_selectedEvaluation!.result != null)
                            Text(
                              'Detector debug: ${_selectedEvaluation!.result!.detectorDebug}',
                              style: TextStyle(
                                fontSize: 12,
                                color: Colors.white.withValues(alpha: 0.80),
                              ),
                            ),
                          Text(
                            'Compared fields: ${_selectedEvaluation!.comparisons.length}',
                          ),
                          if (_selectedEvaluation!.cornerErrorMetrics !=
                              null) ...[
                            Builder(
                              builder: (context) {
                                final metrics =
                                    _selectedEvaluation!.cornerErrorMetrics!;
                                final meanPercent = metrics.meanPercent;
                                final maxPercent = metrics.maxPercent;
                                Color color;
                                String level;
                                if (meanPercent < 4.0 && maxPercent < 8.0) {
                                  color = Colors.lightGreenAccent;
                                  level = 'excellent';
                                } else if (meanPercent < 8.0 &&
                                    maxPercent < 15.0) {
                                  color = Colors.amberAccent;
                                  level = 'acceptable';
                                } else {
                                  color = Colors.redAccent;
                                  level = 'to_fix';
                                }

                                return Text(
                                  'Corner error: mean ${metrics.meanPx.toStringAsFixed(1)} px '
                                  '(${meanPercent.toStringAsFixed(2)}%), '
                                  'max ${metrics.maxPx.toStringAsFixed(1)} px '
                                  '(${metrics.maxPercent.toStringAsFixed(2)}%) '
                                  '[$level]',
                                  style: TextStyle(fontSize: 12, color: color),
                                );
                              },
                            ),
                          ],
                          if (_selectedEvaluation!.error != null)
                            Text(
                              'Error: ${_selectedEvaluation!.error}',
                              style: const TextStyle(color: Colors.redAccent),
                            ),
                          ..._selectedEvaluation!.comparisons.map(
                            (c) => Text(
                              '${c.field}: expected=${c.expected} detected=${c.detected} '
                              '${c.matched ? "OK" : "KO"}',
                              style: TextStyle(
                                color: c.matched
                                    ? Colors.white.withValues(alpha: 0.9)
                                    : Colors.redAccent,
                                fontSize: 12,
                              ),
                            ),
                          ),
                          const SizedBox(height: 8),
                          OutlinedButton.icon(
                            onPressed:
                                (_selectedEvaluation!.result == null ||
                                    _isAutoFillingAllExpected)
                                ? null
                                : _autoFillExpectedForSelectedCase,
                            icon: const Icon(Icons.auto_fix_high_outlined),
                            label: const Text('Auto-fill expected'),
                          ),
                          const SizedBox(height: 10),
                          Text(
                            'Corner annotator (${_manualCorners.length}/4) '
                            'order: ${_datasetReport!.dataset.pointOrder}',
                            style: TextStyle(
                              fontSize: 12,
                              color: Colors.white.withValues(alpha: 0.80),
                            ),
                          ),
                          const SizedBox(height: 8),
                          if (imageBytes != null && _selectedImageSize != null)
                            _CornerAnnotator(
                              bytes: imageBytes,
                              imageSize: _selectedImageSize!,
                              corners: _manualCorners,
                              onTapImage: _onManualCornerTapped,
                            )
                          else if (imageBytes != null)
                            Text(
                              'Loading image metadata...',
                              style: TextStyle(
                                fontSize: 12,
                                color: Colors.white.withValues(alpha: 0.70),
                              ),
                            ),
                          const SizedBox(height: 8),
                          Wrap(
                            spacing: 8,
                            runSpacing: 8,
                            children: [
                              OutlinedButton.icon(
                                onPressed: _manualCorners.isEmpty
                                    ? null
                                    : _undoManualCorner,
                                icon: const Icon(Icons.undo),
                                label: const Text('Undo'),
                              ),
                              OutlinedButton.icon(
                                onPressed: _manualCorners.isEmpty
                                    ? null
                                    : _clearManualCorners,
                                icon: const Icon(Icons.clear),
                                label: const Text('Clear'),
                              ),
                              OutlinedButton.icon(
                                onPressed: _copyManualCornersJson,
                                icon: const Icon(Icons.copy_all_outlined),
                                label: const Text('Copy corners JSON'),
                              ),
                            ],
                          ),
                          if (_manualCorners.isNotEmpty) ...[
                            const SizedBox(height: 8),
                            SelectableText(
                              _manualCornersJson(),
                              style: const TextStyle(fontSize: 12),
                            ),
                          ],
                        ],
                      ],
                    ),
                  ),
                ],
                if (imageBytes != null) ...[
                  const SizedBox(height: 12),
                  _SectionCard(
                    title: 'Source preview',
                    subtitle:
                        '${_selectedImage?.path}\nDomain: ${_captureDomainLabel(_selectedCaptureDomain)}',
                    child: _ImagePreview(bytes: imageBytes),
                  ),
                ],
                if (_errorMessage != null) ...[
                  const SizedBox(height: 12),
                  Text(
                    _errorMessage!,
                    style: const TextStyle(color: Colors.redAccent),
                  ),
                ],
                if (_scanResult != null) ...[
                  const SizedBox(height: 12),
                  _SectionCard(
                    title: 'Detected board',
                    subtitle:
                        'OpenCV final verdict: board=${_scanResult!.boardDetected} '
                        'corners=${_scanResult!.geometry.corners.length}/4',
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        if (rectifiedBytes != null)
                          _ImagePreview(bytes: rectifiedBytes),
                        const SizedBox(height: 8),
                        Text(
                          'Gate allowed: ${_gateAllowedLabel(_scanResult!)} | '
                          'Final boardDetected: ${_scanResult!.boardDetected}',
                          style: TextStyle(
                            fontSize: 12,
                            color: Colors.white.withValues(alpha: 0.8),
                          ),
                        ),
                        if (_isGateFinalMismatch(_scanResult!)) ...[
                          const SizedBox(height: 4),
                          Text(
                            'gateAllowed != boardDetected',
                            style: const TextStyle(
                              fontSize: 12,
                              color: Colors.amberAccent,
                            ),
                          ),
                        ],
                        const SizedBox(height: 8),
                        if (_scanResult!.boardDetected)
                          Text(
                            'Detected corners: ${_formatCorners(_scanResult)}',
                            style: TextStyle(
                              fontSize: 12,
                              color: Colors.white.withValues(alpha: 0.8),
                            ),
                          )
                        else
                          Text(
                            'No board detected: quad hidden',
                            style: TextStyle(
                              fontSize: 12,
                              color: Colors.white.withValues(alpha: 0.75),
                            ),
                          ),
                        const SizedBox(height: 8),
                        SelectableText(
                          'Detector debug: ${_scanResult!.detectorDebug}',
                          style: TextStyle(
                            fontSize: 12,
                            color: Colors.white.withValues(alpha: 0.8),
                          ),
                        ),
                      ],
                    ),
                  ),
                  const SizedBox(height: 12),
                  _SectionCard(
                    title: 'Manual correction',
                    subtitle: 'Tap a square then select a piece',
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Wrap(
                          spacing: 10,
                          runSpacing: 10,
                          children: [
                            OutlinedButton.icon(
                              onPressed: () =>
                                  setState(() => _flipped = !_flipped),
                              icon: const Icon(
                                Icons.flip_camera_android_outlined,
                              ),
                              label: const Text('Flip board'),
                            ),
                            OutlinedButton.icon(
                              onPressed: _scanResult == null
                                  ? null
                                  : () => setState(() {
                                      _editablePosition =
                                          _scanResult!.detectedPosition;
                                      _refreshFenAndValidation();
                                    }),
                              icon: const Icon(Icons.restart_alt),
                              label: const Text('Reset to detected'),
                            ),
                          ],
                        ),
                        const SizedBox(height: 10),
                        if (_editablePosition != null)
                          BoardCorrectionEditor(
                            position: _editablePosition!,
                            squareMapper: _squareMapper,
                            flipped: _flipped,
                            onSquareTap: _editSquare,
                          ),
                        const SizedBox(height: 10),
                        if (_validation.errors.isNotEmpty)
                          ..._validation.errors.map(
                            (e) => Text(
                              'Error: $e',
                              style: const TextStyle(color: Colors.redAccent),
                            ),
                          ),
                        if (_validation.warnings.isNotEmpty)
                          ..._validation.warnings.map(
                            (w) => Text(
                              'Warning: $w',
                              style: const TextStyle(color: Colors.amberAccent),
                            ),
                          ),
                        const SizedBox(height: 10),
                        SelectableText(
                          _finalFen ?? '',
                          style: const TextStyle(fontSize: 13),
                        ),
                        const SizedBox(height: 10),
                        Wrap(
                          spacing: 10,
                          runSpacing: 10,
                          children: [
                            FilledButton.icon(
                              onPressed: _finalFen == null
                                  ? null
                                  : () {
                                      Clipboard.setData(
                                        ClipboardData(text: _finalFen!),
                                      );
                                      ScaffoldMessenger.of(
                                        context,
                                      ).showSnackBar(
                                        const SnackBar(
                                          content: Text('FEN copied'),
                                        ),
                                      );
                                    },
                              icon: const Icon(Icons.copy_all_outlined),
                              label: const Text('Copy FEN'),
                            ),
                            FilledButton.icon(
                              onPressed:
                                  (_finalFen == null || !_validation.isValid)
                                  ? null
                                  : _openAnalysis,
                              icon: const Icon(Icons.analytics_outlined),
                              label: const Text('Open analysis'),
                            ),
                          ],
                        ),
                      ],
                    ),
                  ),
                  const SizedBox(height: 12),
                  _SectionCard(
                    title: 'MVP notes',
                    child: const Text(
                      'Current version uses an OpenCV + statistical hybrid corner detector.\n'
                      'TODO(scan-v2): native OpenCV quadrilateral detection + perspective warp.\n'
                      'TODO(scan-v2): On-device piece classifier (TFLite/LiteRT).\n'
                      'Architecture is already split to allow these upgrades.',
                    ),
                  ),
                ],
              ],
            ),
            if (_isScanning)
              const Positioned.fill(
                child: AbsorbPointer(
                  absorbing: true,
                  child: ColoredBox(
                    color: Color(0x00000000),
                    child: Center(child: CircularProgressIndicator()),
                  ),
                ),
              ),
          ],
        ),
      ),
    );
  }
}

class _SectionCard extends StatelessWidget {
  const _SectionCard({required this.title, required this.child, this.subtitle});

  final String title;
  final String? subtitle;
  final Widget child;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: Colors.white.withValues(alpha: 0.06),
        borderRadius: BorderRadius.circular(14),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            title,
            style: const TextStyle(fontWeight: FontWeight.w700, fontSize: 15),
          ),
          if (subtitle != null) ...[
            const SizedBox(height: 2),
            Text(
              subtitle!,
              style: TextStyle(
                fontSize: 12,
                color: Colors.white.withValues(alpha: 0.70),
              ),
            ),
          ],
          const SizedBox(height: 10),
          child,
        ],
      ),
    );
  }
}

class _CornerAnnotator extends StatefulWidget {
  const _CornerAnnotator({
    required this.bytes,
    required this.imageSize,
    required this.corners,
    required this.onTapImage,
  });

  final Uint8List bytes;
  final Size imageSize;
  final List<BoardCorner> corners;
  final ValueChanged<Offset> onTapImage;

  @override
  State<_CornerAnnotator> createState() => _CornerAnnotatorState();
}

class _CornerAnnotatorState extends State<_CornerAnnotator> {
  late final TransformationController _controller;

  @override
  void initState() {
    super.initState();
    _controller = TransformationController();
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  void _resetView() {
    _controller.value = Matrix4.identity();
  }

  @override
  Widget build(BuildContext context) {
    const previewHeight = 260.0;
    if (widget.imageSize.width <= 0 || widget.imageSize.height <= 0) {
      return const SizedBox(
        height: previewHeight,
        child: Center(child: Text('Invalid image size')),
      );
    }

    return ClipRRect(
      borderRadius: BorderRadius.circular(10),
      child: SizedBox(
        height: previewHeight,
        width: double.infinity,
        child: LayoutBuilder(
          builder: (context, constraints) {
            final output = Size(constraints.maxWidth, previewHeight);
            final fitted = applyBoxFit(
              BoxFit.contain,
              widget.imageSize,
              output,
            );
            final imageRect = Alignment.center.inscribe(
              fitted.destination,
              Offset.zero & output,
            );

            return Stack(
              fit: StackFit.expand,
              children: [
                Container(color: Colors.black.withValues(alpha: 0.24)),
                GestureDetector(
                  behavior: HitTestBehavior.opaque,
                  onDoubleTap: _resetView,
                  onTapUp: (details) {
                    final scenePoint = _controller.toScene(
                      details.localPosition,
                    );
                    if (!imageRect.contains(scenePoint)) {
                      return;
                    }
                    final x =
                        ((scenePoint.dx - imageRect.left) / imageRect.width) *
                        widget.imageSize.width;
                    final y =
                        ((scenePoint.dy - imageRect.top) / imageRect.height) *
                        widget.imageSize.height;
                    widget.onTapImage(
                      Offset(
                        x.clamp(0, widget.imageSize.width),
                        y.clamp(0, widget.imageSize.height),
                      ),
                    );
                  },
                  child: InteractiveViewer(
                    transformationController: _controller,
                    minScale: 1,
                    maxScale: 8,
                    child: SizedBox(
                      width: output.width,
                      height: output.height,
                      child: Stack(
                        fit: StackFit.expand,
                        children: [
                          Positioned.fromRect(
                            rect: imageRect,
                            child: Image.memory(widget.bytes, fit: BoxFit.fill),
                          ),
                          ...widget.corners.asMap().entries.map((entry) {
                            final index = entry.key;
                            final corner = entry.value;
                            final left =
                                imageRect.left +
                                (corner.x / widget.imageSize.width) *
                                    imageRect.width;
                            final top =
                                imageRect.top +
                                (corner.y / widget.imageSize.height) *
                                    imageRect.height;
                            return Positioned(
                              left: left - 10,
                              top: top - 10,
                              child: Container(
                                width: 20,
                                height: 20,
                                alignment: Alignment.center,
                                decoration: BoxDecoration(
                                  color: Colors.redAccent,
                                  shape: BoxShape.circle,
                                  border: Border.all(
                                    color: Colors.white,
                                    width: 1.5,
                                  ),
                                ),
                                child: Text(
                                  '${index + 1}',
                                  style: const TextStyle(
                                    fontSize: 10,
                                    fontWeight: FontWeight.w800,
                                  ),
                                ),
                              ),
                            );
                          }),
                        ],
                      ),
                    ),
                  ),
                ),
                Positioned(
                  top: 8,
                  right: 8,
                  child: DecoratedBox(
                    decoration: BoxDecoration(
                      color: Colors.black.withValues(alpha: 0.45),
                      borderRadius: BorderRadius.circular(8),
                    ),
                    child: Padding(
                      padding: const EdgeInsets.symmetric(
                        horizontal: 8,
                        vertical: 5,
                      ),
                      child: Text(
                        'Pinch to zoom',
                        style: TextStyle(
                          fontSize: 11,
                          color: Colors.white.withValues(alpha: 0.92),
                        ),
                      ),
                    ),
                  ),
                ),
              ],
            );
          },
        ),
      ),
    );
  }
}

class _ImagePreview extends StatelessWidget {
  const _ImagePreview({required this.bytes});

  final Uint8List bytes;

  @override
  Widget build(BuildContext context) {
    return ClipRRect(
      borderRadius: BorderRadius.circular(10),
      child: SizedBox(
        height: 220,
        width: double.infinity,
        child: Image.memory(bytes, fit: BoxFit.cover),
      ),
    );
  }
}
