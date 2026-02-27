import 'dart:convert';
import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image_picker/image_picker.dart';

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
import '../domain/usecases/run_scan_dataset_validation_use_case.dart';
import '../domain/usecases/scan_position_use_case.dart';
import 'widgets/board_correction_editor.dart';
import 'widgets/piece_chooser_sheet.dart';

enum _ScanCaptureDomain { photoReal, screen }

class ScanPage extends StatefulWidget {
  const ScanPage({super.key, required this.onAnalyzeFen});

  final ValueChanged<String> onAnalyzeFen;

  @override
  State<ScanPage> createState() => _ScanPageState();
}

class _ScanPageState extends State<ScanPage> {
  static const String _scanCoreRevision = 'scan-core-r2026-02-27-01';
  static const double _photoRealAcceptThreshold = 0.89;
  static const double _photoRealRejectThreshold = 0.60;
  static const double _screenAcceptThreshold = 0.89;
  static const double _screenRejectThreshold = 0.60;
  static const String _photoRealBoardModelAssetPath =
      'assets/scan_models/board_binary.tflite';
  static const String _screenBoardModelAssetPath =
      'assets/scan_models/board_binary.tflite';

  final ImagePicker _picker = ImagePicker();
  final GridSquareMapper _squareMapper = const GridSquareMapper();
  final FenBuilder _fenBuilder = const BasicFenBuilder();
  final PositionValidator _validator = const BasicPositionValidator();
  late final ScanPositionUseCase _scanUseCasePhotoReal;
  late final ScanPositionUseCase _scanUseCaseScreen;
  late final ScanPositionUseCase _datasetScanUseCase;
  late final ScanPositionUseCase _datasetScanUseCaseFast;
  late final RunScanDatasetValidationUseCase _datasetValidationUseCase;
  late final RunScanDatasetValidationUseCase _datasetValidationUseCaseFast;

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
  bool _selectedImageHasExif = false;
  bool _selectedImageLooksScreenshot = false;

  @override
  void initState() {
    super.initState();
    _scanUseCasePhotoReal = DefaultScanPipelineFactory.create(
      validator: _validator,
      fenBuilder: _fenBuilder,
      useBoardPresenceGate: true,
      boardPresenceThreshold: _photoRealAcceptThreshold,
      boardPresenceRejectThreshold: _photoRealRejectThreshold,
      boardPresenceModelAssetPath: _photoRealBoardModelAssetPath,
    );
    _scanUseCaseScreen = DefaultScanPipelineFactory.create(
      validator: _validator,
      fenBuilder: _fenBuilder,
      useBoardPresenceGate: true,
      boardPresenceThreshold: _screenAcceptThreshold,
      boardPresenceRejectThreshold: _screenRejectThreshold,
      boardPresenceModelAssetPath: _screenBoardModelAssetPath,
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
      final file = await _picker.pickImage(source: source, imageQuality: 95);
      if (file == null) {
        return;
      }
      final bytes = await file.readAsBytes();
      final hasExif = _hasExifMetadata(bytes);
      final looksScreenshot = _looksLikeScreenshotPath(file.path);
      final captureDomain = _resolveCaptureDomain(
        source: source,
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
        _selectedImageHasExif = hasExif;
        _selectedImageLooksScreenshot = looksScreenshot;
        _scanResult = null;
        _editablePosition = null;
        _finalFen = null;
        _errorMessage = null;
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
    final scanUseCase = _scanUseCaseForDomain(_selectedCaptureDomain);

    try {
      final result = await scanUseCase.execute(image);
      if (!mounted) {
        return;
      }
      setState(() {
        _scanResult = result;
        _editablePosition = result.detectedPosition;
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

  ScanPositionUseCase _scanUseCaseForDomain(_ScanCaptureDomain domain) {
    switch (domain) {
      case _ScanCaptureDomain.photoReal:
        return _scanUseCasePhotoReal;
      case _ScanCaptureDomain.screen:
        return _scanUseCaseScreen;
    }
  }

  _ScanCaptureDomain _resolveCaptureDomain({
    required ImageSource? source,
    required bool hasExif,
    required bool looksScreenshot,
  }) {
    if (source == ImageSource.camera) {
      return _ScanCaptureDomain.photoReal;
    }
    if (source == ImageSource.gallery) {
      if (looksScreenshot || !hasExif) {
        return _ScanCaptureDomain.screen;
      }
      return _ScanCaptureDomain.screen;
    }
    if (looksScreenshot || !hasExif) {
      return _ScanCaptureDomain.screen;
    }
    return _ScanCaptureDomain.photoReal;
  }

  String _captureDomainLabel(_ScanCaptureDomain domain) {
    return domain == _ScanCaptureDomain.photoReal ? 'photo_real' : 'screen';
  }

  String _selectedImageSourceLabel() {
    final source = _selectedImageSource;
    if (source == null) {
      return 'unknown';
    }
    return source == ImageSource.camera ? 'camera' : 'gallery';
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
        source: null,
        hasExif: inferredHasExif,
        looksScreenshot: inferredLooksScreenshot,
      );
      _selectedImageHasExif = inferredHasExif;
      _selectedImageLooksScreenshot = inferredLooksScreenshot;
      _scanResult = evaluation.result;
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

    return Scaffold(
      appBar: AppBar(title: const Text('Scan position')),
      backgroundColor: const Color(0xFF282725),
      body: SafeArea(
        child: ListView(
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
                        onPressed: () => _pickImage(ImageSource.camera),
                        icon: const Icon(Icons.photo_camera_outlined),
                        label: const Text('Camera'),
                      ),
                      FilledButton.tonalIcon(
                        onPressed: () => _pickImage(ImageSource.gallery),
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
                        label: Text(_isScanning ? 'Scanning...' : 'Scanner'),
                      ),
                      OutlinedButton.icon(
                        onPressed:
                            (_isRunningDatasetValidation ||
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
                            (_isRunningDatasetValidation ||
                                _isLoadingSelectedEvaluation ||
                                _isAutoFillingAllExpected)
                            ? null
                            : (v) =>
                                  setState(() => _useFastDatasetValidation = v),
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
                      '(source=${_selectedImageSourceLabel()}, '
                      'screenshot_hint=${_selectedImageLooksScreenshot ? "yes" : "no"}, '
                      'exif=${_selectedImageHasExif ? "present" : "absent"})',
                      style: TextStyle(
                        fontSize: 12,
                        color: Colors.white.withValues(alpha: 0.75),
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
                              child: Text('${e.testCase.id}  ${e.statusLabel}'),
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
                              child: CircularProgressIndicator(strokeWidth: 2),
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
                              child: CircularProgressIndicator(strokeWidth: 2),
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
                      if (_selectedEvaluation!.cornerErrorMetrics != null) ...[
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
                            } else if (meanPercent < 8.0 && maxPercent < 15.0) {
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
                    'Corners: ${_scanResult!.geometry.corners.length}/4 (OpenCV hybrid detector)',
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    if (rectifiedBytes != null)
                      _ImagePreview(bytes: rectifiedBytes),
                    const SizedBox(height: 8),
                    Text(
                      'Detected corners: ${_formatCorners(_scanResult)}',
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
                          onPressed: () => setState(() => _flipped = !_flipped),
                          icon: const Icon(Icons.flip_camera_android_outlined),
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
                                  ScaffoldMessenger.of(context).showSnackBar(
                                    const SnackBar(content: Text('FEN copied')),
                                  );
                                },
                          icon: const Icon(Icons.copy_all_outlined),
                          label: const Text('Copy FEN'),
                        ),
                        FilledButton.icon(
                          onPressed: (_finalFen == null || !_validation.isValid)
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
