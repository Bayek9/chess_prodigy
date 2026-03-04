import 'dart:math' as math;
import 'dart:ui' as ui;

import 'package:flutter/foundation.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

import '../../domain/entities/board_scan_position.dart';
import '../../domain/entities/scan_image.dart';
import '../../domain/entities/scan_piece.dart';
import '../../domain/services/grid_square_mapper.dart';
import '../../domain/services/piece_classifier.dart';

class TflitePieceClassifier implements PieceClassifier {
  TflitePieceClassifier({
    this.modelAssetPath = 'assets/scan_models/piece_13cls_fp16.tflite',
    this.inputSize = 128,
    this.threads = 4,
    this.cropInsetFraction = 0.08,
    this.useNnApiForAndroid = false,
    this.logPerfInNonRelease = true,
    this.enableAutoTune = true,
    this.autoTuneBenchmarkInvokes = 3,
  });

  final String modelAssetPath;
  final int inputSize;
  final int threads;
  final double cropInsetFraction;
  final bool useNnApiForAndroid;
  final bool logPerfInNonRelease;
  final bool enableAutoTune;
  final int autoTuneBenchmarkInvokes;

  Interpreter? _interpreter;
  Tensor? _inputTensor;
  Tensor? _outputTensor;

  bool _loadAttempted = false;
  String? _loadError;

  int _resolvedThreads = 4;
  bool _resolvedUseNnApi = false;

  bool _batchInferenceEnabled = false;
  String _inferenceMode = 'uninitialized';
  int _classCount = _classToPiece.length;

  Float32List? _batchInputBuffer;
  Uint8List? _batchInputBytes;

  Float32List? _singleInputBuffer;
  Uint8List? _singleInputBytes;

  PieceClassifierPerfStats? _lastPerfStats;

  static const GridSquareMapper _squareMapper = GridSquareMapper();
  static const int _squareCount = 64;
  static const int _channels = 3;

  static const String _autoTunePrefsVersion = 'r1';
  static const List<_RuntimeConfig> _autoTuneCandidates = <_RuntimeConfig>[
    _RuntimeConfig(threads: 4, useNnApi: false),
    _RuntimeConfig(threads: 2, useNnApi: false),
    _RuntimeConfig(threads: 4, useNnApi: true),
  ];

  static const int _emptyId = 0;
  static const int _whitePawnId = 1;
  static const int _whiteKingId = 6;
  static const int _blackPawnId = 7;
  static const int _blackKingId = 12;

  static const List<ScanPiece?> _classToPiece = <ScanPiece?>[
    null,
    ScanPiece(color: ScanPieceColor.white, type: ScanPieceType.pawn),
    ScanPiece(color: ScanPieceColor.white, type: ScanPieceType.knight),
    ScanPiece(color: ScanPieceColor.white, type: ScanPieceType.bishop),
    ScanPiece(color: ScanPieceColor.white, type: ScanPieceType.rook),
    ScanPiece(color: ScanPieceColor.white, type: ScanPieceType.queen),
    ScanPiece(color: ScanPieceColor.white, type: ScanPieceType.king),
    ScanPiece(color: ScanPieceColor.black, type: ScanPieceType.pawn),
    ScanPiece(color: ScanPieceColor.black, type: ScanPieceType.knight),
    ScanPiece(color: ScanPieceColor.black, type: ScanPieceType.bishop),
    ScanPiece(color: ScanPieceColor.black, type: ScanPieceType.rook),
    ScanPiece(color: ScanPieceColor.black, type: ScanPieceType.queen),
    ScanPiece(color: ScanPieceColor.black, type: ScanPieceType.king),
  ];

  static final List<String> _squaresByIndex = List<String>.generate(
    _squareCount,
    (index) => _squareMapper.squareAt(row: index ~/ 8, col: index % 8),
    growable: false,
  );

  PieceClassifierPerfStats? get lastPerfStats => _lastPerfStats;

  @override
  Future<BoardScanPosition> classify(RectifiedBoardImage rectifiedBoard) async {
    await _ensureLoaded();
    final interpreter = _interpreter;
    if (interpreter == null) {
      if (!kReleaseMode) {
        debugPrint('[scan][piece_tflite] unavailable error=$_loadError');
      }
      return BoardScanPosition(pieces: const <String, ScanPiece>{});
    }

    final decoded = await _decodeRgba(rectifiedBoard.bytes);
    if (decoded == null || decoded.width < 8 || decoded.height < 8) {
      return BoardScanPosition(pieces: const <String, ScanPiece>{});
    }

    final run = _batchInferenceEnabled
        ? _predictSquaresBatch(interpreter: interpreter, decoded: decoded)
        : _predictSquaresSingle(interpreter: interpreter, decoded: decoded);

    final postWatch = Stopwatch()..start();

    final selected = run.predictions
        .map((prediction) => prediction.top1)
        .toList(growable: false);
    final constraintsChanges = _applyChessConstraints(
      run.predictions,
      selected,
    );

    final pieces = <String, ScanPiece>{};
    for (int i = 0; i < run.predictions.length; i++) {
      final classId = selected[i];
      if (classId == _emptyId || classId >= _classToPiece.length) {
        continue;
      }
      final piece = _classToPiece[classId];
      if (piece == null) {
        continue;
      }
      pieces[run.predictions[i].square] = piece;
    }

    postWatch.stop();

    final stats = PieceClassifierPerfStats(
      mode: run.mode,
      preprocessMs: run.preprocessMs,
      invokeMs: run.invokeMs,
      decodeMs: run.decodeMs + postWatch.elapsedMilliseconds,
      fallbackUsed: run.fallbackUsed,
    );
    _lastPerfStats = stats;

    if (!kReleaseMode && logPerfInNonRelease) {
      final constraintsPart = constraintsChanges > 0
          ? ' constraints_changes=$constraintsChanges'
          : '';
      final fallbackPart = run.fallbackUsed ? ' fallback=true' : '';
      debugPrint(
        '[scan][piece_tflite_perf] '
        'mode=${stats.mode} '
        'preprocess_ms=${stats.preprocessMs} '
        'invoke_ms=${stats.invokeMs} '
        'decode_post_ms=${stats.decodeMs} '
        'total_ms=${stats.totalMs}'
        '$fallbackPart'
        '$constraintsPart '
        'pieces=${pieces.length}',
      );
    }

    return BoardScanPosition(pieces: pieces);
  }

  _InferenceRun _predictSquaresBatch({
    required Interpreter interpreter,
    required _DecodedRgbaImage decoded,
  }) {
    final inputBuffer = _batchInputBuffer;
    final inputBytes = _batchInputBytes;
    final inputTensor = _inputTensor;
    final outputTensor = _outputTensor;

    if (inputBuffer == null ||
        inputBytes == null ||
        inputTensor == null ||
        outputTensor == null) {
      _configureSingleLayout(interpreter);
      return _predictSquaresSingle(
        interpreter: interpreter,
        decoded: decoded,
        fallbackUsed: true,
        fallbackMode: 'single64(reinit)',
      );
    }

    final preprocessWatch = Stopwatch()..start();
    _fillBatchInputBuffer(decoded: decoded, inputBuffer: inputBuffer);
    preprocessWatch.stop();

    final invokeWatch = Stopwatch()..start();
    try {
      inputTensor.data = inputBytes;
      interpreter.invoke();
    } catch (e) {
      invokeWatch.stop();
      if (!kReleaseMode) {
        debugPrint('[scan][piece_tflite] batch_invoke_fallback reason=$e');
      }
      _configureSingleLayout(interpreter);
      return _predictSquaresSingle(
        interpreter: interpreter,
        decoded: decoded,
        fallbackUsed: true,
        fallbackMode: 'single64(batch_fallback)',
      );
    }
    invokeWatch.stop();

    final decodeWatch = Stopwatch()..start();
    final predictions = _decodeBatchPredictions(outputTensor: outputTensor);
    decodeWatch.stop();

    return _InferenceRun(
      predictions: predictions,
      preprocessMs: preprocessWatch.elapsedMilliseconds,
      invokeMs: invokeWatch.elapsedMilliseconds,
      decodeMs: decodeWatch.elapsedMilliseconds,
      mode: _inferenceMode,
      fallbackUsed: false,
    );
  }

  _InferenceRun _predictSquaresSingle({
    required Interpreter interpreter,
    required _DecodedRgbaImage decoded,
    bool fallbackUsed = false,
    String fallbackMode = 'single64',
  }) {
    final inputBuffer = _singleInputBuffer;
    final inputBytes = _singleInputBytes;
    final inputTensor = _inputTensor;
    final outputTensor = _outputTensor;

    if (inputBuffer == null ||
        inputBytes == null ||
        inputTensor == null ||
        outputTensor == null) {
      return const _InferenceRun(
        predictions: <_SquarePrediction>[],
        preprocessMs: 0,
        invokeMs: 0,
        decodeMs: 0,
        mode: 'single64(unavailable)',
        fallbackUsed: true,
      );
    }

    final predictions = <_SquarePrediction>[];

    final preprocessWatch = Stopwatch();
    final invokeWatch = Stopwatch();
    final decodeWatch = Stopwatch();

    final geometry = _BoardSamplingGeometry.fromDecoded(
      decoded: decoded,
      cropInsetFraction: cropInsetFraction,
    );

    for (int index = 0; index < _squareCount; index++) {
      final row = index ~/ 8;
      final col = index % 8;

      preprocessWatch.start();
      _writeSquareToBuffer(
        decoded: decoded,
        geometry: geometry,
        row: row,
        col: col,
        target: inputBuffer,
        targetOffset: 0,
      );
      preprocessWatch.stop();

      invokeWatch.start();
      inputTensor.data = inputBytes;
      interpreter.invoke();
      invokeWatch.stop();

      decodeWatch.start();
      final probs = _decodeSingleProbabilities(outputTensor: outputTensor);
      final top2 = _top2ClassIdsFromList(probs);
      predictions.add(
        _SquarePrediction(
          row: row,
          col: col,
          square: _squaresByIndex[index],
          probs: probs,
          top1: top2.$1,
          top2: top2.$2,
        ),
      );
      decodeWatch.stop();
    }

    return _InferenceRun(
      predictions: predictions,
      preprocessMs: preprocessWatch.elapsedMilliseconds,
      invokeMs: invokeWatch.elapsedMilliseconds,
      decodeMs: decodeWatch.elapsedMilliseconds,
      mode: fallbackMode,
      fallbackUsed: fallbackUsed,
    );
  }

  List<_SquarePrediction> _decodeBatchPredictions({
    required Tensor outputTensor,
  }) {
    final data = outputTensor.data;
    if (data.isEmpty) {
      return const <_SquarePrediction>[];
    }

    final floatCount = data.lengthInBytes ~/ Float32List.bytesPerElement;
    final output = data.buffer.asFloat32List(data.offsetInBytes, floatCount);

    final classCountFromOutput = output.length ~/ _squareCount;
    final classCount = math.min(classCountFromOutput, _classCount);
    if (classCount <= 0) {
      return const <_SquarePrediction>[];
    }

    final predictions = <_SquarePrediction>[];
    for (int index = 0; index < _squareCount; index++) {
      final start = index * classCountFromOutput;
      final probs = List<double>.generate(
        classCount,
        (offset) => output[start + offset],
        growable: false,
      );
      final top2 = _top2ClassIdsFromList(probs);
      predictions.add(
        _SquarePrediction(
          row: index ~/ 8,
          col: index % 8,
          square: _squaresByIndex[index],
          probs: probs,
          top1: top2.$1,
          top2: top2.$2,
        ),
      );
    }
    return predictions;
  }

  List<double> _decodeSingleProbabilities({required Tensor outputTensor}) {
    final data = outputTensor.data;
    if (data.isEmpty) {
      return List<double>.filled(_classCount, 0.0, growable: false);
    }

    final floatCount = data.lengthInBytes ~/ Float32List.bytesPerElement;
    final output = data.buffer.asFloat32List(data.offsetInBytes, floatCount);

    final classCount = math.min(_classCount, output.length);
    final probs = List<double>.filled(classCount, 0.0, growable: false);
    for (int i = 0; i < classCount; i++) {
      probs[i] = output[i];
    }
    return probs;
  }

  void _fillBatchInputBuffer({
    required _DecodedRgbaImage decoded,
    required Float32List inputBuffer,
  }) {
    final geometry = _BoardSamplingGeometry.fromDecoded(
      decoded: decoded,
      cropInsetFraction: cropInsetFraction,
    );

    final squareStride = inputSize * inputSize * _channels;
    for (int index = 0; index < _squareCount; index++) {
      final row = index ~/ 8;
      final col = index % 8;
      _writeSquareToBuffer(
        decoded: decoded,
        geometry: geometry,
        row: row,
        col: col,
        target: inputBuffer,
        targetOffset: index * squareStride,
      );
    }
  }

  void _writeSquareToBuffer({
    required _DecodedRgbaImage decoded,
    required _BoardSamplingGeometry geometry,
    required int row,
    required int col,
    required Float32List target,
    required int targetOffset,
  }) {
    final rgba = decoded.rgba;
    final width = decoded.width;
    final height = decoded.height;

    final left =
        geometry.boardLeft + (col * geometry.cellSize) + geometry.inset;
    final top = geometry.boardTop + (row * geometry.cellSize) + geometry.inset;
    final right =
        geometry.boardLeft + ((col + 1) * geometry.cellSize) - geometry.inset;
    final bottom =
        geometry.boardTop + ((row + 1) * geometry.cellSize) - geometry.inset;

    final sampledWidth = math.max(1.0, right - left);
    final sampledHeight = math.max(1.0, bottom - top);

    final maxX = (width - 1).toDouble();
    final maxY = (height - 1).toDouble();
    const inv255 = 1.0 / 255.0;

    var writeIndex = targetOffset;

    for (int y = 0; y < inputSize; y++) {
      final srcY = top + (((y + 0.5) * sampledHeight) / inputSize);
      final clampedY = srcY.clamp(0.0, maxY).toDouble();
      final y0 = clampedY.floor();
      final y1 = math.min(height - 1, y0 + 1);
      final ty = clampedY - y0;

      for (int x = 0; x < inputSize; x++) {
        final srcX = left + (((x + 0.5) * sampledWidth) / inputSize);
        final clampedX = srcX.clamp(0.0, maxX).toDouble();
        final x0 = clampedX.floor();
        final x1 = math.min(width - 1, x0 + 1);
        final tx = clampedX - x0;

        final p00 = (y0 * width + x0) * 4;
        final p10 = (y0 * width + x1) * 4;
        final p01 = (y1 * width + x0) * 4;
        final p11 = (y1 * width + x1) * 4;

        final w00 = (1.0 - tx) * (1.0 - ty);
        final w10 = tx * (1.0 - ty);
        final w01 = (1.0 - tx) * ty;
        final w11 = tx * ty;

        target[writeIndex++] =
            ((rgba[p00] * w00) +
                (rgba[p10] * w10) +
                (rgba[p01] * w01) +
                (rgba[p11] * w11)) *
            inv255;
        target[writeIndex++] =
            ((rgba[p00 + 1] * w00) +
                (rgba[p10 + 1] * w10) +
                (rgba[p01 + 1] * w01) +
                (rgba[p11 + 1] * w11)) *
            inv255;
        target[writeIndex++] =
            ((rgba[p00 + 2] * w00) +
                (rgba[p10 + 2] * w10) +
                (rgba[p01 + 2] * w01) +
                (rgba[p11 + 2] * w11)) *
            inv255;
      }
    }
  }

  (int, int) _top2ClassIdsFromList(List<double> probs) {
    if (probs.isEmpty) {
      return (_emptyId, _emptyId);
    }
    if (probs.length == 1) {
      return (0, 0);
    }

    var best = 0;
    var second = 1;
    if (probs[second] > probs[best]) {
      final tmp = best;
      best = second;
      second = tmp;
    }

    for (int i = 2; i < probs.length; i++) {
      final v = probs[i];
      if (v > probs[best]) {
        second = best;
        best = i;
      } else if (v > probs[second]) {
        second = i;
      }
    }
    return (best, second);
  }

  int _applyChessConstraints(
    List<_SquarePrediction> predictions,
    List<int> selected,
  ) {
    var changes = 0;

    changes += _enforceNoPawnsOnBackRanks(predictions, selected);

    changes += _enforceExactlyOneKing(
      predictions,
      selected,
      kingId: _whiteKingId,
      otherKingId: _blackKingId,
    );
    changes += _enforceExactlyOneKing(
      predictions,
      selected,
      kingId: _blackKingId,
      otherKingId: _whiteKingId,
    );

    changes += _enforceMaxPawns(
      predictions,
      selected,
      pawnId: _whitePawnId,
      maxPawns: 8,
    );
    changes += _enforceMaxPawns(
      predictions,
      selected,
      pawnId: _blackPawnId,
      maxPawns: 8,
    );

    changes += _enforceNoPawnsOnBackRanks(predictions, selected);

    return changes;
  }

  int _enforceNoPawnsOnBackRanks(
    List<_SquarePrediction> predictions,
    List<int> selected,
  ) {
    var changes = 0;
    for (int idx = 0; idx < predictions.length; idx++) {
      final prediction = predictions[idx];
      final classId = selected[idx];
      if (!_isPawnClass(classId)) {
        continue;
      }
      if (prediction.row != 0 && prediction.row != 7) {
        continue;
      }

      final replacement =
          _bestClassFor(
            prediction,
            (candidate) => !_isPawnClass(candidate) && !_isKingClass(candidate),
          ) ??
          _bestClassFor(prediction, (candidate) => !_isPawnClass(candidate));

      if (replacement != null && replacement != classId) {
        selected[idx] = replacement;
        changes += 1;
      }
    }
    return changes;
  }

  int _enforceMaxPawns(
    List<_SquarePrediction> predictions,
    List<int> selected, {
    required int pawnId,
    required int maxPawns,
  }) {
    var changes = 0;
    while (_countClass(selected, pawnId) > maxPawns) {
      final pawnIndices = <int>[];
      for (int i = 0; i < selected.length; i++) {
        if (selected[i] == pawnId) {
          pawnIndices.add(i);
        }
      }
      if (pawnIndices.isEmpty) {
        break;
      }

      pawnIndices.sort((a, b) {
        final pa = predictions[a].probs[pawnId];
        final pb = predictions[b].probs[pawnId];
        final byConfidence = pa.compareTo(pb);
        if (byConfidence != 0) {
          return byConfidence;
        }
        return predictions[a].margin.compareTo(predictions[b].margin);
      });

      var changedOne = false;
      for (final idx in pawnIndices) {
        final replacement =
            _bestClassFor(
              predictions[idx],
              (candidate) => candidate != pawnId && !_isKingClass(candidate),
            ) ??
            _bestClassFor(predictions[idx], (candidate) => candidate != pawnId);

        if (replacement != null && replacement != selected[idx]) {
          selected[idx] = replacement;
          changes += 1;
          changedOne = true;
          break;
        }
      }

      if (!changedOne) {
        break;
      }
    }
    return changes;
  }

  int _enforceExactlyOneKing(
    List<_SquarePrediction> predictions,
    List<int> selected, {
    required int kingId,
    required int otherKingId,
  }) {
    var changes = 0;
    final kingIndices = <int>[];
    for (int i = 0; i < selected.length; i++) {
      if (selected[i] == kingId) {
        kingIndices.add(i);
      }
    }

    if (kingIndices.length > 1) {
      kingIndices.sort(
        (a, b) => predictions[a].probs[kingId].compareTo(
          predictions[b].probs[kingId],
        ),
      );
      for (int i = 0; i < kingIndices.length - 1; i++) {
        final idx = kingIndices[i];
        final replacement = _bestClassFor(
          predictions[idx],
          (candidate) => candidate != kingId && candidate != otherKingId,
        );
        if (replacement != null && replacement != selected[idx]) {
          selected[idx] = replacement;
          changes += 1;
        }
      }
    }

    if (_countClass(selected, kingId) == 0) {
      int? bestIdx;
      double bestScore = -double.infinity;
      for (int i = 0; i < predictions.length; i++) {
        if (selected[i] == otherKingId) {
          continue;
        }
        final score = predictions[i].probs[kingId];
        if (score > bestScore) {
          bestScore = score;
          bestIdx = i;
        }
      }
      bestIdx ??= 0;
      if (selected[bestIdx] != kingId) {
        selected[bestIdx] = kingId;
        changes += 1;
      }
    }

    return changes;
  }

  int _countClass(List<int> selected, int classId) {
    var count = 0;
    for (final value in selected) {
      if (value == classId) {
        count += 1;
      }
    }
    return count;
  }

  bool _isPawnClass(int classId) {
    return classId == _whitePawnId || classId == _blackPawnId;
  }

  bool _isKingClass(int classId) {
    return classId == _whiteKingId || classId == _blackKingId;
  }

  int? _bestClassFor(
    _SquarePrediction prediction,
    bool Function(int classId) predicate,
  ) {
    int? bestId;
    double bestScore = -double.infinity;
    for (int classId = 0; classId < prediction.probs.length; classId++) {
      if (!predicate(classId)) {
        continue;
      }
      final score = prediction.probs[classId];
      if (score > bestScore) {
        bestScore = score;
        bestId = classId;
      }
    }
    return bestId;
  }

  Future<void> _ensureLoaded() async {
    if (_loadAttempted) {
      return;
    }
    _loadAttempted = true;
    try {
      final runtimeConfig = await _resolveRuntimeConfig();
      _resolvedThreads = runtimeConfig.threads;
      _resolvedUseNnApi = runtimeConfig.useNnApi;

      final interpreter = await _createInterpreter(runtimeConfig);
      _interpreter = interpreter;
      _configureInferenceLayouts(interpreter);

      if (!kReleaseMode) {
        debugPrint(
          '[scan][piece_tflite] loaded '
          'mode=$_inferenceMode '
          'threads=$_resolvedThreads '
          'nnapi=$_resolvedUseNnApi '
          'autotune=$enableAutoTune',
        );
      }
    } catch (e) {
      _loadError = e.toString();
      _interpreter = null;
      _clearTensorCaches();
    }
  }

  bool get _supportsAutoTune =>
      !kIsWeb && defaultTargetPlatform == TargetPlatform.android;

  String get _prefsPrefix {
    final safeModel = modelAssetPath.replaceAll(RegExp(r'[^a-zA-Z0-9]+'), '_');
    return 'scan.piece_tflite.autotune.$_autoTunePrefsVersion.$safeModel.$inputSize';
  }

  Future<_RuntimeConfig> _resolveRuntimeConfig() async {
    final manual = _RuntimeConfig(
      threads: math.max(1, threads),
      useNnApi: useNnApiForAndroid && _supportsAutoTune,
    );

    if (!enableAutoTune || !_supportsAutoTune) {
      return manual;
    }

    final stored = await _loadStoredRuntimeConfig();
    if (stored != null) {
      if (!kReleaseMode) {
        debugPrint(
          '[scan][piece_tflite] autotune_use_cached '
          'threads=${stored.threads} nnapi=${stored.useNnApi}',
        );
      }
      return stored;
    }

    final tuned = await _autoTuneBestRuntimeConfig(fallback: manual);
    await _saveRuntimeConfig(tuned);
    return tuned;
  }

  Future<_RuntimeConfig?> _loadStoredRuntimeConfig() async {
    try {
      final prefs = await SharedPreferences.getInstance();
      final storedThreads = prefs.getInt('$_prefsPrefix.threads');
      final storedNnApi = prefs.getBool('$_prefsPrefix.nnapi');
      if (storedThreads == null || storedThreads <= 0 || storedNnApi == null) {
        return null;
      }
      return _RuntimeConfig(
        threads: storedThreads,
        useNnApi: storedNnApi && _supportsAutoTune,
      );
    } catch (_) {
      return null;
    }
  }

  Future<void> _saveRuntimeConfig(_RuntimeConfig config) async {
    try {
      final prefs = await SharedPreferences.getInstance();
      await prefs.setInt('$_prefsPrefix.threads', config.threads);
      await prefs.setBool('$_prefsPrefix.nnapi', config.useNnApi);
    } catch (_) {
      // Ignore persistence issues and keep runtime fallback.
    }
  }

  Future<_RuntimeConfig> _autoTuneBestRuntimeConfig({
    required _RuntimeConfig fallback,
  }) async {
    final candidates = <_RuntimeConfig>[];
    final seen = <String>{};

    void addCandidate(_RuntimeConfig config) {
      final key = '${config.threads}:${config.useNnApi}';
      if (seen.add(key)) {
        candidates.add(config);
      }
    }

    addCandidate(fallback);
    for (final candidate in _autoTuneCandidates) {
      if (candidate.useNnApi && !_supportsAutoTune) {
        continue;
      }
      addCandidate(candidate);
    }

    var best = fallback;
    var bestMs = double.infinity;

    for (final candidate in candidates) {
      final score = await _benchmarkRuntimeConfig(candidate);
      if (score < bestMs) {
        best = candidate;
        bestMs = score;
      }
      if (!kReleaseMode) {
        final pretty = score.isFinite ? score.toStringAsFixed(2) : 'inf';
        debugPrint(
          '[scan][piece_tflite] autotune_probe '
          'threads=${candidate.threads} nnapi=${candidate.useNnApi} '
          'avg_invoke_ms=$pretty',
        );
      }
    }

    if (!kReleaseMode) {
      debugPrint(
        '[scan][piece_tflite] autotune_pick '
        'threads=${best.threads} nnapi=${best.useNnApi}',
      );
    }

    return best;
  }

  Future<double> _benchmarkRuntimeConfig(_RuntimeConfig config) async {
    Interpreter? interpreter;
    try {
      interpreter = await _createInterpreter(config);
      _clearTensorCaches();
      final isBatchReady = _tryConfigureBatchLayout(interpreter);
      if (!isBatchReady || _inputTensor == null || _batchInputBytes == null) {
        return double.infinity;
      }

      final inputTensor = _inputTensor!;
      final inputBytes = _batchInputBytes!;
      inputTensor.data = inputBytes;
      interpreter.invoke(); // warmup

      final runs = math.max(1, autoTuneBenchmarkInvokes);
      final watch = Stopwatch()..start();
      for (int i = 0; i < runs; i++) {
        inputTensor.data = inputBytes;
        interpreter.invoke();
      }
      watch.stop();
      return watch.elapsedMicroseconds / 1000.0 / runs;
    } catch (_) {
      return double.infinity;
    } finally {
      interpreter?.close();
      _clearTensorCaches();
    }
  }

  Future<Interpreter> _createInterpreter(_RuntimeConfig config) async {
    final options = InterpreterOptions()..threads = config.threads;
    if (config.useNnApi) {
      options.useNnApiForAndroid = true;
    }
    return Interpreter.fromAsset(modelAssetPath, options: options);
  }

  void _clearTensorCaches() {
    _inputTensor = null;
    _outputTensor = null;
    _batchInputBuffer = null;
    _batchInputBytes = null;
    _singleInputBuffer = null;
    _singleInputBytes = null;
    _batchInferenceEnabled = false;
    _inferenceMode = 'uninitialized';
    _classCount = _classToPiece.length;
  }

  void _configureInferenceLayouts(Interpreter interpreter) {
    if (_tryConfigureBatchLayout(interpreter)) {
      return;
    }
    _configureSingleLayout(interpreter);
  }

  bool _tryConfigureBatchLayout(Interpreter interpreter) {
    try {
      interpreter.resizeInputTensor(0, <int>[64, inputSize, inputSize, 3]);
      interpreter.allocateTensors();

      final inputTensor = interpreter.getInputTensor(0);
      final outputTensor = interpreter.getOutputTensor(0);
      final outputShape = outputTensor.shape;

      if (outputShape.isEmpty || outputShape.first != 64) {
        throw StateError('unexpected_batch_output_shape=$outputShape');
      }

      final classCount = outputShape.last;
      if (classCount <= 0) {
        throw StateError('invalid_batch_class_count=$classCount');
      }

      _inputTensor = inputTensor;
      _outputTensor = outputTensor;
      _classCount = classCount;

      final batchInput = Float32List(64 * inputSize * inputSize * _channels);
      _batchInputBuffer = batchInput;
      _batchInputBytes = batchInput.buffer.asUint8List(
        batchInput.offsetInBytes,
        batchInput.lengthInBytes,
      );

      _singleInputBuffer = null;
      _singleInputBytes = null;

      _batchInferenceEnabled = true;
      _inferenceMode = 'batch64';
      return true;
    } catch (e) {
      if (!kReleaseMode) {
        debugPrint('[scan][piece_tflite] batch_layout_unavailable reason=$e');
      }
      return false;
    }
  }

  void _configureSingleLayout(Interpreter interpreter) {
    try {
      interpreter.resizeInputTensor(0, <int>[1, inputSize, inputSize, 3]);
      interpreter.allocateTensors();
    } catch (_) {
      // Keep current tensor layout when explicit single-shape resize fails.
    }

    final inputTensor = interpreter.getInputTensor(0);
    final outputTensor = interpreter.getOutputTensor(0);
    final outputShape = outputTensor.shape;

    final classCount = outputShape.isEmpty
        ? _classToPiece.length
        : outputShape.last;

    _inputTensor = inputTensor;
    _outputTensor = outputTensor;
    _classCount = classCount;

    final singleInput = Float32List(inputSize * inputSize * _channels);
    _singleInputBuffer = singleInput;
    _singleInputBytes = singleInput.buffer.asUint8List(
      singleInput.offsetInBytes,
      singleInput.lengthInBytes,
    );

    _batchInputBuffer = null;
    _batchInputBytes = null;

    _batchInferenceEnabled = false;
    _inferenceMode = 'single64';
  }

  Future<_DecodedRgbaImage?> _decodeRgba(Uint8List bytes) async {
    ui.Codec? codec;
    try {
      codec = await ui.instantiateImageCodec(bytes);
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
    } finally {
      codec?.dispose();
    }
  }
}

class _RuntimeConfig {
  const _RuntimeConfig({required this.threads, required this.useNnApi});

  final int threads;
  final bool useNnApi;
}

@immutable
class PieceClassifierPerfStats {
  const PieceClassifierPerfStats({
    required this.mode,
    required this.preprocessMs,
    required this.invokeMs,
    required this.decodeMs,
    required this.fallbackUsed,
  });

  final String mode;
  final int preprocessMs;
  final int invokeMs;
  final int decodeMs;
  final bool fallbackUsed;

  int get totalMs => preprocessMs + invokeMs + decodeMs;
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

class _BoardSamplingGeometry {
  const _BoardSamplingGeometry({
    required this.boardLeft,
    required this.boardTop,
    required this.cellSize,
    required this.inset,
  });

  factory _BoardSamplingGeometry.fromDecoded({
    required _DecodedRgbaImage decoded,
    required double cropInsetFraction,
  }) {
    final boardSize = math.min(decoded.width, decoded.height).toDouble();
    final cellSize = boardSize / 8.0;
    final inset = (cellSize * cropInsetFraction).clamp(0.0, cellSize * 0.35);
    return _BoardSamplingGeometry(
      boardLeft: (decoded.width - boardSize) * 0.5,
      boardTop: (decoded.height - boardSize) * 0.5,
      cellSize: cellSize,
      inset: inset,
    );
  }

  final double boardLeft;
  final double boardTop;
  final double cellSize;
  final double inset;
}

class _SquarePrediction {
  const _SquarePrediction({
    required this.row,
    required this.col,
    required this.square,
    required this.probs,
    required this.top1,
    required this.top2,
  });

  final int row;
  final int col;
  final String square;
  final List<double> probs;
  final int top1;
  final int top2;

  double get margin => probs[top1] - probs[top2];
}

class _InferenceRun {
  const _InferenceRun({
    required this.predictions,
    required this.preprocessMs,
    required this.invokeMs,
    required this.decodeMs,
    required this.mode,
    required this.fallbackUsed,
  });

  final List<_SquarePrediction> predictions;
  final int preprocessMs;
  final int invokeMs;
  final int decodeMs;
  final String mode;
  final bool fallbackUsed;
}
