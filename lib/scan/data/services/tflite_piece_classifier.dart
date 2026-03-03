import 'dart:math' as math;
import 'dart:ui' as ui;

import 'package:flutter/foundation.dart';
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
    this.threads = 2,
    this.cropInsetFraction = 0.08,
  });

  final String modelAssetPath;
  final int inputSize;
  final int threads;
  final double cropInsetFraction;

  Interpreter? _interpreter;
  bool _loadAttempted = false;
  String? _loadError;

  static const GridSquareMapper _squareMapper = GridSquareMapper();

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

    final predictions = <_SquarePrediction>[];
    for (int row = 0; row < 8; row++) {
      for (int col = 0; col < 8; col++) {
        final input = _squareToInputTensor(
          decoded: decoded,
          row: row,
          col: col,
          inputSize: inputSize,
          cropInsetFraction: cropInsetFraction,
        );

        final output = List<List<double>>.generate(
          1,
          (_) =>
              List<double>.filled(_classToPiece.length, 0.0, growable: false),
          growable: false,
        );
        interpreter.run(input, output);

        final probs = List<double>.from(output.first, growable: false);
        final topIds = _topClassIds(probs, count: 2);
        final top1 = topIds.isEmpty ? _emptyId : topIds.first;
        final top2 = topIds.length > 1 ? topIds[1] : top1;
        predictions.add(
          _SquarePrediction(
            row: row,
            col: col,
            square: _squareMapper.squareAt(row: row, col: col),
            probs: probs,
            top1: top1,
            top2: top2,
          ),
        );
      }
    }

    final selected = predictions
        .map((prediction) => prediction.top1)
        .toList(growable: false);
    final constraintsChanges = _applyChessConstraints(predictions, selected);

    final pieces = <String, ScanPiece>{};
    for (int i = 0; i < predictions.length; i++) {
      final classId = selected[i];
      if (classId == _emptyId) {
        continue;
      }
      final piece = _classToPiece[classId];
      if (piece == null) {
        continue;
      }
      pieces[predictions[i].square] = piece;
    }

    if (!kReleaseMode && constraintsChanges > 0) {
      debugPrint(
        '[scan][piece_tflite] constraints_changes=$constraintsChanges',
      );
    }

    return BoardScanPosition(pieces: pieces);
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

  List<int> _topClassIds(List<double> probs, {required int count}) {
    final ids = List<int>.generate(probs.length, (index) => index);
    ids.sort((a, b) => probs[b].compareTo(probs[a]));
    return ids.take(count).toList(growable: false);
  }

  Future<void> _ensureLoaded() async {
    if (_loadAttempted) {
      return;
    }
    _loadAttempted = true;
    try {
      final options = InterpreterOptions()..threads = threads;
      _interpreter = await Interpreter.fromAsset(
        modelAssetPath,
        options: options,
      );
    } catch (e) {
      _loadError = e.toString();
      _interpreter = null;
    }
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

  List<List<List<List<double>>>> _squareToInputTensor({
    required _DecodedRgbaImage decoded,
    required int row,
    required int col,
    required int inputSize,
    required double cropInsetFraction,
  }) {
    final boardSize = math.min(decoded.width, decoded.height).toDouble();
    final boardLeft = (decoded.width - boardSize) * 0.5;
    final boardTop = (decoded.height - boardSize) * 0.5;
    final cellSize = boardSize / 8.0;

    final inset = (cellSize * cropInsetFraction).clamp(0.0, cellSize * 0.35);
    final left = boardLeft + (col * cellSize) + inset;
    final top = boardTop + (row * cellSize) + inset;
    final right = boardLeft + ((col + 1) * cellSize) - inset;
    final bottom = boardTop + ((row + 1) * cellSize) - inset;

    final sampledWidth = math.max(1.0, right - left);
    final sampledHeight = math.max(1.0, bottom - top);

    final image = List<List<List<double>>>.generate(
      inputSize,
      (_) => List<List<double>>.generate(
        inputSize,
        (_) => List<double>.filled(3, 0.0, growable: false),
        growable: false,
      ),
      growable: false,
    );

    for (int y = 0; y < inputSize; y++) {
      final srcY = top + (((y + 0.5) * sampledHeight) / inputSize);
      for (int x = 0; x < inputSize; x++) {
        final srcX = left + (((x + 0.5) * sampledWidth) / inputSize);
        final pixel = _sampleRgbBilinear(
          rgba: decoded.rgba,
          width: decoded.width,
          height: decoded.height,
          x: srcX,
          y: srcY,
        );
        image[y][x][0] = pixel[0] / 255.0;
        image[y][x][1] = pixel[1] / 255.0;
        image[y][x][2] = pixel[2] / 255.0;
      }
    }

    return <List<List<List<double>>>>[image];
  }

  List<int> _sampleRgbBilinear({
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

    return <int>[sample(0), sample(1), sample(2)];
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
