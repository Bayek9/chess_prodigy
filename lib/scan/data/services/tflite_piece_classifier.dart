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

    final pieces = <String, ScanPiece>{};
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

        final classId = _argmax(output.first);
        if (classId <= 0 || classId >= _classToPiece.length) {
          continue;
        }

        final piece = _classToPiece[classId];
        if (piece == null) {
          continue;
        }

        final square = _squareMapper.squareAt(row: row, col: col);
        pieces[square] = piece;
      }
    }

    return BoardScanPosition(pieces: pieces);
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

  int _argmax(List<double> values) {
    if (values.isEmpty) {
      return 0;
    }
    int best = 0;
    double bestValue = values[0];
    for (int i = 1; i < values.length; i++) {
      final v = values[i];
      if (v > bestValue) {
        best = i;
        bestValue = v;
      }
    }
    return best;
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
