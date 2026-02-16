import 'dart:math' as math;
import 'package:flutter/material.dart';

enum SuggestionArrowType { straight, knightL }

class SuggestionArrow {
  final String from; // ex: "e2"
  final String to; // ex: "e4"
  final SuggestionArrowType type;
  final Color color;
  final double widthFactor; // 0.10 -> 0.28 recommandé

  const SuggestionArrow({
    required this.from,
    required this.to,
    this.type = SuggestionArrowType.straight,
    this.color = const Color(0xCCF5D142), // jaune translucide type chess.com-like
    this.widthFactor = 0.17,
  });
}

class SuggestionArrowsOverlay extends StatelessWidget {
  final List<SuggestionArrow> arrows;
  final bool blackSideAtBottom;
  final double boardPadding; // si besoin d'aligner (coords zone, etc.)

  const SuggestionArrowsOverlay({
    super.key,
    required this.arrows,
    required this.blackSideAtBottom,
    this.boardPadding = 0,
  });

  @override
  Widget build(BuildContext context) {
    return IgnorePointer(
      child: CustomPaint(
        painter: _SuggestionArrowsPainter(
          arrows: arrows,
          blackSideAtBottom: blackSideAtBottom,
          boardPadding: boardPadding,
        ),
        size: Size.infinite,
      ),
    );
  }
}

class _SuggestionArrowsPainter extends CustomPainter {
  final List<SuggestionArrow> arrows;
  final bool blackSideAtBottom;
  final double boardPadding;

  _SuggestionArrowsPainter({
    required this.arrows,
    required this.blackSideAtBottom,
    required this.boardPadding,
  });

  @override
  void paint(Canvas canvas, Size size) {
    if (arrows.isEmpty) return;

    final boardSize = math.min(size.width, size.height) - (boardPadding * 2);
    if (boardSize <= 0) return;
    final cell = boardSize / 8.0;

    final offsetX = (size.width - boardSize) / 2 + boardPadding;
    final offsetY = (size.height - boardSize) / 2 + boardPadding;

    canvas.save();
    canvas.translate(offsetX, offsetY);

    for (final a in arrows) {
      final fromSq = _parseSquare(a.from);
      final toSq = _parseSquare(a.to);
      if (fromSq == null || toSq == null) continue;

      final fromGrid = _toBoardGrid(fromSq, blackSideAtBottom);
      final toGrid = _toBoardGrid(toSq, blackSideAtBottom);

      final fromCenter = _cellCenter(fromGrid.x, fromGrid.y, cell);
      final toCenter = _cellCenter(toGrid.x, toGrid.y, cell);

      final isKnight = _isKnightMove(fromGrid, toGrid);
      if (a.type == SuggestionArrowType.knightL || isKnight) {
        _drawKnightArrow(
          canvas: canvas,
          start: fromCenter,
          end: toCenter,
          fromGrid: fromGrid,
          toGrid: toGrid,
          cell: cell,
          color: a.color,
          widthFactor: a.widthFactor,
        );
      } else {
        _drawStraightArrow(
          canvas: canvas,
          start: fromCenter,
          end: toCenter,
          cell: cell,
          color: a.color,
          widthFactor: a.widthFactor,
        );
      }
    }

    canvas.restore();
  }

  @override
  bool shouldRepaint(covariant _SuggestionArrowsPainter oldDelegate) {
    if (blackSideAtBottom != oldDelegate.blackSideAtBottom) return true;
    if (boardPadding != oldDelegate.boardPadding) return true;
    if (arrows.length != oldDelegate.arrows.length) return true;
    for (int i = 0; i < arrows.length; i++) {
      final a = arrows[i];
      final b = oldDelegate.arrows[i];
      if (a.from != b.from ||
          a.to != b.to ||
          a.type != b.type ||
          a.color != b.color ||
          a.widthFactor != b.widthFactor) {
        return true;
      }
    }
    return false;
  }

  // ---------- Drawing helpers ----------

  void _drawStraightArrow({
    required Canvas canvas,
    required Offset start,
    required Offset end,
    required double cell,
    required Color color,
    required double widthFactor,
  }) {
    final v = end - start;
    final len = v.distance;
    if (len < 1) return;

    final u = v / len;
    final perp = Offset(-u.dy, u.dx);

    final bodyW = (cell * widthFactor.clamp(0.10, 0.28)).toDouble();
    final headLen = cell * 0.45;
    final headW = cell * 0.42;

    final startP = start + u * (cell * 0.18);
    final tip = end - u * (cell * 0.10);
    final bodyEnd = tip - u * headLen;

    final outline = Paint()
      ..style = PaintingStyle.stroke
      ..strokeCap = StrokeCap.round
      ..strokeJoin = StrokeJoin.round
      ..strokeWidth = bodyW + 2
      ..color = Colors.black.withValues(alpha: 0.22);

    final body = Paint()
      ..style = PaintingStyle.stroke
      ..strokeCap = StrokeCap.round
      ..strokeJoin = StrokeJoin.round
      ..strokeWidth = bodyW
      ..color = color;

    canvas.drawLine(startP, bodyEnd, outline);
    canvas.drawLine(startP, bodyEnd, body);

    final headPath = Path()
      ..moveTo(tip.dx, tip.dy)
      ..lineTo((bodyEnd + perp * (headW / 2)).dx,
          (bodyEnd + perp * (headW / 2)).dy)
      ..lineTo((bodyEnd - perp * (headW / 2)).dx,
          (bodyEnd - perp * (headW / 2)).dy)
      ..close();

    final headOutline = Paint()
      ..style = PaintingStyle.fill
      ..color = Colors.black.withValues(alpha: 0.22);
    final headFill = Paint()
      ..style = PaintingStyle.fill
      ..color = color;

    canvas.drawPath(headPath, headOutline);
    canvas.drawPath(headPath, headFill);
  }

  void _drawKnightArrow({
    required Canvas canvas,
    required Offset start,
    required Offset end,
    required _GridPos fromGrid,
    required _GridPos toGrid,
    required double cell,
    required Color color,
    required double widthFactor,
  }) {
    final dx = toGrid.x - fromGrid.x;
    final dy = toGrid.y - fromGrid.y;

    if (!((dx.abs() == 1 && dy.abs() == 2) || (dx.abs() == 2 && dy.abs() == 1))) {
      _drawStraightArrow(
        canvas: canvas,
        start: start,
        end: end,
        cell: cell,
        color: color,
        widthFactor: widthFactor,
      );
      return;
    }

    // Pivot en "L" : grand segment puis petit segment
    final pivot = (dx.abs() == 1 && dy.abs() == 2)
        ? _cellCenter(
            fromGrid.x,
            toGrid.y,
            cell,
          ) // vertical long puis horizontal court
        : _cellCenter(
            toGrid.x,
            fromGrid.y,
            cell,
          ); // horizontal long puis vertical court

    final seg1 = pivot - start;
    final seg2 = end - pivot;
    final l1 = seg1.distance;
    final l2 = seg2.distance;
    if (l1 < 1 || l2 < 1) return;

    final u1 = seg1 / l1;
    final u2 = seg2 / l2;
    final perp2 = Offset(-u2.dy, u2.dx);

    final bodyW = (cell * widthFactor.clamp(0.10, 0.28)).toDouble();
    final headLen = cell * 0.45;
    final headW = cell * 0.42;

    final startP = start + u1 * (cell * 0.18);
    final tip = end - u2 * (cell * 0.10);
    final beforeTip = tip - u2 * headLen;

    final path = Path()
      ..moveTo(startP.dx, startP.dy)
      ..lineTo(pivot.dx, pivot.dy)
      ..lineTo(beforeTip.dx, beforeTip.dy);

    final outline = Paint()
      ..style = PaintingStyle.stroke
      ..strokeCap = StrokeCap.round
      ..strokeJoin = StrokeJoin.round
      ..strokeWidth = bodyW + 2
      ..color = Colors.black.withValues(alpha: 0.22);

    final body = Paint()
      ..style = PaintingStyle.stroke
      ..strokeCap = StrokeCap.round
      ..strokeJoin = StrokeJoin.round
      ..strokeWidth = bodyW
      ..color = color;

    canvas.drawPath(path, outline);
    canvas.drawPath(path, body);

    final headPath = Path()
      ..moveTo(tip.dx, tip.dy)
      ..lineTo((beforeTip + perp2 * (headW / 2)).dx,
          (beforeTip + perp2 * (headW / 2)).dy)
      ..lineTo((beforeTip - perp2 * (headW / 2)).dx,
          (beforeTip - perp2 * (headW / 2)).dy)
      ..close();

    final headOutline = Paint()
      ..style = PaintingStyle.fill
      ..color = Colors.black.withValues(alpha: 0.22);
    final headFill = Paint()
      ..style = PaintingStyle.fill
      ..color = color;

    canvas.drawPath(headPath, headOutline);
    canvas.drawPath(headPath, headFill);
  }

  // ---------- Geometry helpers ----------

  _Square? _parseSquare(String s) {
    if (s.length != 2) return null;
    final f = s.codeUnitAt(0) - 97; // a=0
    final r = int.tryParse(s[1]);
    if (r == null) return null;
    final rank = r - 1; // "1" => 0
    if (f < 0 || f > 7 || rank < 0 || rank > 7) return null;
    return _Square(file: f, rank: rank);
  }

  _GridPos _toBoardGrid(_Square sq, bool blackBottom) {
    // Coordonnées board écran
    final x = blackBottom ? (7 - sq.file) : sq.file;
    final y = blackBottom ? sq.rank : (7 - sq.rank);
    return _GridPos(x, y);
  }

  Offset _cellCenter(int x, int y, double cell) {
    return Offset((x + 0.5) * cell, (y + 0.5) * cell);
  }

  bool _isKnightMove(_GridPos a, _GridPos b) {
    final dx = (a.x - b.x).abs();
    final dy = (a.y - b.y).abs();
    return (dx == 1 && dy == 2) || (dx == 2 && dy == 1);
  }
}

class _Square {
  final int file; // 0..7
  final int rank; // 0..7 (1=>0)
  _Square({required this.file, required this.rank});
}

class _GridPos {
  final int x; // 0..7 écran
  final int y; // 0..7 écran
  _GridPos(this.x, this.y);
}
