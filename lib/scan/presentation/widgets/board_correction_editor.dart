import 'package:flutter/material.dart';

import '../../domain/entities/board_scan_position.dart';
import '../../domain/services/grid_square_mapper.dart';

class BoardCorrectionEditor extends StatelessWidget {
  const BoardCorrectionEditor({
    super.key,
    required this.position,
    required this.squareMapper,
    required this.flipped,
    required this.onSquareTap,
  });

  final BoardScanPosition position;
  final GridSquareMapper squareMapper;
  final bool flipped;
  final ValueChanged<String> onSquareTap;

  static const Color _light = Color(0xFFEDD6AF);
  static const Color _dark = Color(0xFFB88761);

  @override
  Widget build(BuildContext context) {
    return AspectRatio(
      aspectRatio: 1,
      child: ClipRRect(
        borderRadius: BorderRadius.circular(12),
        child: Column(
          children: [
            for (int row = 0; row < 8; row++)
              Expanded(
                child: Row(
                  children: [
                    for (int col = 0; col < 8; col++)
                      Expanded(
                        child: _SquareCell(
                          color: ((row + col) % 2 == 0) ? _light : _dark,
                          square: squareMapper.squareAt(
                            row: row,
                            col: col,
                            flipped: flipped,
                          ),
                          pieceGlyph:
                              position
                                  .pieceAt(
                                    squareMapper.squareAt(
                                      row: row,
                                      col: col,
                                      flipped: flipped,
                                    ),
                                  )
                                  ?.glyph ??
                              '',
                          onTap: onSquareTap,
                        ),
                      ),
                  ],
                ),
              ),
          ],
        ),
      ),
    );
  }
}

class _SquareCell extends StatelessWidget {
  const _SquareCell({
    required this.color,
    required this.square,
    required this.pieceGlyph,
    required this.onTap,
  });

  final Color color;
  final String square;
  final String pieceGlyph;
  final ValueChanged<String> onTap;

  @override
  Widget build(BuildContext context) {
    return Material(
      color: color,
      child: InkWell(
        onTap: () => onTap(square),
        child: Stack(
          children: [
            Center(
              child: Text(pieceGlyph, style: const TextStyle(fontSize: 28)),
            ),
            Positioned(
              right: 3,
              bottom: 2,
              child: Text(
                square,
                style: TextStyle(
                  fontSize: 9,
                  color: Colors.black.withValues(alpha: 0.35),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
