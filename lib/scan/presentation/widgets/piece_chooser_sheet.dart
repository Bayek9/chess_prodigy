import 'package:flutter/material.dart';

import '../../domain/entities/scan_piece.dart';

@immutable
class PieceChoiceResult {
  const PieceChoiceResult.select(this.piece) : clear = false;

  const PieceChoiceResult.clear() : clear = true, piece = null;

  final bool clear;
  final ScanPiece? piece;
}

class PieceChooserSheet extends StatelessWidget {
  const PieceChooserSheet({super.key, this.currentPiece});

  final ScanPiece? currentPiece;

  static Future<PieceChoiceResult?> show(
    BuildContext context, {
    ScanPiece? currentPiece,
  }) {
    return showModalBottomSheet<PieceChoiceResult>(
      context: context,
      showDragHandle: true,
      builder: (_) => PieceChooserSheet(currentPiece: currentPiece),
    );
  }

  @override
  Widget build(BuildContext context) {
    final white = ScanPiece.palette
        .where((piece) => piece.color == ScanPieceColor.white)
        .toList(growable: false);
    final black = ScanPiece.palette
        .where((piece) => piece.color == ScanPieceColor.black)
        .toList(growable: false);

    return SafeArea(
      child: Padding(
        padding: const EdgeInsets.fromLTRB(16, 8, 16, 20),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Modifier la case',
              style: Theme.of(context).textTheme.titleMedium,
            ),
            const SizedBox(height: 12),
            Wrap(
              spacing: 8,
              runSpacing: 8,
              children: [
                OutlinedButton.icon(
                  onPressed: () =>
                      Navigator.pop(context, const PieceChoiceResult.clear()),
                  icon: const Icon(Icons.delete_outline),
                  label: const Text('Vider'),
                ),
              ],
            ),
            const SizedBox(height: 12),
            Text('Blanc', style: Theme.of(context).textTheme.labelLarge),
            const SizedBox(height: 8),
            Wrap(
              spacing: 8,
              runSpacing: 8,
              children: white
                  .map((piece) => _pieceButton(context, piece))
                  .toList(),
            ),
            const SizedBox(height: 12),
            Text('Noir', style: Theme.of(context).textTheme.labelLarge),
            const SizedBox(height: 8),
            Wrap(
              spacing: 8,
              runSpacing: 8,
              children: black
                  .map((piece) => _pieceButton(context, piece))
                  .toList(),
            ),
          ],
        ),
      ),
    );
  }

  Widget _pieceButton(BuildContext context, ScanPiece piece) {
    final selected = currentPiece == piece;
    return Material(
      color: selected ? Colors.white24 : Colors.white10,
      borderRadius: BorderRadius.circular(10),
      child: InkWell(
        borderRadius: BorderRadius.circular(10),
        onTap: () => Navigator.pop(context, PieceChoiceResult.select(piece)),
        child: SizedBox(
          width: 54,
          height: 54,
          child: Center(
            child: Text(piece.glyph, style: const TextStyle(fontSize: 30)),
          ),
        ),
      ),
    );
  }
}
