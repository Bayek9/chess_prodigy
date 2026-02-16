import 'package:flutter/material.dart';
import 'package:chess/chess.dart' as chesslib;
import 'package:simple_chess_board/simple_chess_board.dart';

class HandlingPromotionsBoard extends StatefulWidget {
  const HandlingPromotionsBoard({super.key});

  @override
  State<HandlingPromotionsBoard> createState() =>
      _HandlingPromotionsBoardState();
}

class _HandlingPromotionsBoardState extends State<HandlingPromotionsBoard> {
  final _chess = chesslib.Chess.fromFEN('1k6/p2KP3/1p6/8/4B3/8/8/8 w - - 0 1');
  final _highlightCells = <String, Color>{};

  void tryMakingMove({required ShortMove move}) {
    final success = _chess.move(<String, String?>{
      'from': move.from,
      'to': move.to,
      'promotion': move.promotion?.name,
    });
    if (success) {
      setState(() {});
    }
  }

  Widget _promotionTile({
    required PieceType pieceType,
    required String label,
  }) {
    return InkWell(
      onTap: () => Navigator.of(context).pop(pieceType),
      child: Padding(
        padding: const EdgeInsets.symmetric(vertical: 8),
        child: Center(
          child: Text(
            label,
            style: const TextStyle(fontSize: 20, fontWeight: FontWeight.w600),
          ),
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Handling promotions'),
      ),
      body: Center(
        child: SizedBox(
          width: 300,
          height: 300,
          child: SimpleChessBoard(
            engineThinking: false,
            fen: _chess.fen,
            onMove: ({required ShortMove move}) {
              debugPrint('${move.from}|${move.to}|${move.promotion}');
            },
            blackSideAtBottom: false,
            whitePlayerType: PlayerType.human,
            blackPlayerType: PlayerType.computer,
            lastMoveToHighlight: null,
            onPromote: () {
              return showDialog<PieceType>(
                context: context,
                builder: (_) {
                  return AlertDialog(
                    title: const Text('Promotion'),
                    content: Column(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        _promotionTile(pieceType: PieceType.queen, label: 'Queen'),
                        _promotionTile(pieceType: PieceType.rook, label: 'Rook'),
                        _promotionTile(pieceType: PieceType.bishop, label: 'Bishop'),
                        _promotionTile(pieceType: PieceType.knight, label: 'Knight'),
                      ],
                    ),
                  );
                },
              );
            },
            cellHighlights: _highlightCells,
            chessBoardColors: ChessBoardColors(),
            onPromotionCommited: ({
              required ShortMove moveDone,
              required PieceType pieceType,
            }) {
              moveDone.promotion = pieceType;
              tryMakingMove(move: moveDone);
            },
            onTap: ({required String cellCoordinate}) {
              if (_highlightCells[cellCoordinate] == null) {
                _highlightCells[cellCoordinate] = Colors.red.withAlpha(70);
                setState(() {});
              } else {
                _highlightCells.remove(cellCoordinate);
                setState(() {});
              }
            },
          ),
        ),
      ),
    );
  }
}
