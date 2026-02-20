import 'dart:async';
import 'dart:math' as math;

import 'package:chess/chess.dart' as chess;
import 'package:flutter/material.dart';
import 'package:simple_chess_board/simple_chess_board.dart';
import 'package:chess_prodigy/widgets/elo_slider.dart';
import 'package:chess_prodigy/widgets/suggestion_arrows_overlay.dart';

import 'engine/chess_engine.dart';
import 'engine/chess_engine_factory.dart';

void main() {
  runApp(const ChessProdigyApp());
}

class ChessProdigyApp extends StatelessWidget {
  const ChessProdigyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'Chess Prodigy',
      theme: ThemeData.dark(),
      home: const ChessHomePage(),
    );
  }
}

class ChessHomePage extends StatefulWidget {
  const ChessHomePage({super.key});

  @override
  State<ChessHomePage> createState() => _ChessHomePageState();
}

class _ChessHomePageState extends State<ChessHomePage> {
  final chess.Chess _game = chess.Chess();
  late final ChessEngine _engine;

  String _fen = chess.Chess.DEFAULT_POSITION;
  BoardArrow? _lastMoveArrow;
  final List<SuggestionArrow> _suggestionArrows = const [];
  bool _thinking = false;
  bool _engineReady = false;
  int _engineElo = 1200;
  static const List<int> _eloTicks = <int>[250, 1000, 1600, 2000, 2400, 3200];
  Timer? _eloDebounce;
  Timer? _aiDebounceTimer;
  int _aiRequestToken = 0;
  final math.Random _rng = math.Random();

  @override
  void initState() {
    super.initState();
    _fen = _game.fen;
    _engine = createChessEngine();
    _initEngine();
  }

  Future<void> _initEngine() async {
    try {
      await _engine.init();
      await _engine.setPosition(_game.fen);
      await _engine.setTargetElo(_engineElo);
      if (mounted) {
        setState(() {
          _engineReady = true;
        });
      }
    } catch (_) {
      if (mounted) {
        setState(() {
          _engineReady = false;
        });
      }
    }
  }

  bool _isBlackTurn() => _game.turn == chess.Color.BLACK;

  String? _pieceTypeToPromotion(PieceType? pieceType) {
    if (pieceType == null) return null;
    switch (pieceType) {
      case PieceType.queen:
        return 'q';
      case PieceType.rook:
        return 'r';
      case PieceType.bishop:
        return 'b';
      case PieceType.knight:
        return 'n';
      default:
        return null;
    }
  }

  bool _applyMove({
    required String from,
    required String to,
    PieceType? promotionPiece,
    String? promotionUci,
  }) {
    final promotion = promotionUci ?? _pieceTypeToPromotion(promotionPiece);
    final move = <String, dynamic>{
      'from': from,
      'to': to,
    };
    if (promotion != null) {
      move['promotion'] = promotion;
    }

    final result = _game.move(move);
    if (!result) return false;

    setState(() {
      _fen = _game.fen;
      _lastMoveArrow = BoardArrow(from: from, to: to);
    });
    return true;
  }

  Future<void> _requestAiMove({required int token}) async {
    if (token != _aiRequestToken || _thinking || !_isBlackTurn() || _game.game_over) return;

    // On bloque les actions humaines, mais on ne rallonge pas le calcul moteur.
    setState(() {
      _thinking = true;
    });

    // Délai esthétique "humain" : 1 à 2 secondes visibles après ton coup.
    // IMPORTANT : on ne change PAS bestMove(700) => pas d'impact sur le niveau.
    // _queueAiMove attend déjà 280ms, donc on vise 700..1700ms ici => ~980..1980ms visibles.
    final sw = Stopwatch()..start();
    final targetMs = 700 + _rng.nextInt(1001); // 700..1700

    String? uciMove;
    if (_engineReady) {
      try {
        await _engine.setPosition(_game.fen);
        uciMove = await _engine.bestMove(700); // on garde 700ms = même force
      } catch (_) {
        uciMove = null;
      }
    }

    if (!mounted) return;

    if (token != _aiRequestToken) {
      setState(() => _thinking = false);
      return;
    }

    // Attente esthétique (sans effet sur la force)
    final remainingMs = targetMs - sw.elapsedMilliseconds;
    if (remainingMs > 0) {
      await Future.delayed(Duration(milliseconds: remainingMs));
    }

    if (!mounted) return;
    if (token != _aiRequestToken) {
      setState(() => _thinking = false);
      return;
    }

    if (uciMove != null && uciMove.length >= 4 && _isBlackTurn() && !_game.game_over) {
      final from = uciMove.substring(0, 2);
      final to = uciMove.substring(2, 4);
      final promotion = uciMove.length > 4 ? uciMove[4] : null;
      _applyMove(from: from, to: to, promotionUci: promotion);
    }

    if (mounted) {
      setState(() => _thinking = false);
    }
  }

  void _queueAiMove() {
    _aiDebounceTimer?.cancel(); // annule l'ancienne demande
    final token = ++_aiRequestToken;

    _aiDebounceTimer = Timer(const Duration(milliseconds: 280), () {
      if (!mounted || _game.game_over) return;
      unawaited(_requestAiMove(token: token));
    });
  }

  Future<void> _onHumanMove(ShortMove move) async {
    if (_thinking || _game.game_over || _game.turn != chess.Color.WHITE) {
      return;
    }

    final applied = _applyMove(
      from: move.from,
      to: move.to,
      promotionPiece: move.promotion,
    );
    if (!applied || _game.game_over) return;
    _queueAiMove();
  }

  Future<void> _newGame() async {
    // Annule les timers (IA / elo debounce)
    _aiDebounceTimer?.cancel();
    _eloDebounce?.cancel();

    // Invalide toute requÃªte IA en cours (si un bestMove revient, il sera ignorÃ©)
    _aiRequestToken++;

    // Reset logique du jeu (revient Ã  la position initiale)
    _game.reset();
    final startFen = _game.fen;

    if (!mounted) return;

    // Reset UI
    setState(() {
      _fen = startFen;
      _lastMoveArrow = null;
      _thinking = false;
    });

    // Reset moteur (position de dÃ©part)
    if (_engineReady) {
      try {
        await _engine.newGame();
        await _engine.setPosition(startFen);
        // optionnel: rÃ©-applique l'Elo actuel (pas obligatoire si tu l'as deja)
        await _engine.setTargetElo(_engineElo);
      } catch (_) {
        // ignore
      }
    }
  }

  @override
  void dispose() {
    _eloDebounce?.cancel();
    _aiDebounceTimer?.cancel();
    unawaited(_engine.dispose());
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Chess Prodigy'),
        actions: [
          IconButton(
            tooltip: 'Nouvelle partie',
            icon: const Icon(Icons.refresh),
            onPressed: _newGame,
          ),
        ],
      ),
      backgroundColor: const Color(0xFF1E1E1E),
      body: SafeArea(
        child: Column(
          children: [
            Padding(
              padding: const EdgeInsets.fromLTRB(12, 8, 12, 4),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text('Niveau moteur: $_engineElo'),
                  EloSlider(
                    value: _engineElo,
                    marks: _eloTicks,
                    onChanged: (v) {
                      setState(() => _engineElo = v);

                      if (_engineReady) {
                        _eloDebounce?.cancel();
                        _eloDebounce = Timer(const Duration(milliseconds: 180), () async {
                          await _engine.setTargetElo(_engineElo);
                        });
                      }
                    },
                    onChangeEnd: (v) async {
                      setState(() => _engineElo = v);
                      _eloDebounce?.cancel();
                      if (_engineReady) {
                        await _engine.setTargetElo(v.toInt());
                      }
                    },
                  ),
                ],
              ),
            ),
            Expanded(
              child: BoardView(
                fen: _fen,
                lastMoveToHighlight: _lastMoveArrow,
                suggestionArrows: _suggestionArrows,
                thinking: _thinking,
                blackSideAtBottom: false,
                onMove: _onHumanMove,
                onPromote: () async => PieceType.queen,
                onPromotionCommited: ({
                  required ShortMove moveDone,
                  required PieceType pieceType,
                }) {
                  if (_thinking || _game.game_over) return;
                  final applied = _applyMove(
                    from: moveDone.from,
                    to: moveDone.to,
                    promotionPiece: pieceType,
                  );
                  if (!applied || _game.game_over) return;
                  _queueAiMove();
                },
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class BoardView extends StatelessWidget {
  const BoardView({
    super.key,
    required this.fen,
    required this.lastMoveToHighlight,
    required this.suggestionArrows,
    required this.thinking,
    required this.onMove,
    required this.blackSideAtBottom,
    required this.onPromote,
    required this.onPromotionCommited,
  });

  final String fen;
  final BoardArrow? lastMoveToHighlight;
  final List<SuggestionArrow> suggestionArrows;
  final bool thinking;
  final bool blackSideAtBottom;
  final void Function(ShortMove move) onMove;
  final Future<PieceType?> Function() onPromote;
  final void Function({
    required ShortMove moveDone,
    required PieceType pieceType,
  }) onPromotionCommited;

  static const Color _light = Color(0xFFEDD6AF);
  static const Color _dark = Color(0xFFB88761);

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(
      builder: (context, c) {
        final boardSize = math.min(c.maxWidth, c.maxHeight);

        return Center(
          child: SizedBox.square(
            dimension: boardSize,
            child: Stack(
              children: [
                SimpleChessBoard(
                  fen: fen,
                  blackSideAtBottom: blackSideAtBottom,
                  whitePlayerType: PlayerType.human,
                  blackPlayerType: PlayerType.human,
                  onMove: ({required ShortMove move}) => onMove(move),
                  onPromote: onPromote,
                  onPromotionCommited: onPromotionCommited,
                  showCoordinatesZone: false,
                  engineThinking: false,
                  highlightLastMoveSquares: true,
                  showPossibleMoves: true,
                  normalMoveIndicatorBuilder: (cellSize) => Center(
                    child: Container(
                      width: cellSize * 0.22,
                      height: cellSize * 0.22,
                      decoration: BoxDecoration(
                        shape: BoxShape.circle,
                        color: Colors.grey.withAlpha(95),
                      ),
                    ),
                  ),
                  captureMoveIndicatorBuilder: (cellSize) => Center(
                    child: Container(
                      width: cellSize * 0.78,
                      height: cellSize * 0.78,
                      decoration: BoxDecoration(
                        shape: BoxShape.circle,
                        border: Border.all(
                          color: Colors.grey.withAlpha(135),
                          width: cellSize * 0.065,
                        ),
                      ),
                    ),
                  ),
                  lastMoveToHighlight: lastMoveToHighlight,
                  cellHighlights: const <String, Color>{},
                  chessBoardColors: ChessBoardColors()
                    ..lightSquaresColor = _light
                    ..darkSquaresColor = _dark
                    ..startSquareColor = Colors.transparent
                    ..endSquareColor = Colors.transparent
                    ..lastMoveArrowColor = Colors.transparent
                    ..coordinatesZoneColor = Colors.transparent
                    ..coordinatesColor = Colors.transparent,
                  onTap: ({required String cellCoordinate}) {},
                ),
                Positioned.fill(
                  child: SuggestionArrowsOverlay(
                    arrows: suggestionArrows,
                    blackSideAtBottom: blackSideAtBottom,
                    boardPadding: 0,
                  ),
                ),
                IgnorePointer(
                  child: _InsideCoords(
                    boardSize: boardSize,
                    blackSideAtBottom: blackSideAtBottom,
                    lightSquareColor: _light,
                    darkSquareColor: _dark,
                  ),
                ),
              ],
            ),
          ),
        );
      },
    );
  }
}

class _InsideCoords extends StatelessWidget {
  const _InsideCoords({
    required this.boardSize,
    required this.blackSideAtBottom,
    required this.lightSquareColor,
    required this.darkSquareColor,
  });

  final double boardSize;
  final bool blackSideAtBottom;
  final Color lightSquareColor;
  final Color darkSquareColor;

  bool _isDarkSquare(int col, int rowFromTop) {
    return ((col + rowFromTop) % 2) == 1;
  }

  Color _labelColorForSquare(int col, int rowFromTop) {
    final dark = _isDarkSquare(col, rowFromTop);
    return (dark ? lightSquareColor : darkSquareColor).withValues(alpha: 0.98);
  }

  @override
  Widget build(BuildContext context) {
    final cell = boardSize / 8.0;

    final files = blackSideAtBottom
        ? const ['h', 'g', 'f', 'e', 'd', 'c', 'b', 'a']
        : const ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'];

    final ranksTopToBottom = blackSideAtBottom
        ? const ['1', '2', '3', '4', '5', '6', '7', '8']
        : const ['8', '7', '6', '5', '4', '3', '2', '1'];

    final fontSize = math.max(9.0, cell * 0.11);

    return Stack(
      children: [
        for (int row = 0; row < 8; row++)
          Positioned(
            left: 3,
            top: row * cell + 2,
            child: Text(
              ranksTopToBottom[row],
              style: TextStyle(
                fontSize: fontSize,
                fontWeight: FontWeight.w600,
                height: 1,
                color: _labelColorForSquare(0, row),
              ),
            ),
          ),
        for (int col = 0; col < 8; col++)
          Positioned(
            left: col * cell + cell - (cell * 0.14),
            bottom: 2,
            child: Text(
              files[col],
              style: TextStyle(
                fontSize: fontSize,
                fontWeight: FontWeight.w600,
                height: 1,
                color: _labelColorForSquare(col, 7),
              ),
            ),
          ),
      ],
    );
  }
}

