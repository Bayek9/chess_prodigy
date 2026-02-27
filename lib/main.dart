import 'dart:async';
import 'dart:math' as math;
import 'dart:ui' show FontVariation, ImageFilter;

import 'package:chess/chess.dart' as chess;
import 'package:colorful_iconify_flutter/icons/fluent_emoji_flat.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:iconify_flutter/iconify_flutter.dart';
import 'package:lucide_icons_flutter/lucide_icons.dart';
import 'package:material_symbols_icons/symbols.dart';
import 'package:simple_chess_board/simple_chess_board.dart';
import 'package:chess_prodigy/scan/presentation/scan_page.dart';
import 'package:chess_prodigy/widgets/elo_slider.dart';
import 'package:chess_prodigy/widgets/suggestion_arrows_overlay.dart';

import 'engine/chess_engine.dart';
import 'engine/chess_engine_factory.dart';

const String _solarFlameBoldDuotoneSvg =
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path fill="currentColor" d="M20 15c0 4.255-2.618 6.122-4.641 6.751a.44.44 0 0 1-.233.012c-.289-.069-.432-.453-.224-.751c.88-1.266 1.898-3.196 1.898-5.012c0-1.95-1.644-4.253-2.928-5.674c-.293-.324-.805-.11-.821.328c-.053 1.45-.282 3.388-1.268 4.908a.412.412 0 0 1-.677.036c-.308-.39-.616-.871-.924-1.252c-.166-.204-.466-.207-.657-.026c-.747.707-1.792 1.809-1.792 3.18c0 .93.36 1.905.767 2.69c.202.39-.103.851-.482.77a.5.5 0 0 1-.122-.046C6.113 19.98 4 18.084 4 15c0-3.146 4.31-7.505 5.956-11.623c.26-.65 1.06-.955 1.617-.531C14.943 5.414 20 10.378 20 15"/><path fill="currentColor" d="M7.733 17.5c0 .93.36 1.905.767 2.69c.202.39-.103.852-.482.77c.482.54 3.658.957 7.108.803c-.289-.069-.432-.453-.224-.751c.88-1.265 1.898-3.196 1.898-5.012c0-1.95-1.644-4.253-2.928-5.674c-.293-.324-.805-.11-.821.328c-.053 1.45-.282 3.388-1.268 4.908a.412.412 0 0 1-.677.036c-.308-.39-.616-.871-.924-1.251c-.166-.205-.466-.208-.657-.027c-.747.707-1.792 1.809-1.792 3.18" opacity=".5"/></svg>';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  SystemChrome.setEnabledSystemUIMode(SystemUiMode.edgeToEdge);
  SystemChrome.setSystemUIOverlayStyle(
    const SystemUiOverlayStyle(
      statusBarColor: Colors.transparent,
      systemNavigationBarColor: Colors.transparent,
      systemNavigationBarDividerColor: Colors.transparent,
      statusBarIconBrightness: Brightness.light,
      statusBarBrightness: Brightness.dark,
      systemNavigationBarIconBrightness: Brightness.light,
      systemStatusBarContrastEnforced: false,
      systemNavigationBarContrastEnforced: false,
    ),
  );
  runApp(const ChessProdigyApp());
}

class ChessProdigyApp extends StatelessWidget {
  const ChessProdigyApp({super.key});

  @override
  Widget build(BuildContext context) {
    const overlayStyle = SystemUiOverlayStyle(
      statusBarColor: Colors.transparent,
      systemNavigationBarColor: Colors.transparent,
      systemNavigationBarDividerColor: Colors.transparent,
      statusBarIconBrightness: Brightness.light,
      statusBarBrightness: Brightness.dark,
      systemNavigationBarIconBrightness: Brightness.light,
      systemStatusBarContrastEnforced: false,
      systemNavigationBarContrastEnforced: false,
    );

    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'Chess Prodigy',
      theme: ThemeData.dark(),
      builder: (context, child) {
        return AnnotatedRegion<SystemUiOverlayStyle>(
          value: overlayStyle,
          child: child ?? const SizedBox.shrink(),
        );
      },
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
  Map<String, Color> _cellHighlights = <String, Color>{};
  int _engineElo = 1200;
  static const List<int> _eloTicks = <int>[250, 1000, 1600, 2000, 2400, 3200];
  Timer? _eloDebounce;
  Timer? _aiDebounceTimer;
  int _aiRequestToken = 0;
  final math.Random _rng = math.Random();
  int _tabIndex = 0;
  final int _streakDays = 9; // temporaire

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
    final move = <String, dynamic>{'from': from, 'to': to};
    if (promotion != null) {
      move['promotion'] = promotion;
    }

    final result = _game.move(move);
    if (!result) return false;

    setState(() {
      _fen = _game.fen;
      _lastMoveArrow = BoardArrow(from: from, to: to);
      _cellHighlights = _computeMateHighlights(_fen);
    });
    return true;
  }

  String? _findKingSquareFromFen(String fen, {required bool white}) {
    final boardPart = fen.split(' ').first;
    final target = white ? 'K' : 'k';

    final ranks = boardPart.split('/');
    for (int r = 0; r < 8; r++) {
      int file = 0;
      for (final ch in ranks[r].split('')) {
        final digit = int.tryParse(ch);
        if (digit != null) {
          file += digit;
          continue;
        }
        if (ch == target) {
          final rankNumber = 8 - r; // r=0 => rank 8
          final fileChar = String.fromCharCode('a'.codeUnitAt(0) + file);
          return '$fileChar$rankNumber';
        }
        file += 1;
      }
    }
    return null;
  }

  Map<String, Color> _computeMateHighlights(String fen) {
    if (!_game.in_checkmate) return <String, Color>{};

    // AprÃƒÆ’Ã‚Â¨s le coup gagnant, cÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢est AU TOUR du perdant => cÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢est lui qui est mat.
    final turn = fen.split(' ')[1]; // 'w' ou 'b'
    final loserIsWhite = turn == 'w';

    final winnerSq = _findKingSquareFromFen(fen, white: !loserIsWhite);

    final m = <String, Color>{};
    if (winnerSq != null) m[winnerSq] = Colors.lightGreenAccent;
    return m;
  }

  Future<void> _requestAiMove({required int token}) async {
    if (token != _aiRequestToken ||
        _thinking ||
        !_isBlackTurn() ||
        _game.game_over) {
      return;
    }

    // On bloque les actions humaines, mais on ne rallonge pas le calcul moteur.
    setState(() {
      _thinking = true;
    });

    // DÃƒÆ’Ã‚Â©lai esthÃƒÆ’Ã‚Â©tique "humain" : 1 ÃƒÆ’Ã‚Â  2 secondes visibles aprÃƒÆ’Ã‚Â¨s ton coup.
    // IMPORTANT : on ne change PAS bestMove(700) => pas d'impact sur le niveau.
    // _queueAiMove attend dÃƒÆ’Ã‚Â©jÃƒÆ’Ã‚Â  280ms, donc on vise 700..1700ms ici => ~980..1980ms visibles.
    final sw = Stopwatch()..start();
    final targetMs = 700 + _rng.nextInt(1001); // 700..1700

    String? uciMove;
    if (_engineReady) {
      try {
        await _engine.setPosition(_game.fen);
        uciMove = await _engine.bestMove(700); // on garde 700ms = mÃƒÆ’Ã‚Âªme force
      } catch (_) {
        uciMove = null;
      }
    }

    if (!mounted) return;

    if (token != _aiRequestToken) {
      setState(() => _thinking = false);
      return;
    }

    // Attente esthÃƒÆ’Ã‚Â©tique (sans effet sur la force)
    final remainingMs = targetMs - sw.elapsedMilliseconds;
    if (remainingMs > 0) {
      await Future.delayed(Duration(milliseconds: remainingMs));
    }

    if (!mounted) return;
    if (token != _aiRequestToken) {
      setState(() => _thinking = false);
      return;
    }

    if (uciMove != null &&
        uciMove.length >= 4 &&
        _isBlackTurn() &&
        !_game.game_over) {
      final from = uciMove.substring(0, 2);
      final to = uciMove.substring(2, 4);
      final promotion = uciMove.length > 4 ? 'q' : null;
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

  MaterialPageRoute<void> _buildGamePageRoute() {
    return MaterialPageRoute<void>(
      builder: (_) => GamePage(
        engineElo: _engineElo,
        eloTicks: _eloTicks,
        engineReady: _engineReady,
        onEloChanged: (v) {
          setState(() => _engineElo = v);
          if (_engineReady) {
            _eloDebounce?.cancel();
            _eloDebounce = Timer(const Duration(milliseconds: 180), () async {
              await _engine.setTargetElo(_engineElo);
            });
          }
        },
        onEloChangeEnd: (v) async {
          setState(() => _engineElo = v);
          _eloDebounce?.cancel();
          if (_engineReady) {
            await _engine.setTargetElo(v);
          }
        },
        fen: _fen,
        lastMoveToHighlight: _lastMoveArrow,
        suggestionArrows: _suggestionArrows,
        cellHighlights: _cellHighlights,
        thinking: _thinking,
        onMove: _onHumanMove,
        onPromote: () async => PieceType.queen,
        onPromotionCommited:
            ({required ShortMove moveDone, required PieceType pieceType}) {
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
    );
  }

  void _openGamePage() {
    Navigator.push(context, _buildGamePageRoute());
  }

  Future<void> _openAnalysisFromFen(String fen) async {
    final validation = chess.Chess.validate_fen(fen);
    final isValid = validation['valid'] == true;
    if (!isValid) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('FEN invalide depuis le scan.')),
      );
      return;
    }

    _aiDebounceTimer?.cancel();
    _aiRequestToken++;
    final loaded = _game.load(fen);
    if (!loaded) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Impossible de charger cette position.')),
      );
      return;
    }

    if (_engineReady) {
      try {
        await _engine.newGame();
        await _engine.setPosition(_game.fen);
        await _engine.setTargetElo(_engineElo);
      } catch (_) {
        // Keep app usable even if engine refresh fails.
      }
    }

    if (!mounted) return;

    setState(() {
      _fen = _game.fen;
      _lastMoveArrow = null;
      _thinking = false;
      _cellHighlights = _computeMateHighlights(_fen);
    });

    _openGamePage();
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
    final topInset = MediaQuery.viewPaddingOf(context).top;
    const topCellToolbarHeight = 48.0;
    const topBlurHeight = 48.0;

    return Scaffold(
      extendBody: true,
      extendBodyBehindAppBar: true,
      appBar: PreferredSize(
        preferredSize: Size.fromHeight(topCellToolbarHeight + topInset),
        child: Stack(
          children: [
            SizedBox(
              height: topInset + topBlurHeight,
              width: double.infinity,
              child: BlurOnly(
                sigma: 16,
                child: const ColoredBox(color: Colors.transparent),
              ),
            ),
            Column(
              children: [
                SizedBox(height: topInset),
                SizedBox(
                  height: topCellToolbarHeight,
                  child: AppBar(
                    primary: false,
                    toolbarHeight: topCellToolbarHeight,
                    backgroundColor: Colors.transparent,
                    elevation: 0,
                    scrolledUnderElevation: 0,
                    shadowColor: Colors.transparent,
                    surfaceTintColor: null,
                    forceMaterialTransparency: true,
                    leading: _PulseIconButton(
                      tooltip: 'Profil',
                      icon: Transform.translate(
                        offset: const Offset(2, 0),
                        child: const Icon(Icons.person_outline),
                      ),
                      onPressed: () {
                        Navigator.push(
                          context,
                          MaterialPageRoute(
                            builder: (_) => const ProfilePage(),
                          ),
                        );
                      },
                    ),
                    title: Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        const SizedBox(
                          width: 27,
                          height: 27,
                          child: Stack(
                            alignment: Alignment.center,
                            children: [
                              Iconify(
                                _solarFlameBoldDuotoneSvg,
                                size: 26,
                                color: Color(0xFFD92B2B),
                              ),
                            ],
                          ),
                        ),
                        const SizedBox(width: 6),
                        Text(
                          '$_streakDays',
                          style: const TextStyle(
                            fontSize: 16,
                            fontWeight: FontWeight.w900,
                            letterSpacing: -0.2,
                          ),
                        ),
                      ],
                    ),
                    actions: [
                      _PulseIconButton(
                        tooltip: 'Scan',
                        icon: const Icon(Symbols.eye_tracking, weight: 600),
                        onPressed: () {
                          Navigator.push(
                            context,
                            MaterialPageRoute(
                              builder: (_) => ScanPage(
                                onAnalyzeFen: (fen) {
                                  unawaited(_openAnalysisFromFen(fen));
                                },
                              ),
                            ),
                          );
                        },
                      ),
                      _PulseIconButton(
                        tooltip: 'Param\u00E8tres',
                        icon: Transform.translate(
                          offset: const Offset(-4, 0),
                          child: Iconify(FluentEmojiFlat.gear, size: 24),
                        ),
                        onPressed: () {
                          Navigator.push(
                            context,
                            MaterialPageRoute(
                              builder: (_) => const SettingsPage(),
                            ),
                          );
                        },
                      ),
                    ],
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
      backgroundColor: const Color(0xFF282725),
      body: IndexedStack(
        index: _tabIndex,
        children: const [
          AccueilPage(),
          ProblemePage(),
          EntrainementPage(),
          ApprendrePage(),
        ],
      ),
      bottomNavigationBar: BottomGlassBar(
        index: _tabIndex,
        onTab: (i) => setState(() => _tabIndex = i),
        onPlay: () {
          _openGamePage();
        },
        onAnalyse: () {
          // TODO: aller sur ton flow "Analyse"
        },
      ),
    );
  }
}

class AccueilPage extends StatelessWidget {
  const AccueilPage({super.key});

  @override
  Widget build(BuildContext context) {
    final topInset = MediaQuery.viewPaddingOf(context).top;
    final bottomInset = MediaQuery.viewPaddingOf(context).bottom;
    const topCellToolbarHeight = 48.0;

    return Stack(
      children: [
        Positioned.fill(
          child: Container(
            decoration: const BoxDecoration(
              gradient: LinearGradient(
                begin: Alignment.topCenter,
                end: Alignment.bottomCenter,
                colors: [Color(0xFF282725), Color(0xFF282725)],
              ),
            ),
          ),
        ),
        ListView(
          padding: EdgeInsets.zero,
          children: [
            SizedBox(height: topCellToolbarHeight + topInset + 58),
            HomeCell(
              title: 'Probl\u00E8mes',
              subtitle: 'Continuez votre parcours',
              onTap: () => Navigator.push(
                context,
                MaterialPageRoute(builder: (_) => const ProblemePage()),
              ),
            ),
            HomeCell(
              title: 'Apprendre de vos erreurs',
              subtitle: 'Revoyez vos coups cl\u00E9s',
              onTap: () => Navigator.push(
                context,
                MaterialPageRoute(builder: (_) => const ErrorReviewPage()),
              ),
            ),
            HomeCell(
              title: 'Entra\u00EEnements espac\u00E9s',
              subtitle: 'R\u00E9viser au bon moment',
              onTap: () => Navigator.push(
                context,
                MaterialPageRoute(builder: (_) => const SpacedTrainingPage()),
              ),
            ),
            HomeCell(
              title: 'Dojo',
              subtitle: 'Entra\u00EEnement libre',
              onTap: () => Navigator.push(
                context,
                MaterialPageRoute(builder: (_) => const EntrainementPage()),
              ),
            ),
            HomeCell(
              title: 'Apprendre',
              subtitle: 'Le\u00E7ons et bases',
              onTap: () => Navigator.push(
                context,
                MaterialPageRoute(builder: (_) => const ApprendrePage()),
              ),
            ),
            SizedBox(height: 190 + bottomInset),
          ],
        ),
      ],
    );
  }
}

class HomeCell extends StatefulWidget {
  final String title;
  final String subtitle;
  final VoidCallback onTap;

  const HomeCell({
    super.key,
    required this.title,
    required this.subtitle,
    required this.onTap,
  });

  @override
  State<HomeCell> createState() => _HomeCellState();
}

class _HomeCellState extends State<HomeCell> {
  static const int _minDownVisibleMs = 90;
  static const int _releaseAnimMs = 95;

  bool _down = false;
  bool _busy = false;
  int _pressToken = 0;
  DateTime? _downAt;

  void _handleTapDown(TapDownDetails _) {
    if (_busy) return;
    _pressToken++;
    _downAt = DateTime.now();
    setState(() => _down = true);
  }

  Future<void> _handleTapUp(TapUpDetails _) async {
    if (_busy) return;
    final token = _pressToken;
    _busy = true;
    try {
      final downAt = _downAt;
      if (downAt != null) {
        final elapsed = DateTime.now().difference(downAt).inMilliseconds;
        final remaining = _minDownVisibleMs - elapsed;
        if (remaining > 0) {
          await Future.delayed(Duration(milliseconds: remaining));
        }
      }
      if (!mounted || token != _pressToken) return;
      setState(() => _down = false);
      await Future.delayed(const Duration(milliseconds: _releaseAnimMs));
      if (!mounted || token != _pressToken) return;
      widget.onTap();
    } finally {
      _busy = false;
    }
  }

  void _handleTapCancel() {
    _pressToken++;
    _downAt = null;
    if (_down) {
      setState(() => _down = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      child: GestureDetector(
        behavior: HitTestBehavior.opaque,
        onTapDown: _handleTapDown,
        onTapUp: _handleTapUp,
        onTapCancel: _handleTapCancel,
        child: AnimatedScale(
          scale: _down ? 0.955 : 1.0,
          duration: const Duration(milliseconds: _releaseAnimMs),
          curve: Curves.easeOut,
          child: Container(
            height: 115,
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              color: Colors.white.withValues(alpha: 0.06),
              borderRadius: BorderRadius.circular(12),
            ),
            child: Row(
              children: [
                Container(
                  width: 90,
                  height: 90,
                  decoration: BoxDecoration(
                    color: Colors.white.withValues(alpha: 0.10),
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: Icon(
                    Icons.image_outlined,
                    size: 40,
                    color: Colors.white.withValues(alpha: 0.75),
                  ),
                ),
                const SizedBox(width: 10),
                Expanded(
                  child: Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        widget.title,
                        style: const TextStyle(
                          fontSize: 16,
                          fontWeight: FontWeight.w700,
                        ),
                      ),
                      const SizedBox(height: 5),
                      Text(
                        widget.subtitle,
                        style: TextStyle(
                          fontSize: 12,
                          color: Colors.white.withValues(alpha: 0.70),
                        ),
                      ),
                    ],
                  ),
                ),
                Icon(
                  Icons.chevron_right,
                  size: 24,
                  color: Colors.white.withValues(alpha: 0.55),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

class GamePage extends StatelessWidget {
  const GamePage({
    super.key,
    required this.engineElo,
    required this.eloTicks,
    required this.engineReady,
    required this.onEloChanged,
    required this.onEloChangeEnd,
    required this.fen,
    required this.lastMoveToHighlight,
    required this.suggestionArrows,
    required this.cellHighlights,
    required this.thinking,
    required this.onMove,
    required this.onPromote,
    required this.onPromotionCommited,
  });

  final int engineElo;
  final List<int> eloTicks;
  final bool engineReady;
  final ValueChanged<int> onEloChanged;
  final ValueChanged<int> onEloChangeEnd;
  final String fen;
  final BoardArrow? lastMoveToHighlight;
  final List<SuggestionArrow> suggestionArrows;
  final Map<String, Color> cellHighlights;
  final bool thinking;
  final void Function(ShortMove move) onMove;
  final Future<PieceType?> Function() onPromote;
  final void Function({
    required ShortMove moveDone,
    required PieceType pieceType,
  })
  onPromotionCommited;

  @override
  Widget build(BuildContext context) {
    return Stack(
      children: [
        Positioned.fill(
          child: Container(
            decoration: const BoxDecoration(
              gradient: LinearGradient(
                begin: Alignment.topCenter,
                end: Alignment.bottomCenter,
                colors: [Color(0xFF282725), Color(0xFF282725)],
              ),
            ),
          ),
        ),
        SafeArea(
          child: ListView(
            padding: const EdgeInsets.fromLTRB(0, kToolbarHeight, 0, 180),
            children: [
              Padding(
                padding: const EdgeInsets.all(12),
                child: ClipRRect(
                  borderRadius: BorderRadius.circular(18),
                  child: BlurOnly(
                    sigma: 14,
                    child: Padding(
                      padding: const EdgeInsets.all(12),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text('Niveau moteur: $engineElo'),
                          EloSlider(
                            value: engineElo,
                            marks: eloTicks,
                            onChanged: onEloChanged,
                            onChangeEnd: onEloChangeEnd,
                          ),
                        ],
                      ),
                    ),
                  ),
                ),
              ),
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 12),
                child: AspectRatio(
                  aspectRatio: 1,
                  child: BoardView(
                    fen: fen,
                    lastMoveToHighlight: lastMoveToHighlight,
                    suggestionArrows: suggestionArrows,
                    cellHighlights: cellHighlights,
                    thinking: thinking,
                    blackSideAtBottom: false,
                    onMove: onMove,
                    onPromote: onPromote,
                    onPromotionCommited: onPromotionCommited,
                  ),
                ),
              ),
            ],
          ),
        ),
      ],
    );
  }
}

class GradientBlurOnly extends StatelessWidget {
  final Widget child;
  final double maxSigma;
  final int steps; // plus grand = plus lisse

  const GradientBlurOnly({
    super.key,
    required this.child,
    this.maxSigma = 16,
    this.steps = 12,
  });

  @override
  Widget build(BuildContext context) {
    return ClipRect(
      child: LayoutBuilder(
        builder: (context, c) {
          final h = c.maxHeight;
          final bandH = h / steps;

          return Stack(
            fit: StackFit.expand,
            children: [
              for (int i = 0; i < steps; i++)
                Positioned(
                  left: 0,
                  right: 0,
                  top: bandH * i,
                  height: bandH + 0.5,
                  child: ClipRect(
                    child: BackdropFilter(
                      filter: ImageFilter.blur(
                        sigmaX: maxSigma * (1 - i / (steps - 1)),
                        sigmaY: maxSigma * (1 - i / (steps - 1)),
                      ),
                      child: const ColoredBox(color: Colors.transparent),
                    ),
                  ),
                ),
              Positioned.fill(child: child),
            ],
          );
        },
      ),
    );
  }
}

class BlurOnly extends StatelessWidget {
  final Widget child;
  final double sigma;

  const BlurOnly({super.key, required this.child, this.sigma = 14});

  @override
  Widget build(BuildContext context) {
    return ClipRect(
      child: BackdropFilter(
        filter: ImageFilter.blur(sigmaX: sigma, sigmaY: sigma),
        child: ColoredBox(color: Colors.transparent, child: child),
      ),
    );
  }
}

class ProblemePage extends StatelessWidget {
  const ProblemePage({super.key});

  @override
  Widget build(BuildContext context) {
    return const _PlaceholderPage(title: 'Probl\u00E8mes');
  }
}

class EntrainementPage extends StatelessWidget {
  const EntrainementPage({super.key});

  @override
  Widget build(BuildContext context) {
    return const _PlaceholderPage(title: 'Dojo');
  }
}

class ApprendrePage extends StatelessWidget {
  const ApprendrePage({super.key});

  @override
  Widget build(BuildContext context) {
    return const _PlaceholderPage(title: 'Apprendre');
  }
}

class ProfilePage extends StatelessWidget {
  const ProfilePage({super.key});

  @override
  Widget build(BuildContext context) => const _PlaceholderPage(title: 'Profil');
}

class SettingsPage extends StatelessWidget {
  const SettingsPage({super.key});

  @override
  Widget build(BuildContext context) =>
      const _PlaceholderPage(title: 'Param\u00E8tres');
}

class ErrorReviewPage extends StatelessWidget {
  const ErrorReviewPage({super.key});

  @override
  Widget build(BuildContext context) =>
      const _PlaceholderPage(title: 'Apprendre de vos erreurs');
}

class SpacedTrainingPage extends StatelessWidget {
  const SpacedTrainingPage({super.key});

  @override
  Widget build(BuildContext context) =>
      const _PlaceholderPage(title: 'Entra\u00EEnements espac\u00E9s');
}

class _PlaceholderPage extends StatelessWidget {
  const _PlaceholderPage({required this.title});

  final String title;

  @override
  Widget build(BuildContext context) {
    return SafeArea(
      child: Center(
        child: Text(title, style: Theme.of(context).textTheme.headlineSmall),
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
    required this.cellHighlights,
    required this.thinking,
    required this.onMove,
    required this.blackSideAtBottom,
    required this.onPromote,
    required this.onPromotionCommited,
  });

  final String fen;
  final BoardArrow? lastMoveToHighlight;
  final List<SuggestionArrow> suggestionArrows;
  final Map<String, Color> cellHighlights;
  final bool thinking;
  final bool blackSideAtBottom;
  final void Function(ShortMove move) onMove;
  final Future<PieceType?> Function() onPromote;
  final void Function({
    required ShortMove moveDone,
    required PieceType pieceType,
  })
  onPromotionCommited;

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
                  cellHighlights: cellHighlights,
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

class BottomGlassBar extends StatelessWidget {
  final int index;
  final ValueChanged<int> onTab;
  final VoidCallback onPlay;
  final VoidCallback onAnalyse;

  const BottomGlassBar({
    super.key,
    required this.index,
    required this.onTab,
    required this.onPlay,
    required this.onAnalyse,
  });

  @override
  Widget build(BuildContext context) {
    final bottomInset = MediaQueryData.fromView(
      View.of(context),
    ).viewPadding.bottom;

    final labelStyle = TextStyle(
      fontSize: 11,
      fontWeight: FontWeight.w800,
      color: Colors.white.withValues(alpha: 0.78),
    );

    return BlurOnly(
      sigma: 16,
      child: Container(
        width: double.infinity,
        padding: EdgeInsets.fromLTRB(16, 8, 16, 14 + bottomInset),
        color: Colors.transparent, // pas de cellule grise, pas de bordure
        child: Transform.translate(
          offset: const Offset(0, 14),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Row(
                children: [
                  Expanded(
                    child: SizedBox(
                      height: 54,
                      child: Glossy3DActionButton(
                        label: 'Analyser',
                        baseColor: const Color.fromARGB(255, 90, 203, 225),
                        textColor: Colors.white,
                        onPressed: onAnalyse,
                        gradientStrength: 0.72,
                      ),
                    ),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: SizedBox(
                      height: 54,
                      child: Glossy3DActionButton(
                        label: 'Jouer',
                        baseColor: const Color.fromARGB(255, 123, 195, 51),
                        textColor: Colors.white,
                        onPressed: onPlay,
                      ),
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 8),
              Row(
                children: [
                  _ZoomNavItem(
                    key: const ValueKey(0),
                    tabIndex: 0,
                    currentIndex: index,
                    icon: Symbols.home,
                    iconWeight: 600,
                    label: 'Accueil',
                    labelStyle: labelStyle,
                    onTap: onTab,
                  ),
                  _ZoomNavItem(
                    key: const ValueKey(1),
                    tabIndex: 1,
                    currentIndex: index,
                    icon: LucideIcons.puzzle,
                    label: 'Probl\u00E8mes',
                    labelStyle: labelStyle,
                    onTap: onTab,
                  ),
                  _ZoomNavItem(
                    key: const ValueKey(2),
                    tabIndex: 2,
                    currentIndex: index,
                    icon: LucideIcons.dumbbell,
                    label: 'Dojo',
                    labelStyle: labelStyle,
                    onTap: onTab,
                  ),
                  _ZoomNavItem(
                    key: const ValueKey(3),
                    tabIndex: 3,
                    currentIndex: index,
                    icon: Icons.school_outlined,
                    label: 'Apprendre',
                    labelStyle: labelStyle,
                    onTap: onTab,
                  ),
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class Glossy3DActionButton extends StatefulWidget {
  final String label;
  final Color baseColor;
  final Color textColor;
  final VoidCallback onPressed;
  final double gradientStrength;

  const Glossy3DActionButton({
    super.key,
    required this.label,
    required this.baseColor,
    required this.textColor,
    required this.onPressed,
    this.gradientStrength = 1.0,
  });

  @override
  State<Glossy3DActionButton> createState() => _Glossy3DActionButtonState();
}

class _Glossy3DActionButtonState extends State<Glossy3DActionButton> {
  bool _down = false;

  @override
  Widget build(BuildContext context) {
    final r = BorderRadius.circular(16);
    final strength = widget.gradientStrength.clamp(0.4, 1.2).toDouble();

    final hsl = HSLColor.fromColor(widget.baseColor);
    final satFactor = (0.83 + (hsl.saturation * 0.30))
        .clamp(0.83, 1.14)
        .toDouble();
    final up = 0.16 * (1.0 - hsl.lightness) * satFactor * strength;
    final down = 0.18 * hsl.lightness * satFactor * strength;

    final top = hsl
        .withLightness((hsl.lightness + up).clamp(0.0, 1.0))
        .toColor();
    final mid = widget.baseColor;
    final bot = hsl
        .withLightness((hsl.lightness - down).clamp(0.0, 1.0))
        .toColor();
    final glossTopAlpha = _down ? 0.15 : (0.22 + (0.04 * hsl.saturation));
    final innerShadowAlpha = _down
        ? 0.10
        : (0.16 + (0.025 * (1.0 - hsl.lightness)));

    final translateY = _down ? 2.0 : 0.0;
    final shadowY = _down ? 2.0 : 4.0;
    final shadowBlur = _down ? 6.0 : 10.0;

    return GestureDetector(
      behavior: HitTestBehavior.opaque,
      onTap: widget.onPressed,
      onTapDown: (_) => setState(() => _down = true),
      onTapUp: (_) => setState(() => _down = false),
      onTapCancel: () => setState(() => _down = false),
      child: Transform.translate(
        offset: Offset(0, translateY), // effet "appuye"
        child: AnimatedContainer(
          duration: const Duration(milliseconds: 90),
          curve: Curves.easeOut,
          decoration: BoxDecoration(
            borderRadius: r,
            boxShadow: [
              BoxShadow(
                color: Colors.black.withValues(alpha: 0.22),
                blurRadius: shadowBlur,
                offset: Offset(0, shadowY),
              ),
              // micro highlight autour (donne un cote "3D")
              BoxShadow(
                color: Colors.white.withValues(alpha: _down ? 0.02 : 0.05),
                blurRadius: 1.5,
                offset: const Offset(0, -0.5),
              ),
            ],
          ),
          child: ClipRRect(
            borderRadius: r,
            child: Stack(
              fit: StackFit.expand,
              children: [
                // Base gradient (le "degrade")
                DecoratedBox(
                  decoration: BoxDecoration(
                    gradient: LinearGradient(
                      begin: Alignment.topCenter,
                      end: Alignment.bottomCenter,
                      colors: [top, mid, bot],
                    ),
                  ),
                ),

                // Reflet glossy en haut
                Align(
                  alignment: Alignment.topCenter,
                  child: FractionallySizedBox(
                    heightFactor: 0.55,
                    child: DecoratedBox(
                      decoration: BoxDecoration(
                        gradient: LinearGradient(
                          begin: Alignment.topCenter,
                          end: Alignment.bottomCenter,
                          colors: [
                            Colors.white.withValues(alpha: glossTopAlpha),
                            Colors.transparent,
                          ],
                        ),
                      ),
                    ),
                  ),
                ),

                // Ombre interne en bas (donne la profondeur)
                Align(
                  alignment: Alignment.bottomCenter,
                  child: FractionallySizedBox(
                    heightFactor: 0.55,
                    child: DecoratedBox(
                      decoration: BoxDecoration(
                        gradient: LinearGradient(
                          begin: Alignment.bottomCenter,
                          end: Alignment.topCenter,
                          colors: [
                            Colors.black.withValues(alpha: innerShadowAlpha),
                            Colors.transparent,
                          ],
                        ),
                      ),
                    ),
                  ),
                ),

                Center(
                  child: Text(
                    widget.label,
                    style: TextStyle(
                      fontFamily: 'Nunito',
                      fontSize: 20,
                      fontWeight: FontWeight.w900,
                      fontVariations: const [FontVariation('wght', 1000)],
                      letterSpacing: -0.3,
                      color: widget.textColor,
                      shadows: [
                        Shadow(
                          color: Colors.black.withValues(
                            alpha: _down ? 0.18 : 0.27,
                          ),
                          blurRadius: 1.8,
                          offset: const Offset(0, 1.1),
                        ),
                        Shadow(
                          color: Colors.black.withValues(
                            alpha: _down ? 0.08 : 0.12,
                          ),
                          blurRadius: 0.4,
                          offset: const Offset(0, 0.45),
                        ),
                      ],
                    ),
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

class _PulseIconButton extends StatefulWidget {
  final String tooltip;
  final Widget icon;
  final VoidCallback onPressed;

  const _PulseIconButton({
    required this.tooltip,
    required this.icon,
    required this.onPressed,
  });

  @override
  State<_PulseIconButton> createState() => _PulseIconButtonState();
}

class _PulseIconButtonState extends State<_PulseIconButton>
    with SingleTickerProviderStateMixin {
  late final AnimationController _c;
  late final Animation<double> _pulseScale;
  bool _busy = false;

  @override
  void initState() {
    super.initState();
    _c = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 140),
    );

    _pulseScale = TweenSequence<double>([
      TweenSequenceItem(
        tween: Tween(
          begin: 1.0,
          end: 1.28,
        ).chain(CurveTween(curve: Curves.easeOut)),
        weight: 55,
      ),
      TweenSequenceItem(
        tween: Tween(
          begin: 1.28,
          end: 1.0,
        ).chain(CurveTween(curve: Curves.easeIn)),
        weight: 45,
      ),
    ]).animate(_c);
  }

  @override
  void dispose() {
    _c.dispose();
    super.dispose();
  }

  Future<void> _handleTap() async {
    if (_busy) return;
    _busy = true;
    try {
      await _c.forward(from: 0);
      if (!mounted) return;
      widget.onPressed();
    } finally {
      _busy = false;
    }
  }

  @override
  Widget build(BuildContext context) {
    return ScaleTransition(
      scale: _pulseScale,
      child: Tooltip(
        message: widget.tooltip,
        child: GestureDetector(
          behavior: HitTestBehavior.opaque,
          onTap: _handleTap,
          child: SizedBox(
            width: 48,
            height: 48,
            child: Center(child: widget.icon),
          ),
        ),
      ),
    );
  }
}

class _ZoomNavItem extends StatefulWidget {
  final int tabIndex;
  final int currentIndex;
  final IconData icon;
  final double? iconWeight;
  final String label;
  final TextStyle labelStyle;
  final ValueChanged<int> onTap;

  const _ZoomNavItem({
    super.key,
    required this.tabIndex,
    required this.currentIndex,
    required this.icon,
    this.iconWeight,
    required this.label,
    required this.labelStyle,
    required this.onTap,
  });

  @override
  State<_ZoomNavItem> createState() => _ZoomNavItemState();
}

class _ZoomNavItemState extends State<_ZoomNavItem>
    with SingleTickerProviderStateMixin {
  late final AnimationController _c;
  late final Animation<double> _pulseScale;

  @override
  void initState() {
    super.initState();
    _c = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 140),
    );

    _pulseScale = TweenSequence<double>([
      TweenSequenceItem(
        tween: Tween(
          begin: 1.0,
          end: 1.28,
        ).chain(CurveTween(curve: Curves.easeOut)),
        weight: 55,
      ),
      TweenSequenceItem(
        tween: Tween(
          begin: 1.28,
          end: 1.0,
        ).chain(CurveTween(curve: Curves.easeIn)),
        weight: 45,
      ),
    ]).animate(_c);
  }

  @override
  void dispose() {
    _c.dispose();
    super.dispose();
  }

  void _handleTap() {
    widget.onTap(widget.tabIndex); // sÃƒÆ’Ã‚Â©lection immÃƒÆ’Ã‚Â©diate (blanc/gris)
    _c.forward(from: 0); // pulse immÃƒÆ’Ã‚Â©diat (zoom/dezoom)
  }

  @override
  Widget build(BuildContext context) {
    final selected = widget.tabIndex == widget.currentIndex;

    final iconColor = selected
        ? Colors.white
        : Colors.white.withValues(alpha: 0.55);
    final labelColor = selected ? Colors.white : iconColor;

    return Expanded(
      child: GestureDetector(
        behavior: HitTestBehavior.opaque, // zone tap large, sans effet blanc
        onTap: _handleTap,
        child: Padding(
          padding: const EdgeInsets.symmetric(vertical: 6),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              ScaleTransition(
                scale: _pulseScale,
                child: Icon(
                  widget.icon,
                  color: iconColor,
                  size: 24,
                  weight: widget.iconWeight,
                ),
              ),
              const SizedBox(height: 3),
              Text(
                widget.label,
                style: widget.labelStyle.copyWith(color: labelColor),
              ),
            ],
          ),
        ),
      ),
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
