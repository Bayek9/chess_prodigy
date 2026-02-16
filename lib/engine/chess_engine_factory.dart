import 'chess_engine.dart';
import 'chess_engine_factory_stub.dart'
    if (dart.library.io) 'chess_engine_factory_io.dart' as platform;

ChessEngine createChessEngine() => platform.createPlatformChessEngine();
