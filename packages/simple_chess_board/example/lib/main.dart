import 'package:flutter/material.dart';
import 'chess_board_with_history.dart';
import 'variants/custom_move_indicator.dart';
import 'variants/handling_promotions.dart';
import 'variants/interactive.dart';
import 'variants/simple.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'Simple chess board Demo',
      theme: ThemeData(
        primarySwatch: Colors.blue,
      ),
      home: const MyHomePage(),
    );
  }
}

class MyHomePage extends StatelessWidget {
  const MyHomePage({
    super.key,
  });

  Future<void> _goToSimpleBoard(BuildContext context) async {
    await Navigator.of(context).push(MaterialPageRoute(builder: (context) {
      return const SimpleBoardVariant();
    }));
  }

  Future<void> _goToInteractiveBoard(BuildContext context) async {
    await Navigator.of(context).push(MaterialPageRoute(builder: (context) {
      return const InteractiveBoard();
    }));
  }

  Future<void> _goToCustomIndicatorsBoard(BuildContext context) async {
    await Navigator.of(context).push(MaterialPageRoute(builder: (context) {
      return const CustomMoveIndicator();
    }));
  }

  Future<void> _goToPromotionHandling(BuildContext context) async {
    await Navigator.of(context).push(MaterialPageRoute(builder: (context) {
      return const HandlingPromotionsBoard();
    }));
  }

  Future<void> _goToHistoryExample(BuildContext context) async {
    await Navigator.of(context).push(MaterialPageRoute(builder: (context) {
      return const ChessBoardWithHistory();
    }));
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Simple chess board demo'),
      ),
      body: Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          mainAxisAlignment: MainAxisAlignment.center,
          crossAxisAlignment: CrossAxisAlignment.center,
          spacing: 10,
          children: [
            ElevatedButton(
              onPressed: () => _goToSimpleBoard(context),
              child: const Text('See simple board'),
            ),
            ElevatedButton(
              onPressed: () => _goToInteractiveBoard(context),
              child: const Text('See interactive board'),
            ),
            ElevatedButton(
              onPressed: () => _goToCustomIndicatorsBoard(context),
              child: const Text('See custom indicators board'),
            ),
            ElevatedButton(
              onPressed: () => _goToPromotionHandling(context),
              child: const Text('See nicer promotion handling'),
            ),
            ElevatedButton(
              onPressed: () => _goToHistoryExample(context),
              child: const Text('See history example'),
            ),
          ],
        ),
      ),
    );
  }
}
