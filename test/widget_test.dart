import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:chess_prodigy/main.dart';

void main() {
  testWidgets('Chess app renders home page', (WidgetTester tester) async {
    await tester.pumpWidget(const ChessProdigyApp());

    expect(find.text('Chess Prodigy - vs IA'), findsOneWidget);
    expect(find.byIcon(Icons.refresh), findsOneWidget);
  });
}
