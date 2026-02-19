import 'dart:math' as math;
import 'package:flutter/material.dart';

class EloSlider extends StatelessWidget {
  const EloSlider({
    super.key,
    required this.value,
    required this.marks,
    required this.onChanged,
    this.onChangeEnd,
  });

  final int value;
  final List<int> marks;
  final ValueChanged<int> onChanged;
  final ValueChanged<int>? onChangeEnd;

  @override
  Widget build(BuildContext context) {
    final sortedMarks = marks.toSet().toList()..sort();
    if (sortedMarks.length < 2) {
      return const SizedBox.shrink();
    }

    final minValue = sortedMarks.first;
    final maxValue = sortedMarks.last;
    final clampedValue = value.clamp(minValue, maxValue);

    // Valeurs autorisées : 250, 300, 400, 500 ... jusqu'à maxValue
    final allowed = <int>[minValue];
    final start = minValue + 50; // 300 si minValue=250
    for (int e = start; e <= maxValue; e += 100) {
      allowed.add(e);
    }
    final maxIndex = allowed.length - 1;

    // Trouver l'index le plus proche pour "snap"
    int nearestIndex(int elo) {
      var bestI = 0;
      var bestD = (allowed[0] - elo).abs();
      for (var i = 1; i < allowed.length; i++) {
        final d = (allowed[i] - elo).abs();
        if (d < bestD) {
          bestD = d;
          bestI = i;
        }
      }
      return bestI;
    }

    final currentIndex = nearestIndex(clampedValue);

    // Style demandé
    const double thumbRadius = 16; // plus gros bouton
    const double trackHeight = 10; // barre plus épaisse

    final TextStyle labelStyle =
        Theme.of(context).textTheme.bodySmall?.copyWith(
              color: const Color(0xFFBEBEBE),
              fontSize: 12,
            ) ??
            const TextStyle(
              color: Color(0xFFBEBEBE),
              fontSize: 12,
            );

    return LayoutBuilder(
      builder: (context, constraints) {
        final totalWidth = constraints.maxWidth;

        // Avec RoundedRectSliderTrackShape, la track est paddée par le plus grand
        // rayon entre overlay et thumb. Comme overlay = noOverlay, on prend thumbRadius.
        final trackStart = thumbRadius;
        final trackWidth = math.max(0.0, totalWidth - (trackStart * 2));

        final labels = <Widget>[];
        for (final mark in sortedMarks) {
          final ratio = nearestIndex(mark) / maxIndex;
          final x = trackStart + ratio * trackWidth;

          final text = '$mark';
          final painter = TextPainter(
            text: TextSpan(text: text, style: labelStyle),
            textDirection: Directionality.of(context),
            maxLines: 1,
          )..layout();

          final left = (x - painter.width / 2)
              .clamp(0.0, math.max(0.0, totalWidth - painter.width))
              .toDouble();

          labels.add(
            Positioned(
              left: left,
              top: 0,
              child: Text(text, style: labelStyle),
            ),
          );
        }

        return Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            SliderTheme(
              data: SliderTheme.of(context).copyWith(
                // Bouts arrondis + barre épaisse
                trackShape: const RoundedRectSliderTrackShape(),
                trackHeight: trackHeight,

                // Gros thumb rond
                thumbShape: const RoundSliderThumbShape(
                  enabledThumbRadius: thumbRadius,
                  disabledThumbRadius: thumbRadius,
                ),

                // Pas de halo
                overlayShape: SliderComponentShape.noOverlay,

                // IMPORTANT : aucune barre de graduation sur la track
                tickMarkShape: SliderTickMarkShape.noTickMark,

                // Couleurs (tu peux ajuster)
                activeTrackColor: const Color(0xFF8FD46A),
                inactiveTrackColor: const Color(0xFF7B7B7B),
                thumbColor: const Color(0xFFEFEFEF),
              ),
              child: Slider(
                min: 0,
                max: maxIndex.toDouble(),
                divisions: maxIndex, // discret : N intervalles -> N+1 valeurs possibles
                value: currentIndex.toDouble(),
                onChanged: (v) => onChanged(allowed[v.round()]),
                onChangeEnd: onChangeEnd == null
                    ? null
                    : (v) => onChangeEnd!(allowed[v.round()]),
              ),
            ),
            const SizedBox(height: 6),
            SizedBox(
              height: 18,
              child: Stack(children: labels),
            ),
          ],
        );
      },
    );
  }
}
