import 'package:flutter/material.dart';

class EloSlider extends StatelessWidget {
  const EloSlider({
    super.key,
    required this.value,
    required this.onChanged,
    required this.onChangeEnd,
    this.min = 250,
    this.max = 3200,
    this.marks = const [250, 1000, 1600, 2000, 2400, 3200],
  });

  final int value;
  final int min;
  final int max;
  final List<int> marks;
  final ValueChanged<int> onChanged;
  final ValueChanged<int> onChangeEnd;

  double _t(int v) => (v - min) / (max - min);

  int _snap100(double raw) {
    final x = raw.clamp(min.toDouble(), max.toDouble());

    if (x < 275) return min;

    int snapped = ((x / 100).round() * 100).clamp(300, max);

    if (x >= max - 50) snapped = max;

    return snapped;
  }

  @override
  Widget build(BuildContext context) {
    final v = value.clamp(min, max);
    final displayValue = _snap100(v.toDouble());

    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        SizedBox(
          height: 42,
          child: LayoutBuilder(
            builder: (context, c) {
              const horizontalPad = 12.0;
              final trackW = c.maxWidth - horizontalPad * 2;

              return Stack(
                alignment: Alignment.center,
                children: [
                  Padding(
                    padding: const EdgeInsets.symmetric(horizontal: horizontalPad),
                    child: SliderTheme(
                      data: SliderTheme.of(context).copyWith(
                        trackHeight: 4,
                        activeTrackColor: const Color(0xFF6CC04A),
                        inactiveTrackColor: const Color(0xFF6B6B6B),
                        thumbColor: Colors.white,
                        overlayColor: Colors.transparent,
                        overlayShape: SliderComponentShape.noOverlay,
                        tickMarkShape: SliderTickMarkShape.noTickMark,
                      ),
                      child: Slider(
                        min: min.toDouble(),
                        max: max.toDouble(),
                        value: v.toDouble(),
                        divisions: null,
                        label: '$displayValue',
                        onChanged: (x) => onChanged(_snap100(x)),
                        onChangeEnd: (x) => onChangeEnd(_snap100(x)),
                      ),
                    ),
                  ),
                  Positioned.fill(
                    child: IgnorePointer(
                      child: Padding(
                        padding: const EdgeInsets.symmetric(horizontal: horizontalPad),
                        child: Stack(
                          children: [
                            for (final m in marks)
                              Positioned(
                                left: (trackW * _t(m)).clamp(0.0, trackW) - 0.5,
                                top: 18,
                                child: Container(
                                  width: 1,
                                  height: 10,
                                  color: const Color(0xFF8A8A8A),
                                ),
                              ),
                          ],
                        ),
                      ),
                    ),
                  ),
                ],
              );
            },
          ),
        ),
        const SizedBox(height: 6),
        SizedBox(
          height: 18,
          child: LayoutBuilder(
            builder: (context, c) {
              const horizontalPad = 12.0;
              final trackW = c.maxWidth - horizontalPad * 2;

              return Padding(
                padding: const EdgeInsets.symmetric(horizontal: horizontalPad),
                child: Stack(
                  children: [
                    for (final m in marks)
                      Positioned(
                        left: (trackW * _t(m)).clamp(0.0, trackW) - 18,
                        child: SizedBox(
                          width: 36,
                          child: Text(
                            '$m',
                            textAlign: TextAlign.center,
                            style: const TextStyle(
                              fontSize: 12,
                              color: Color(0xFFBEBEBE),
                            ),
                          ),
                        ),
                      ),
                  ],
                ),
              );
            },
          ),
        ),
      ],
    );
  }
}
