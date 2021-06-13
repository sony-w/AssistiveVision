import 'package:flutter_tts/flutter_tts.dart';

class AudibleGuide {
  final FlutterTts flutterTts = FlutterTts();

  AudibleGuide(String language) {
    flutterTts.setLanguage(language);
  }

  void sayTakingPicture() async {
    await flutterTts.speak("Taking picture.");
  }
}