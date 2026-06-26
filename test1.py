from reachy_mini import ReachyMini


with ReachyMini() as mini:
    mini.audio.play_file("tts_output.wav")