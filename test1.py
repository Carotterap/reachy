from reachy_mini import ReachyMini

with ReachyMini(media_backend="default") as mini:
    frame = mini.media.get_frame()
    print("picture")