#!/usr/bin/python3
import environment, os

if __name__ == '__main__':

    local_dir = os.path.dirname(__file__)
    video_dir = os.path.join(local_dir, 'Video')
    if not os.path.isdir(video_dir):
        print("Could not find 'Video' directory, creating one...")
        os.makedirs(video_dir)

    e = environment.Environment()
    e.run()
locals() # why is this line here again?
