# use the .png files in testMovie/ to make a movie, make it 12fps
#!/bin/bash
ffmpeg -framerate 3 -pattern_type glob -i 'testMovie/*.png' -c:v libx264 -pix_fmt yuv420p -vf "scale=3600:3000" testMovie.mp4