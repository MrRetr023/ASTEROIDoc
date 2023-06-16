import urllib3.request
import vlc
import pafy
text = input('Are you ready?')
order66 = 'https://www.youtube.com/watch?v=dQw4w9WgXcQ'

video = pafy.new(order66)
best = video.getbest()

media = vlc.MediaPlayer(best.order66)
def execute_order66():
    media.play()

if text=='yes':
    execute_order66()
