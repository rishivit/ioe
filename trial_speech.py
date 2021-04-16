import pyttsx3

engine = pyttsx3.init()
engine.setProperty('volume',1.0)
engine.setProperty('rate', 125)
engine.say("Hi, Welcome to Circuit Digest Tutorial")
engine.runAndWait()