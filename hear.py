# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 18:18:24 2020

@author: Manya
"""

import speech_recognition as sr 
import pyttsx3  
from textblob import TextBlob
# Initialize the recognizer  
r = sr.Recognizer()  
  
def SpeakText(command): 
      
    # Initialize the engine 
    engine = pyttsx3.init() 
    engine.say(command)  
    engine.runAndWait() 
def analyze(command):
    blob=TextBlob(command)
    print(blob.sentiment)
    if(blob.sentiment.polarity<0):
        print("Negative thoughts!")
    elif(blob.sentiment.polarity>0):
        print("Positive thoughts!")
    else:
        print("Neutral statement")
print("Welcome to the emotions test")
while(1):     
      
    # Exception handling to handle 
    # exceptions at the runtime 
    try: 
          
        # use the microphone as source for input. 
        with sr.Microphone() as source2: 
              
            # wait for a second to let the recognizer 
            # adjust the energy threshold based on 
            # the surrounding noise level
            print("what would you like to say pal?")
            r.adjust_for_ambient_noise(source2, duration=0.2) 
              
            #listens for the user's input  
            audio2 = r.listen(source2) 
              
            # Using ggogle to recognize audio 
            MyText = r.recognize_google(audio2) 
            MyText = MyText.lower() 
  
            print("Did you say "+MyText) 
            SpeakText(MyText) 
            analyze(MyText)
        ch=input("would you like to continue? (y/n)")
        if(ch=="y" or ch=="Y"):
            continue
        else:
            print("Thank you for taking the emotions test ")
            break
    except sr.RequestError as e: 
        print("Could not request results; {0}".format(e)) 
          
    except sr.UnknownValueError: 
        print("unknown error occured") 