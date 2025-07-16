# Python tools for picking DAS events manually

# References:
# https://stackoverflow.com/questions/39079562/matplotlib-animat ion-vertical-cursor-line-through-subplots 
# https://stackoverflow.com/questions/48446351/distinguish-button-press-event-from-drag-and-zoom-clicks-in-matplotlib 

import matplotlib.animation as animation
import numpy as np
import matplotlib.pyplot as plt


class Click():
    def __init__(self, ax, func, button=1):
        self.ax=ax
        self.func=func
        self.button=button
        self.press=False
        self.move = False
        self.c1=self.ax.figure.canvas.mpl_connect('button_press_event', self.onpress)
        self.c2=self.ax.figure.canvas.mpl_connect('button_release_event', self.onrelease)
        self.c3=self.ax.figure.canvas.mpl_connect('motion_notify_event', self.onmove)

    def onclick(self,event):
        if event.inaxes == self.ax:
            if event.button == self.button:
                self.func(event, self.ax)
    def onpress(self,event):
        self.press=True
    def onmove(self,event):
        if self.press:
            self.move=True
    def onrelease(self,event):
        if self.press and not self.move:
            self.onclick(event)
        self.press=False; self.move=False
        
        
class MouseMove:
    def __init__(self, ax, func):
        self.ax=ax
        self.func = func
        self.c1=self.ax.figure.canvas.mpl_connect('motion_notify_event', self.onMouseMove)
    def onMouseMove(self,event):
        if event.inaxes == self.ax:
            self.func(event, self.ax)