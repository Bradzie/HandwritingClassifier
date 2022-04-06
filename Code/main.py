import glob
import os, os.path
import time

import tensorflow as tf
import pygame
import pyscreenshot
from skimage import color, data
from skimage.transform import resize, rotate
import numpy as np
import matplotlib.pyplot as plt
import cv2

# NOTES --------------------------------- NOTES
# draw area screenshot needs to grab regardless of window position
# make text needs to accompany different text sizes

# import win32gui
# from keras.preprocessing.image import array_to_img

# SCREEN RESOLUTION AND SETTINGS

windowHeight = 400
windowWidth = 600
FPS = 60

# COLOURS

white = (255, 255, 255)
black = (0, 0, 0)
grey = (200, 200, 200)
green = (0, 255, 0)
red = (255, 0, 0)

# INITIALISE
pygame.init()
display = pygame.display.set_mode((windowWidth, windowHeight))
pygame.display.set_caption('Understanding Handwriting with Machine Learning')
clock = pygame.time.Clock()
clock.tick(FPS)
icon = pygame.image.load("icon.png")
pygame.display.set_icon(icon)

# Images & text

textFontSmall = pygame.font.SysFont("monospace", 16)
textFontMed = pygame.font.SysFont("monospace", 20)
textFontBig = pygame.font.SysFont("monospace", 28)
textFontHuge = pygame.font.SysFont("monospace", 72)
background = pygame.image.load('background.png')
userImage = 0


# Creates and displays text
def makeText(text, bold, textColor, size, location):
    return display.blit(pygame.font.SysFont("monospace", size).render(text, bold, textColor), location)


# Creates and displays button, also listens for click to perform user-defined function
def makeButton(rect, onClick, noClick, text):
    pygame.draw.rect(display, black, (rect[0] + 4, rect[1] + 8, rect[2], rect[3]))
    if pygame.draw.rect(display, white, rect).collidepoint(checkMouse()):
        makeText(text, True, black, (rect[2] + rect[3])//5, (rect[0] + rect[2]//3 - 10, rect[1] + rect[3]//3))
        pygame.draw.rect(display, grey, rect)
        if checkClick():
            return onClick
    makeText(text, True, black, 20, (rect[0] + rect[2]//3 - 10, rect[1] + rect[3]//3))
    return noClick


screen = 'MainMenu'
prediction = 'Error'
drawMemory = []
brushSize = 12
enterTextInput = ""
savedFile = False


def menuGuessScreen(funMouse, funClick):
    newScreen = 'MenuGuess'

    display.blit(background, (0, 0))
    makeText('Your Drawing:', True, green, 20, (windowWidth // 16, windowHeight // 16))  # Title
    makeText('This is the number: ' + str(prediction), True, green, 28, (windowWidth // 3, windowHeight // 12))  # Title
    display.blit(userImage, (windowWidth // 16, windowHeight // 8))  # User's drawn number
    newScreen = makeButton((400, 320, 160, 50), 'MenuDraw', 'MenuGuess', 'Back')

    return newScreen


def enterTextScreen(funMouse, funClick):
    global enterTextInput
    # - Layout Bottom Layer -
    display.blit(background, (0, 0))
    makeText("Please enter the number (0-9)", True, green, 20, (windowWidth // 4, windowHeight // 16))
    makeText("Press enter to submit", True, green, 20, (windowWidth // 3.5, windowHeight // 8))
    pygame.draw.rect(display, black, (225, 100, 150, 200))
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:
                if len(enterTextInput) == 1:
                    labels = open('customData/labels.txt', 'r')
                    content = labels.readline()
                    labels.close()
                    labels = open('customData/labels.txt', 'w')
                    labels.write(content + enterTextInput)
                    labels.close()
                    global savedFile
                    savedFile = True
                    return "MenuDraw"
            else:
                enterTextInput = event.unicode
    makeText(enterTextInput, True, white, 72, (280, 150))

    return "EnterText"


def menuDrawScreen(funMouse, funClick):
    newScreen = "MenuGuess"
    global brushSize
    # - Layout Bottom Layer -
    display.blit(background, (0, 0))
    makeText('Draw a number', True, green, 20, (windowWidth // 2.75, windowHeight // 16))
    pygame.draw.rect(display, green, (178, 58, 244, 244))
    pygame.draw.rect(display, black, (180, 60, 240, 240))  # Drawing Area
    pygame.draw.rect(display, black, (44, 258, 120, 50))
    pygame.draw.rect(display, white, (40, 250, 120, 50))  # Save to directory
    pygame.draw.rect(display, black, (44, 328, 160, 50))
    pygame.draw.rect(display, white, (40, 320, 160, 50))  # Submit

    # - Layout middle layer -

    # Draw area detection
    if pygame.Rect(180, 60, 240, 240).collidepoint(funMouse):
        pygame.draw.circle(display, grey, funMouse, brushSize + 2)
        if funClick:
            drawMemory.append([display, white, funMouse, brushSize])

    # Brush Buttons
    if makeButton((450, 80, 50, 50), True, False, "Small"):
        brushSize = 8

    if makeButton((450, 160, 50, 50), True, False, "Medium"):
        brushSize = 12

    if makeButton((450, 240, 50, 50), True, False, "Big"):
        brushSize = 16

    # Back button
    newScreen = makeButton((400, 320, 160, 50), "MainMenu", "MenuDraw", "Back")

    # Clear button
    if makeButton((220, 320, 160, 50), True, False, "Clear"):
        drawMemory.clear()

    # Submit button detection
    if makeButton((40, 320, 160, 50), True, False, "Submit"):
        global prediction
        captureAndSubsample()
        prediction = classifyUserImage()
        newScreen = "MenuGuess"

    # Save to directory button
    if makeButton((40, 250, 120, 50), True, False, "Export"):
        captureAndSubsample()
        fileCount = 1
        for path in os.listdir('customData'):
            fileCount = fileCount + 1
        global userImage
        pygame.image.save(userImage, 'customData/userImage' + str(fileCount) + '.png')
        global title
        title = 'What number did you draw? e.g. 0-9'

    #  - Layout top layer -
    makeText('Submit', True, black, 20, (70, 335))
    makeText('Save to', True, black, 20, (60, 255))
    makeText('directory', True, black, 20, (50, 270))
    for dot in drawMemory:  # Add drawn circles to screen
        pygame.draw.circle(dot[0], dot[1], dot[2], dot[3])

    global savedFile
    if savedFile:
        makeText('Saved! ', True, green, 20, (60, 220))

    return newScreen


def menuTrainScreen(funMouse, funClick):
    display.blit(background, (0, 0))
    # pygame.draw.rect(display, black, (214, 95, 185, 80))  # Turn into variables?
    pygame.draw.rect(display, white, (80, 100, 185, 80))
    pygame.draw.rect(display, white, (335, 100, 185, 80))
    makeText('Train Menu', True, green, 20, (windowWidth // 3, windowHeight // 16))

    return "MenuTrain"


def mainMenuScreen(funMouse, funClick):
    # - Layout bottom layer -
    display.blit(background, (0, 0))
    pygame.draw.rect(display, black, (214, 95, 185, 80))  # Turn into variables?
    pygame.draw.rect(display, white, (210, 87, 185, 80))  # Draw
    pygame.draw.rect(display, black, (214, 220, 185, 80))
    pygame.draw.rect(display, white, (210, 212, 185, 80))  # Train
    pygame.draw.rect(display, black, (224, 328, 165, 50))
    pygame.draw.rect(display, white, (220, 320, 165, 50))  # Settings
    pygame.draw.rect(display, black, (430, 95, 40, 40))
    pygame.draw.rect(display, white, (426, 87, 40, 40))  # Info
    makeText('Understanding Handwriting with Machine Learning', True, green, 20,
             (windowWidth // 32, windowHeight // 16))

    # - Layout middle layer -

    # Draw button detection
    if pygame.draw.rect(display, white, (210, 87, 185, 80)).collidepoint(funMouse):
        pygame.draw.rect(display, grey, (210, 87, 185, 80))
        if funClick:
            return "MenuDraw"

    # Train button detection
    if pygame.draw.rect(display, white, (210, 212, 185, 80)).collidepoint(funMouse):
        pygame.draw.rect(display, grey, (210, 212, 185, 80))
        if funClick:
            return "MenuTrain"

    # Settings button detection
    if pygame.draw.rect(display, white, (220, 320, 165, 50)).collidepoint(funMouse):
        pygame.draw.rect(display, grey, (220, 320, 165, 50))
        if funClick:
            return "SettingsMenu"

    # Info button detection
    if pygame.draw.rect(display, white, (426, 87, 40, 40)).collidepoint(funMouse):
        pygame.draw.rect(display, grey, (426, 87, 40, 40))
        pygame.draw.rect(display, white, (410, 160, 155, 200))

        # New line function?
        makeText('"Draw" option', False, black, 16, (412, 158))
        makeText('will let you', False, black, 16, (412, 178))
        makeText('draw your own', False, black, 16, (412, 198))
        makeText('number and then', False, black, 16, (412, 218))
        makeText('will attempt', False, black, 16, (412, 238))
        makeText('to classify it.', False, black, 16, (412, 258))
        makeText('"Train" will', False, black, 16, (412, 278))
        makeText('classify local', False, black, 16, (412, 298))
        makeText('data with the', False, black, 16, (412, 318))
        makeText('correct format', False, black, 16, (412, 338))  # Potentially needs re-writing

    # - Layout top layer -

    makeText('Draw', True, black, 28, (265, 110))
    makeText('Train', True, black, 28, (260, 235))
    makeText('i', False, black, 20, (440, 95))
    makeText('Settings', False, black, 20, (255, 335))

    return "MainMenu"


def menuSettingsScreen(funMouse, funClick):
    # - Layout bottom layer -
    display.blit(background, (0, 0))
    makeText('Settings Menu', True, green, 20, (windowWidth // 3, windowHeight // 16))

    # - Layout middle layer -

    # - Layout top layer -
    return "SettingsMenu"


# Function to capture number on-screen, then subsample to 28x28 and save to directory
def captureAndSubsample():
    global userImage
    funUserImage = pyscreenshot.grab(bbox=(785, 335, 1125, 675))  # Grab image of draw area
    funUserImage = color.rgb2gray(funUserImage)  # Convert to greyscale
    funUserImage = resize(funUserImage, (28, 28))  # Resize
    cv2.imwrite('userImage.png', 255 * funUserImage)  # Save file to directory
    userImage = pygame.image.load("userImage.png")  # Load file as type surface
    userImage = pygame.transform.scale(userImage, (150, 150))  # Scale loaded file for display


# Function to update cursor position
def checkMouse():
    funMousePos = pygame.mouse.get_pos()
    return funMousePos


# Function to update click boolean
def checkClick():
    left, right, middle = pygame.mouse.get_pressed()
    funClick = False
    if left:
        funClick = True
        return funClick


# Function to handle screen changes
def runScreen():
    if screen == 'MainMenu':
        newScreen = mainMenuScreen(checkMouse(), checkClick())
    elif screen == 'MenuDraw':
        newScreen = menuDrawScreen(checkMouse(), checkClick())
    elif screen == 'MenuTrain':
        newScreen = menuTrainScreen(checkMouse(), checkClick())
    elif screen == 'MenuGuess':
        newScreen = menuGuessScreen(checkMouse(), checkClick())
    elif screen == 'EnterText':
        newScreen = enterTextScreen(checkMouse(), checkClick())
    elif screen == 'SettingsMenu':
        newScreen = menuSettingsScreen(checkMouse(), checkClick())
    else:
        newScreen = 'MainMenu'
        print("ERROR: Screen declaration mismatch")
    return newScreen


# Function to check for quit event
def checkExit():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()


predictionPercent = []


# Function to load default net model for userImage classification
def classifyUserImage():
    model = tf.keras.models.load_model('defaultClassifier')
    try:
        global predictionPercent
        funUserImage = cv2.imread('userImage.png', cv2.IMREAD_GRAYSCALE)
        funUserImage = tf.keras.utils.normalize(funUserImage, axis=1)
        funUserImage = tf.reshape(funUserImage, shape=[-1, 28, 28, 1])
        funPrediction = model.predict([funUserImage])  # Takes probability distributions
        i = 0
        x = 0
        for value in funPrediction[0]:
            i = value
            if i > x:
                x = i
                print(x)
        predictionPercent = int(x * 100)
        funPrediction = np.argmax(funPrediction)  # Calculates the highest possible number
        return funPrediction
    except:
        print("ERROR: Non-existent userImage file!\nReturning to main menu...")
        global screen
        screen = 'MainMenu'


# Menu screens need creating for draw and neural network systems to be implemented.

# Pre-assigned vars
click = False
mousePos = (0, 0)

# MAIN LOOP #

while True:
    screen = runScreen()
    pygame.display.flip()
    checkExit()
