import os
import shutil
import keras.backend
import tensorflow as tf
import pygame
import pyscreenshot
from skimage import color, data
from skimage.transform import resize, rotate
import numpy as np
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
dark_grey = (100, 100, 100)
green = (0, 255, 0)
dark_green = (0, 100, 0)
red = (255, 0, 0)

# INITIALISE
pygame.init()
display = pygame.display.set_mode((windowWidth, windowHeight))
pygame.display.set_caption('Understanding Handwriting with Machine Learning')
clock = pygame.time.Clock()
clock.tick(FPS)
icon = pygame.image.load("icon.png")
pygame.display.set_icon(icon)
predictionPercent = 0

# Images & text

textFontSmall = pygame.font.SysFont("monospace", 16)
textFontMed = pygame.font.SysFont("monospace", 20)
textFontBig = pygame.font.SysFont("monospace", 28)
textFontHuge = pygame.font.SysFont("monospace", 72)
background = pygame.image.load('background.png')
userImage = 0

# Check for current user data
if os.path.exists('data.txt'):
    file = open('data.txt', 'r')
    content = file.readline()
    if content == "":
        file.close()
        currentNet = 'None'
    else:
        currentNet = content
        print("Current network loaded from file!")

else:
    newFile = open("data.txt", 'w')
    newFile.close()
    currentNet = 'None'
    print("First time? data.txt created to store current network between sessions")


# Creates and displays text
def makeText(text, bold, textColor, size, location):
    return display.blit(pygame.font.SysFont("monospace", size).render(text, bold, textColor), location)


# Creates and displays a paragraph
def makeParagraph(text, bold, textColor, size, location, newLine, lineGap):
    line = 0
    count = 0
    while count < len(text):
        newText: str = ""
        if count + newLine > len(text):
            newLine = len(text) - count
        while len(newText) < newLine:
            newText += text[count]
            count += 1
        display.blit(pygame.font.SysFont("monospace", size).render(newText, bold, textColor),
                     (location[0], location[1] + lineGap * line))
        line += 1


# Creates and displays button, also listens for click to perform user-defined function
def makeButton(rect, onClick, noClick, text):
    pygame.draw.rect(display, black, (rect[0] + 4, rect[1] + 8, rect[2], rect[3]))
    if pygame.draw.rect(display, white, rect).collidepoint(checkMouse()):
        makeText(text, True, black, 20, (rect[0] + (rect[2] // 3) - len(text) * 2.2, rect[1] + (rect[3] // 3)))
        pygame.draw.rect(display, grey, rect)
        if checkClick():
            pygame.draw.rect(display, dark_green, rect)
            return onClick
    makeText(text, True, black, 20, (rect[0] + (rect[2] // 3) - len(text) * 2.2, rect[1] + (rect[3] // 3)))
    return noClick


# Creates and displays an interactive textbox with optional filter
def makeTextBox(left, top, textSize, maxChars, currentText, filter=[]):
    if pygame.draw.rect(display, black, (left, top, (textSize * maxChars) * 0.7, textSize * 1.1)) \
            .collidepoint(checkMouse()):
        pygame.draw.rect(display, dark_grey, (left, top, (textSize * maxChars) * 0.7, textSize * 1.1))
        for event in event_list:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_BACKSPACE:
                    if len(currentText) > 0:
                        currentText = currentText[0:len(currentText) - 1]
                elif event.unicode in filter or filter == []:
                    if len(currentText) <= maxChars:
                        currentText += event.unicode
    makeText(currentText, True, white, textSize, (left + textSize / 10, top + textSize / 10))
    return currentText


screen = 'Main'
prediction = 'Error'
drawMemory = []
brushSize = 12
enterTextInput = ""
savedFile = False


# Capture number on-screen, then subsample to 28x28 and save to directory
def captureAndSubsample():
    global userImage
    funUserImage = pyscreenshot.grab(bbox=(785, 335, 1125, 675))  # Grab image of draw area
    funUserImage = color.rgb2gray(funUserImage)  # Convert to greyscale
    funUserImage = resize(funUserImage, (28, 28))  # Resize
    cv2.imwrite('userImage.png', funUserImage * 255)  # Save file to directory
    userImage = 255 * funUserImage


# Create a new net with parameters passed from newNetworkScreen()
def createNewNet(name, layers, optimizer, layerContent):
    global currentNet
    mnist = tf.keras.datasets.mnist
    (xTrain, yTrain), (xTest, yTest) = mnist.load_data()
    xTrain = tf.keras.utils.normalize(xTrain, axis=1)
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())
    for layer in layerContent:
        print(layer)
        model.add(tf.keras.layers.Dense(int(layer[0]), activation=eval(layer[1])))
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(xTrain, yTrain, epochs=5)
    model.save('networks/' + name)
    print("Net created! Initialised with 5 epochs, see train option in the network manager to refine")


# Removes all data from /customData and labels.txt
def clearUserData():
    for image in os.listdir('customData'):
        os.remove("customData/" + image)
    labels = open('customData/labels.txt', 'w')
    labels.close()


# Train net with specified epochs on the mnist test set
def trainNet(newEpochs):
    global currentNet
    mnist = tf.keras.datasets.mnist
    (xTrain, yTrain), (xTest, yTest) = mnist.load_data()
    model = tf.keras.models.load_model('networks/' + currentNet)
    model.fit(xTrain, yTrain, epochs=newEpochs)


# Update cursor position
def checkMouse():
    funMousePos = pygame.mouse.get_pos()
    return funMousePos


# Update click boolean
def checkClick():
    left, right, middle = pygame.mouse.get_pressed()
    funClick = False
    if left:
        funClick = True
        return funClick


def guessScreen(funMouse, funClick):
    global screen, predictionPercent

    # Title
    makeText('Your Drawing:', True, green, 32, (windowWidth // 16, windowHeight // 16))

    # Number prediction title
    makeText('I am ' + str(predictionPercent) + "%", True, green, 30,
             (windowWidth // 1.7, windowHeight // 12))
    makeText('sure this is', True, green, 28,
             (windowWidth // 1.7, windowHeight // 6))
    makeText('the number:', True, green, 28,
             (windowWidth // 1.7, windowHeight // 4))

    # Number prediction
    makeText(str(prediction), True, green, 128, (windowWidth // 1.5, windowHeight // 3))

    # Submitted drawing
    userImage = pygame.image.load("userImage.png")  # Load file as type surface
    userImage = pygame.transform.scale(userImage, (300, 300))  # Scale loaded file for display
    display.blit(userImage, (25, 75))

    # Back button
    if makeButton((400, 320, 160, 50), True, False, 'Back'):
        screen = "Draw"


def enterTextScreen(funMouse, funClick):
    global screen, enterTextInput

    # Title
    makeText("Please enter the number (0-9)", True, green, 20, (windowWidth // 4, windowHeight // 16))

    # Hint
    makeText("Press enter to submit", True, green, 20, (windowWidth // 3.5, windowHeight // 8))

    # Display for number input
    pygame.draw.rect(display, black, (225, 100, 150, 200))

    # Check for keyboard input
    for event in event_list:
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
                    screen = "Draw"
            else:
                if event.unicode in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
                    enterTextInput = event.unicode
    makeText(enterTextInput, True, white, 72, (280, 150))

allTested = False
correct, total = 0, 0

def drawScreen(funMouse, funClick):
    global brushSize, screen, allTested, correct, total

    # Title
    makeText('Draw a number', True, green, 20, (windowWidth // 2.75, windowHeight // 16))

    # Draw Area
    pygame.draw.rect(display, green, (178, 58, 244, 244))
    pygame.draw.rect(display, black, (180, 60, 240, 240))

    # Draw area detection
    if pygame.Rect(180, 60, 240, 240).collidepoint(funMouse):
        pygame.draw.circle(display, grey, funMouse, brushSize + 2)
        if funClick:
            drawMemory.append([display, white, funMouse, brushSize])

    # Brush Buttons
    if makeButton((450, 80, 100, 50), True, False, "Small"):
        brushSize = 8

    if makeButton((450, 160, 100, 50), True, False, "Medium"):
        brushSize = 12

    if makeButton((450, 240, 100, 50), True, False, "Big"):
        brushSize = 16

    # Back button
    if makeButton((400, 320, 160, 50), True, False, "Back"):
        screen = "Main"

    # Clear button
    if makeButton((220, 320, 160, 50), True, False, "Clear"):
        drawMemory.clear()

    # Submit button detection
    if makeButton((40, 320, 160, 50), True, False, "Submit"):
        global prediction
        captureAndSubsample()
        prediction = classifyUserImage('userImage.png', True)
        screen = "Guess"

    # Save to directory button
    if makeButton((40, 250, 120, 50), True, False, "Save"):
        captureAndSubsample()
        fileCount = 1
        for path in os.listdir('customData'):
            fileCount = fileCount + 1
        global userImage
        # pygame.image.save(userImage, 'customData/userImage' + str(fileCount) + '.png')
        userImage = resize(userImage, (28, 28))  # Resize
        cv2.imwrite('customData/userImage' + str(fileCount) + '.png', 255 * userImage)  # Save file to directory
        screen = "EnterText"

    # Test all button
    if makeButton((40, 150, 120, 50), True, False, "Test All"):
        labels = open('customData/labels.txt')
        content = labels.readline()
        correct, total = 0, 0
        for count, image in enumerate(os.listdir('customData')):
            if count == 0:
                continue
            print(image)
            prediction = classifyUserImage(image, False)
            if count <= len(os.listdir('customData')) - 2:
                print('Actual: ' + str(prediction) + ' Expected: ' + str(content[count]))
                if str(prediction) == str(content[count-1]):
                    correct += 1
                total += 1
            else:
                break
        allTested = True
        print(str(correct) + '/' + str(total) + " Were correctly classified")

    # Print test results
    if allTested:
        if correct == 0:
            makeText(str(correct) + "/" + str(total) + " | 0%", True, black, 20, (25, 100))
        else:
            makeText(str(correct) + "/" + str(total) + " | " + str(round((correct/total) * 100, 2)) + "%", True, black, 20, (25, 100))

    # Display currently drawn dots
    for dot in drawMemory:
        pygame.draw.circle(dot[0], dot[1], dot[2], dot[3])

    global savedFile
    if savedFile:
        makeText('Saved! ', True, green, 20, (60, 220))


scrollPos = 0


def networksScreen(funMouse, funClick):
    global screen, scrollPos, currentNet

    # Scroll Menu
    pygame.draw.rect(display, white, (25, 25, 350, 350))
    pygame.draw.rect(display, black, (325, 25, 50, 350))
    for count, net in enumerate(os.listdir('networks')):
        if currentNet == net:
            netString = "--> " + net
        else:
            netString = net

        if makeButton((25, (count + 1) * 60, 300, 50), True, False, netString):
            currentNet = net
            file = open('data.txt', 'w')
            file.write(net)
            file.close()
            keras.backend.clear_session()

    # New net button
    if makeButton((400, 25, 150, 70), True, False, "New"):
        screen = "NewNetwork"

    # Train net button
    if makeButton((400, 115, 150, 70), True, False, "Train"):
        screen = "TrainNetwork"

    # Delete net button
    if makeButton((400, 205, 150, 70), True, False, "Delete"):
        if os.path.exists("networks/" + currentNet):
            shutil.rmtree("networks/" + currentNet)

    # Back button
    if makeButton((400, 295, 150, 70), True, False, "Back"):
        if os.path.exists("networks/" + currentNet):
            screen = "Main"
        else:
            makeText("Please select a net!", False, black, 24, (25, 25))


newName = ''
newLayers = ''
newOptimizer = ''


def newNetworkScreen(funMouse, funClick):
    global screen, event, newName, newLayers, newOptimizer

    # New network name title and text-box
    makeText('Name', True, green, 40, (50, 50))
    newName = makeTextBox(150, 50, 40, 12, newName)

    # New network layers title and text-box
    makeText('Layers', True, green, 40, (50, 150))
    newLayers = makeTextBox(200, 150, 40, 2, newLayers, ['1', '2', '3', '4', '5', '6', '7', '8', '9'])

    # New network optimizer title and text-box
    makeText('Optimizer', True, green, 40, (50, 250))
    newOptimizer = makeTextBox(275, 250, 40, 12, newOptimizer)

    # Continue button
    if makeButton((25, 325, 250, 50), True, False, "Continue"):
        if newName != "":
            if newLayers in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']:
                if newOptimizer in ['adam', 'sgd', 'rmsprop', 'adadelta', 'adagrad', 'adamax', 'nadam', 'ftrl']:
                    screen = "EditLayers"

    # Back button
    if makeButton((325, 325, 250, 50), True, False, "Back"):
        screen = "Networks"


count = 1
allLayers = []
newLayer = (str("0"), str(""))


def editLayersScreen(funMouse, funClick):
    global screen, count, newLayers, newOptimizer, newName, newLayer, event

    # New layer title
    makeText("Layer " + str(count), True, green, 30, (250, 50))

    # New layer node title and text-box
    makeText('Nodes', True, green, 40, (25, 150))
    newLayer = (makeTextBox(150, 150, 40, 3, newLayer[0]), newLayer[1])

    # New layer activation title and text-box
    makeText('Activation', True, green, 40, (25, 250))
    newLayer = (newLayer[0], makeTextBox(275, 250, 40, 10, newLayer[1]))

    # Continue button
    if makeButton((150, 325, 250, 50), True, False, "Continue"):
        print(newLayer[0])
        if newLayer[1] in ["relu", "sigmoid", "softmax", "softplus", "softsign", "tanh", "selu", "elu", "exponential"]:
            newLayer = (newLayer[0], "tf.nn." + newLayer[1])
            if count == int(newLayers):
                pygame.draw.rect(display, black, (100, 100, 400, 200))
                makeText("Net is being trained,", True, white, 26, (110, 150))
                makeText("This could take a while!", True, white, 26, (110, 190))
                pygame.display.flip()
                newLayer = ("10", newLayer[1])
                allLayers.append(newLayer)
                newLayer = ("0", "")
                count = 1
                createNewNet(newName, newLayers, newOptimizer, allLayers)
                screen = "Networks"
            allLayers.append(newLayer)
            newLayer = ("0", "")
            count += 1


newEpochs = '0'


def trainNetworkScreen(funMouse, funClick):
    global screen, newEpochs

    # Epochs title and text-box
    makeText("Epochs", True, green, 40, (50, 100))
    newEpochs = makeTextBox(200, 100, 40, 3, newEpochs, ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0'])

    # Continue button
    if makeButton((50, 325, 225, 50), True, False, "Continue"):
        if newEpochs != "":
            pygame.draw.rect(display, black, (100, 100, 400, 200))
            makeText("Net is being trained,", True, white, 26, (110, 150))
            makeText("This could take a while!", True, white, 26, (110, 190))
            pygame.display.flip()
            trainNet(int(newEpochs))

    # Back button
    if makeButton((300, 325, 225, 50), True, False, "Back"):
        screen = 'Networks'


def mainScreen(funMouse, funClick):
    global screen

    # Title
    makeText('Understanding Handwriting with Machine Learning', True, green, 20,
             (windowWidth // 32, windowHeight // 16))

    # Select a network if none selected
    if currentNet == 'None':
        makeText("Create a net", True, black, 20, (50, 230))
        makeText("To begin! ->", True, black, 20, (50, 260))

    # Draw button detection
    if makeButton((210, 87, 185, 80), True, False, "Draw") and currentNet != 'None':
        screen = "Draw"

    # Network button detection
    if makeButton((210, 212, 185, 80), True, False, "Networks"):
        screen = "Networks"

    # Settings button detection
    if makeButton((220, 320, 165, 50), True, False, "Settings"):
        screen = "Settings"

    # Info button detection
    if makeButton((426, 87, 40, 40), True, False, "i"):
        pygame.draw.rect(display, white, (410, 160, 155, 220))

        makeParagraph('"Draw" option will let you draw your own number, then the current algorithm will attempt to'
                      'classify it. "Train" allows for the customization of neural nets and training using datasets',
                      False, black, 14, (412, 158), 18, 20)


def settingsScreen(funMouse, funClick):
    global screen

    # Title
    makeText('Settings Menu', True, green, 20, (220, windowHeight // 16))

    # Clear user data button
    if makeButton((200, 50, 205, 80), True, False, "Clear Data"):
        clearUserData()

    # Create dataset from custom images button
    if makeButton((180, 175, 245, 80), True, False, "Create dataset"):
        pathToCustomData = "customData"
        vectorizedImages = []
        for _, file in enumerate(os.listdir(pathToCustomData)):
            image = cv2.imread(pathToCustomData + file)
            imageArray = np.array(image)
            vectorizedImages.append(imageArray)
        np.savez("datasets/customDataset.npz", DataX=vectorizedImages)

        # Saving labels
        labels = open('customData/labels.txt', 'r')
        content = labels.readline()
        labels.close()
        npLabels = np.empty((0,), dtype=np.int32)
        for digit in content:
            npLabels = np.append(int(digit))
        np.save('datasets/labels', npLabels)

    # Back button
    if makeButton((220, 300, 165, 80), True, False, "Back"):
        screen = "Main"


# Handle screen changes
def runScreen(funScreen):
    # Display background
    display.blit(background, (0, 0))

    # Display screen
    if funScreen == 'Main':
        mainScreen(checkMouse(), checkClick())

    elif funScreen == 'Draw':
        drawScreen(checkMouse(), checkClick())

    elif funScreen == 'Networks':
        networksScreen(checkMouse(), checkClick())

    elif funScreen == 'NewNetwork':
        newNetworkScreen(checkMouse(), checkClick())

    elif funScreen == 'TrainNetwork':
        trainNetworkScreen(checkMouse(), checkClick())

    elif funScreen == 'EditLayers':
        editLayersScreen(checkMouse(), checkClick())

    elif funScreen == 'Guess':
        guessScreen(checkMouse(), checkClick())

    elif funScreen == 'EnterText':
        enterTextScreen(checkMouse(), checkClick())

    elif funScreen == 'Settings':
        settingsScreen(checkMouse(), checkClick())

    else:
        global screen
        screen = 'Main'
        print("ERROR: Screen mismatch")


# Load default net model for userImage classification
def classifyUserImage(image, submitted):
    model = tf.keras.models.load_model('networks/' + currentNet)
    try:
        global predictionPercent
        if not submitted:
            funUserImage = cv2.imread('customData/' + image, cv2.IMREAD_GRAYSCALE)
        else:
            funUserImage = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        funUserImage = tf.keras.utils.normalize(funUserImage, axis=1)
        funUserImage = tf.reshape(funUserImage, shape=[-1, 28, 28, 1])
        funPrediction = model.predict([funUserImage])  # Takes probability distributions
        i = 0
        x = 0
        for value in funPrediction[0]:
            i = value
            if i > x:
                x = i
        global predictionPercent
        predictionPercent = int(x * 100)
        funPrediction = np.argmax(funPrediction)  # Calculates the highest possible number
        return funPrediction
    except:
        print("ERROR: Non-existent userImage file!\nReturning to main menu...")
        global screen
        screen = 'Main'


# Pre-assigned vars
click = False
mousePos = (0, 0)

# MAIN LOOP #

while True:
    event_list = []
    for newEvent in pygame.event.get():
        event_list.append(newEvent)
        if newEvent.type == pygame.QUIT:
            pygame.quit()
    runScreen(screen)
    pygame.display.flip()
