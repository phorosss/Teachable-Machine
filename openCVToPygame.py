import pygame

import cv2
import numpy
import tensorflow.keras

def main():
    #-----------------------------Setup------------------------------------------------------#
    """ Set up the game and run the main game loop """
    pygame.init()      # Prepare the pygame module for use
    surfaceSize = 600   # Desired physical surface size, in pixels.
    
    clock = pygame.time.Clock()  #Force frame rate to be slower

    # Create surface of (width, height), and its window.
    mainSurface = pygame.display.set_mode((surfaceSize, surfaceSize))
    font = pygame.font.SysFont("Arial", 26)  #Create a font object

    #Set up video and learning model
    cap = cv2.VideoCapture(0)
    size = (224, 224)
    model = tensorflow.keras.models.load_model('brooksExample/keras_model.h5')
    classes = ['Mr Brooks', 'Stick Person', 'Ruler','Charizard X', 'Eevee', 'None']

    #-----------------------------Program Variable Initialization----------------------------#
    # Set up some data to describe a small circle and its color
    circleColor = (255, 0, 0)        # A color is a mix of (Red, Green, Blue)


    #-----------------------------Main Program Loop---------------------------------------------#
    while cap.isOpened():       
        #-----------------------------Event Handling-----------------------------------------#
        ev = pygame.event.poll()    # Look for any event
        if ev.type == pygame.QUIT:  # Window close button clicked?
            break                   #   ... leave game loop


        #-----------------------------Program Logic---------------------------------------------#
        #Read an image from the video stream
        ret, img = cap.read()
        if not ret:
            break

        #Do the required image processing to format image correctly
        #imgHeight, imgWidth, _ = img.shape             #Get the height and width of the image (if needed)
        img = cv2.flip(img, 1)
        imgProcessed = cv2.resize(img, size)
        imgProcessed = cv2.cvtColor(imgProcessed, cv2.COLOR_BGR2RGB)
        imgProcessed = (imgProcessed.astype(numpy.float32) / 127.0) - 1
        imgProcessed = numpy.expand_dims(imgProcessed, axis=0)
        
        #Get a prediction from the learning model
        prediction = model.predict(imgProcessed)
        index = numpy.argmax(prediction)
        threshold = numpy.max(prediction)
        threshold = round(threshold*100, 0)
        #print(index, threshold, prediction)
        outputText = (f"The color is {classes[index]} with {threshold}% certainty")
        
        #Convert the cv2 image format to a pygame image format
        displayImage = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        displayImage = numpy.fliplr(displayImage)
        displayImage = numpy.rot90(displayImage)
        displayImage = pygame.surfarray.make_surface(displayImage)


        #-----------------------------Drawing Everything-------------------------------------#
        # We draw everything from scratch on each frame.
        # So first fill everything with the background color
        mainSurface.fill((0, 200, 255))
        
        #Put the camera image on the mainSurface
        mainSurface.blit(displayImage, (0, 0))

        #Display the text on the mainSurface
        renderedText = font.render(outputText, 1, pygame.Color("black"))
        mainSurface.blit(renderedText, (10,500))

        # Now the surface is ready, tell pygame to display it!
        pygame.display.flip()
        
        clock.tick(24) #Force frame rate to be slower


    pygame.quit()     # Once we leave the loop, close the window.

main()