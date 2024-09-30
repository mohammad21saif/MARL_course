import pygame, sys, os
from pygame.locals import *

from collections import deque


pygame.init()
## Set the size of the pygame display window to 400 pixels wide, 300 pixels high
screen = pygame.display.set_mode((400,300))

## Load image elements from a single file
skinfilename = os.path.join('borgar.png')

try:
    skin = pygame.image.load(skinfilename)
except pygame.error as msg:
    print('cannot load skin')
    raise SystemExit(msg)

skin = skin.convert()

## Set the background color of the window to the element at coordinates (0,0) in the skin file
screen.fill(skin.get_at((0,0)))

clock = pygame.time.Clock()
pygame.key.set_repeat(200,50)

## Game main loop
while True:
    clock.tick(60)
    pass

## Get game events
for event in pygame.event.get():
    ## Quit game event
    if event.type == QUIT:
        pygame.quit()
        sys.exit()
    ## Keyboard operation
    elif event.type == KEYDOWN:
        ## Move left
        if event.key == K_LEFT:
            pass
        ## Move up
        elif event.key == K_UP:
            pass
        ## Move right
        elif event.key == K_RIGHT:
            pass
        ## Move down
        elif event.key == K_DOWN:
            pass
        ## Undo operation
        elif event.key == K_BACKSPACE:
            pass
        ## Redo operation
        elif event.key == K_SPACE:
            pass

class Sokoban:

    ## Initialize the Sokoban game
    def __init__(self):
        ## Set the map
        self.level = list(
            "----#####----------"
            "----#---#----------"
            "----#$--#----------"
            "--###--$##---------"
            "--#--$-$-#---------"
            "###-#-##-#---######"
            "#---#-##-#####--..#"
            "#-$--$----------..#"
            "#####-###-#@##--..#"
            "----#-----#########"
            "----#######--------")

        ## Set the width and height of the map and the position of the player in the map (index value in the map list)
        ## Total 19 columns
        self.w = 19

        ## Total 11 rows
        self.h = 11

        ## The initial position of the player is at self.level[163]
        self.man = 163
    
        screen.blit(skin, (i*w, j*w), (0,0,w,w))

    def draw(self, screen, skin):

        ## Get the width of each image element
        w = skin.get_width() / 4

        ## Iterate through each character element in the map level
        for i in range(0, self.w):
            for j in range(0, self.h):

                ## Get the character at the j-th row and i-th column in the map
                item = self.level[j*self.w + i]

                ## Display as a wall(#) at this position
                if item == '#':
                    ## Use the blit method from pygame to display the image at the specified position,
                    ## with the position coordinates (i*w, j*w), and the coordinates and length-width of the image in the skin as (0,2*w,w,w)
                    screen.blit(skin, (i*w, j*w), (0,2*w,w,w))
                ## Display as a space(-) at this position
                elif item == '-':
                    screen.blit(skin, (i*w, j*w), (0,0,w,w))
                ## Display as a player(@) at this position
                elif item == '@':
                    screen.blit(skin, (i*w, j*w), (w,0,w,w))
                ## Display as a box($) at this position
                elif item == '$':
                    screen.blit(skin, (i*w, j*w), (2*w,0,w,w))
                ## Display as a target point(.) at this position
                elif item == '.':
                    screen.blit(skin, (i*w, j*w), (0,w,w,w))
                ## Display as the player on a target point effect
                elif item == '+':
                    screen.blit(skin, (i*w, j*w), (w,w,w,w))
                ## Display as the box placed on a target point effect
                elif item == '*':
                    screen.blit(skin, (i*w, j*w), (2*w,w,w,w))

    def _move(self, d):
        ## Get the displacement in the map for the movement
        h = get_offset(d, self.w)

        ## If the target area of the movement is empty space or a target point, only the player needs to move
        if self.level[self.man + h] == '-' or self.level[self.man + h] == '.':
            ## Move the player to the target position
            move_man(self.level, self.man + h)
            ## Set the original position of the player after movement
            move_floor(self.level, self.man)
            ## The new position of the player
            self.man += h
            ## Add the move operation to the solution
            self.solution += d

        ## If the target area of the movement is a box, both the box and the player need to move
        elif self.level[self.man + h] == '*' or self.level[self.man + h] == '$':
            ## The displacement of the box and the player's position
            h2 = h * 2
            ## The box can only be moved if the next position is empty space or a target point
            if self.level[self.man + h2] == '-' or self.level[self.man + h2] == '.':
                ## Move the box to the target point
                move_box(self.level, self.man + h2)
                ## Move the player to the target point
                move_man(self.level, self.man + h)
                ## Reset the current position of the player
                move_floor(self.level, self.man)
                ## Set the player's new position
                self.man += h
                ## Mark the move operation as an uppercase character to indicate that a box was pushed in this step
                self.solution += d.upper()
                ## Increment the number of steps for pushing the box
                self.push += 1
    def undo(self):
        ## Check if there is a movement record
        if self.solution.__len__()>0:
            ## Store the movement record in the todo list for redo operation
            self.todo.append(self.solution[-1])
            ## Delete the movement record
            self.solution.pop()

            ## Get the offset to be moved for the undo operation: the negative of the offset of the last movement
            h = get_offset(self.todo[-1],self.w) * -1

            ## Check if this operation only moves the character without pushing a box
            if self.todo[-1].islower():
                ## Move the character back to its original position
                move_man(self.level, self.man + h)
                ## Set the current position of the character
                move_floor(self.level, self.man)
                ## Set the position of the character on the map
                self.man += h
            else:
                ## If this step pushes a box, move the character, box, and perform related operations in _move
                move_floor(self.level, self.man - h)
                move_box(self.level, self.man)
                move_man(self.level, self.man + h)
                self.man += h
                self.push -= 1
    
    def redo(self):
        ## Check if there is an undo operation recorded
        if self.todo.__len__() > 0:
            ## Move back the undone steps
            self._move(self.todo[-1].lower())
            ## Delete this record
            self.todo.pop()