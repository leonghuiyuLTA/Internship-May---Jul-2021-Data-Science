import pygame

from time import sleep


# Adapted from https://gist.github.com/jaytaph/01162edc901efd54d46c330bb0b07fcb

# Version 1:
# In this version, we deal with a road that is 200m in length (+-50m). The bus
# should travel the 200m and stop at 200m mark. Take position from the center of the bus
# scale: 1 pixel = 20cm ==> 200m = 1000 pixels
class Bus(pygame.sprite.Sprite):
    def __init__(self, surface):
        self.bounded_rect = surface.get_rect()

        super(Bus, self).__init__()

        self._original_image = pygame.image.load('bus.png')  # 60 by 25, scale (length 1 pixel = 20cm)
        self.image = self._original_image
        self.rect = self.image.get_rect()

        # Set the bus's initial position
        self.rect.centery = self.bounded_rect.height / 2 + 20  # middle row of screen
        self.rect.centerx = 200  # self.bounded_rect.width / 10


class BusStop(pygame.sprite.Sprite):
    def __init__(self, surface, position):
        self.bounded_rect = surface.get_rect()

        super(BusStop, self).__init__()

        self._original_image = pygame.image.load('busstop.png')  # 60 by 25, scale (length 1 pixel = 20cm)
        self.image = self._original_image
        self.rect = self.image.get_rect()

        # Set the busstop's initial position
        self.rect.centery = 85  # middle row of screen
        self.rect.centerx = 75 + position  # self.bounded_rect.width / 10


class BusViewer(object):
    def __init__(self, width=1150, height=200):
        # Initialize the game
        pygame.init()
        pygame.display.set_caption("V1 Auto-Bus")

        # Create the screen and background for the game
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((self.width, self.height), pygame.DOUBLEBUF)
        self.background = pygame.Surface(self.screen.get_size()).convert()

        self.font = pygame.font.Font(None, 24)

        # Create bus and add it to a "spritegroup"
        self.bus = Bus(self.screen)
        self.busstop1 = BusStop(self.screen, 0)
        self.busstop2 = BusStop(self.screen, 1000)
        self.sprites = pygame.sprite.RenderPlain((self.bus, self.busstop1, self.busstop2))

        # Road Parameters
        self.road_top = 100
        self.road_bottom = 140
        # Paint initial road (it will be a straight road since we don't update the road)
        for i in range(self.width):
            self.paint_road()

        # Is the game running or not
        self.running = False

    def paint_road(self):
        # Move whole background one pixel to the left
        self.background.scroll(1, 0)

        # Find the top and bottom of the road in pixels
        top = self.road_top
        bottom = self.road_bottom

        # Iterate the whole height of the game screen
        for i in range(self.height):
            if i > bottom:
                # Print a green pixel at the bottom
                c = (0, 200, 0)
            elif i == bottom:
                # Print the road edge
                c = (255, 255, 255)
            elif i > top:
                # Print the actual road
                c = (40, 40, 40)
            elif i == top:
                # Print an edge again
                c = (255, 255, 255)
            else:
                # Print the top green pixels
                c = (0, 200, 0)
            self.background.set_at((0, i), c)

    def plot_text(self, x, y, text):
        surface = self.font.render(text, True, (0, 0, 0))
        self.screen.blit(surface, (x, y))

    def update_screen(self, position, velocity):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return True
        self.screen.blit(self.background, (0, 0))

        self.plot_text(10, 10, "Position: {:.3f}".format(float(position)))
        self.plot_text(10, 30, "Velocity: {:.3f}".format(float(velocity)))
        self.bus.rect.centerx = 75 + position * 5
        self.sprites.draw(self.screen)
        pygame.display.flip()


if __name__ == "__main__":
    pos = 0
    vel = 1
    End = False
    for i in range(201):
        End = BusViewer().update_screen(pos, vel)
        if End: break
        pos += 1
        sleep(0.05)
    sleep(10)
