import pygame
import time
import sys
from classes import *

WHITE_BCKG = (250, 248, 239)
GRID = (187, 173, 160)
WHITE_FNT = (249, 246, 242)
BLACK_FNT = (119, 110, 101)

EMPTY_TILE = (205, 192, 180)
TWO_TILE = (238, 228, 218)
FOUR_TILE = (237, 224, 200)
EIGHT_TILE = (242, 177, 121)
SIXTEEN_TILE = (245, 149, 99)
THIRTY_TILE = (246, 124, 95)
SIXTY_TILE = (246, 94, 59)
HUNDRED_TILE = (237, 207, 114)
TH_TILE = (237, 204, 97)
FH_TILE = (237, 200, 80)
THOUSAND_TILE = (237, 197, 63)
TT_TILE = (237, 194, 46)

colours = {0: [EMPTY_TILE, BLACK_FNT], 2: [TWO_TILE, BLACK_FNT], 4: [FOUR_TILE, BLACK_FNT], 8: [EIGHT_TILE, WHITE_FNT],
           16: [SIXTEEN_TILE, WHITE_FNT], 32: [THIRTY_TILE, WHITE_FNT], 64: [SIXTY_TILE, WHITE_FNT],
           128: [HUNDRED_TILE, WHITE_FNT], 256: [TH_TILE, WHITE_FNT], 512: [FH_TILE, WHITE_FNT],
           1024: [THOUSAND_TILE, WHITE_FNT], 2048: [TT_TILE, WHITE_FNT]}



def display_state(state, score, SURFACE, myfont, scorefont):
    SURFACE.fill(WHITE_BCKG)
    pygame.draw.rect(SURFACE, GRID, (0, 100, 400, 400))
    for i in range(4):
        for j in range(4):
            nb = state[i][j]
            pygame.draw.rect(SURFACE, colours[nb][0],
                             (i * (350 / 4) + 10 * (i + 1), j * (350 / 4) + 100 + 10 * (j + 1), 350 / 4, 350 / 4))

            label = myfont.render(str(nb), 1, colours[nb][1])
            label2 = scorefont.render("Score:" + str(score), 1, BLACK_FNT)

            if nb != 0:
                h_space = 33 * (len(str(nb)) == 1) + 25 * (len(str(nb)) == 2) + 12 * (len(str(nb)) == 3) + 3 * (
                            len(str(nb)) == 4)
                SURFACE.blit(label, (i * (350 / 4) + 10 * (i + 1) + h_space, j * (350 / 4) + 125 + 10 * (j + 1)))
            SURFACE.blit(label2, (10, 20))

    pygame.display.update()
    time.sleep(0.2)


def play_game_fancy(env, QNetwork):
    pygame.init()
    SURFACE = pygame.display.set_mode((400, 500), 0, 32)
    pygame.display.set_caption("2048")
    SURFACE.fill(WHITE_BCKG)
    pygame.draw.rect(SURFACE, GRID, (0, 100, 400, 400))
    myfont = pygame.font.SysFont("monospace", 35, bold=True)
    scorefont = pygame.font.SysFont("monospace", 50)

    state = env.reset()
    episode_reward = 0
    done = False

    display_state(state, episode_reward, SURFACE, myfont, scorefont)
    while not done:


        x = change_values(state)
        x = torch.from_numpy(np.flip(x, axis=0).copy())
        output = QNetwork.forward(x)
        for action in output.argsort()[0].cpu().numpy()[::-1]:
            next_state, reward, done, _ = env.step(action)
            if (state == next_state).all() == False:
                break

        episode_reward += reward
        state = next_state
        display_state(state, episode_reward, SURFACE, myfont, scorefont)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

    return ()
