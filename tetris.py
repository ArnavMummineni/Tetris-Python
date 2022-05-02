from copy import deepcopy
from math import factorial
import random

import pygame
import numpy as np
import pygame.surfarray as surfarray

pygame.init()


screen = None

def init_window(array):
    global screen
    screen = pygame.display.set_mode(array.shape[:2])
    update_window(array)


def update_window(array):
    surfarray.blit_array(screen, array)
    pygame.display.update()


def update_pf():
    for i in range(pf_shown[0]):
        for j in range(pf_shown[1]):
            block_object = pf[i + len(pf) - pf_shown[0], j + len(pf[0]) - pf_shown[1]]
            if not(block_object) and block_object in tetromino_shapes:
                renderer = tetromino_shapes[block_object].get('renderer')
            elif hasattr(block_object, 'renderer'):
                renderer = block_object.renderer
            else:
                renderer = default_renderer
            set_block(renderer(block_object, (i, j)), (i, j))
    update_window(scr_pf)


def brick(tetromino, *args, shaded=True, **kwargs):
    if 'color' in kwargs:
        color = kwargs['color']
    elif hasattr(tetromino, 'color'):
        color = tetromino.color
    else:
        color = default_color
    brick = np.zeros((sf, sf, 3))
    brick[:, :] = color
    if not shaded:
        return brick
    if not (color == (color * shade_factor)).all():
        for i in range(1, shadow_thickness + 1):
            brick[i:, sf-i] = color * shade_factor
            brick[sf-i, i:] = color * shade_factor
    return brick


def default_outline(*args, color=None):
    brick = np.zeros((sf, sf, 3))
    gw = ghost_width = sf * ghost_percent // 100
    brick[:, :] = default_outline_color if color is None else color
    brick[gw:-gw, gw:-gw] = pf_background
    return brick


def set_block(pixels, loc):
    scr_pf[loc[0]*sf:loc[0]*sf+sf, loc[1]*sf:loc[1]*sf+sf, :] = pixels


#Possibly borken
def fill_random():
    for i in range(pf.shape[0]):
        for j in range(pf.shape[1]):
            pf[i, j] = colors[color_names[random.randrange(len(color_names))]]
    return brick

#Borken
def rainbow(*args):
    offset = (pygame.time.get_ticks() // 100) % 255
    brick = np.zeros((sf, sf, 3))
    for i in range(sf):
        brick[i, :] = np.array([offset + (10 * i)] * 3)
    return brick


colors = {
    'cyan': np.array([0, 255, 255]),
    'blue': np.array([0, 0, 255]),
    'orange': np.array([255, 127, 0]),
    'yellow': np.array([255, 255, 0]),
    'green': np.array([0, 255, 0]),
    'purple': np.array([255, 0, 255]),
    'red': np.array([255, 0, 0]),
    'black': np.array([0, 0, 0]),
    'white': np.array([255, 255, 255]),
}

color_names = list(colors.keys())

sf = 32
shade_factor = 0.7
shadow_thickness = 3
ghost_percent = 15

pf = np.zeros((10, 23), dtype=object)
pf[:, :] = None
pf_shown = [10, 20]
scr_pf = np.zeros([pf_shown[0] * sf, pf_shown[1] * sf, 3], dtype=np.uint8)
pf_background = colors['black']

default_renderer = lambda tetromino, *args: brick(tetromino, shaded=True)
default_outline_function = default_outline
default_color = colors['white']
default_outline_color = colors['white']

side_walls = True
disable_hold_limit = False

default_wall_kicks = {
    'cw': {
        0: ((0,0),(-1,0),(-1,1),(0,-2),(-1,-2)),
        1: ((0,0),(1,0),(1,-1),(0,2),(1,2)),
        2: ((0,0),(1,0),(1,1),(0,-2),(1,-2)),
        3: ((0,0),(-1,0),(-1,-1),(0,2),(-1,2)),
    },
    'ccw': {
        0: ((0,0),(1,0),(1,1),(0,-2),(1,-2)),
        1: ((0,0),(1,0),(1,-1),(0,2),(1,2)),
        2: ((0,0),(-1,0),(-1,1),(0,-2),(-1,-2)),
        3: ((0,0),(-1,0),(-1,-1),(0,2),(-1,2)),
    }
}

tetromino_shapes = {
    None: {
        'color': pf_background,
        'renderer': lambda *args: brick(None, color=pf_background, shaded=False),
    },
    'I': {
        'blocks': [
            [0, 0, 0, 0],
            [1, 1, 1, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        'color': colors['cyan'],
        'wall_kicks': {
            'cw': {
                0: ((0,0),(-2,0),(1,0),(-2,-1),(1,2)),
                1: ((0,0),(-1,0),(2,0),(-1,2),(2,-1)),
                2: ((0,0),(2,0),(-1,0),(2,1),(-1,-2)),
                3: ((0,0),(1,0),(-2,0),(1,-2),(-2,1)),
            },
            'ccw': {
                0: ((0,0),(-1,0),(2,0),(-1,2),(2,-1)),
                1: ((0,0),(2,0),(-1,0),(2,1),(-1,-2)),
                2: ((0,0),(1,0),(-2,0),(1,-2),(-2,1)),
                3: ((0,0),(-2,0),(1,0),(-2,-1),(1,2)),
            },
        },
    },
    'J': {
        'blocks': [
            [1, 0, 0],
            [1, 1, 1],
            [0, 0, 0],
        ],
        'color': colors['blue'],
    },
    'L': {
        'blocks': [
            [0, 0, 1],
            [1, 1, 1],
            [0, 0, 0],
        ],
        'color': colors['orange'],
    },
    'O': {
        'blocks': [
            [1, 1],
            [1, 1],
        ],
        'color': colors['yellow'],
        'position': [4, 0],
    },
    'S': {
        'blocks': [
            [0, 1, 1],
            [1, 1, 0],
            [0, 0, 0],
        ],
        'color': colors['green'],
    },
    'T': {
        'blocks': [
            [0, 1, 0],
            [1, 1, 1],
            [0, 0, 0],
        ],
        'color': colors['purple'],
    },
    'Z': {
        'blocks': [
            [1, 1, 0],
            [0, 1, 1],
            [0, 0, 0],
        ],
        'color': colors['red'],
    },
}

tetromino_list = [t for t in list(tetromino_shapes.keys()) if t]



def remove_line(line_index):
    pf[:, 1:line_index + 1] = pf[:, :line_index]
    pf[:, 0] = None
    return True


class GhostTetromino():
    renderer = default_outline
    def __init__(self, parent):
        self.parent = parent


class Tetromino():

    @staticmethod
    def get_random_tetromino():
        type_ = random.randrange(len(tetromino_list))
        name = tetromino_shapes.get(tetromino_list[type_])
        if name is None:
            tetromino_list.remove(name)
            return Tetromino.get_random_tetromino()
        return Tetromino.get_tetromino_instance_from_name(tetromino_list[type_])

    @staticmethod
    def get_randomized_bag():
        bag = tetromino_list.copy()
        random.shuffle(bag)
        for index, name in enumerate(bag):
            bag[index] = Tetromino.get_tetromino_instance_from_name(name)
        return bag

    @staticmethod
    def get_tetromino_instance_from_name(name):
        if tetromino_shapes[name].get('class') is not None:
            kwargs = tetromino_shapes[name].get('kwargs', {})
            return tetromino_shapes[name].get('class')(**kwargs)
        if tetromino_shapes[name].get('blocks') is not None:
            return Tetromino(**{**tetromino_shapes[name], 'name': name})
        return None
    
    def __init__(
            self,
            blocks,
            position=None,
            wall_kicks='default',
            max_rotations=4,
            color=None,
            renderer=brick,
            name=None,
            **kwargs):
        self.blocks = np.array(blocks)
        self.position = position if position is not None else [3, 0]
        self.kicks = wall_kicks if wall_kicks != 'default' else default_wall_kicks
        self.drawn = False
        self.rotation = 0
        self.max_rotations = max_rotations
        self.renderer = renderer
        self.color = default_color if color is None else color
        self.ghost = GhostTetromino(self)
        self.name = name or 'Unnamed'
        self.kwargs = kwargs
        self.original = deepcopy(self)

    def rotate_cw(self, draw=None):
        return self._rotate_for_kicks(1, 'cw', draw)

    def rotate_ccw(self, draw=None):
        return self._rotate_for_kicks(-1, 'ccw', draw)

    def _rotate_for_kicks(self, id_change, kick_id, draw=None):
        draw = draw or self.drawn
        after = np.rot90(self.blocks, axes=(1, 0), k=id_change)
        for kick in self.kicks[kick_id][self.rotation]:
            #print('kick:', kick)
            if self._confirm_changes(self.blocks, after, kick, draw):
                self.rotation += id_change
                self.rotation %= self.max_rotations
                return True
        return False
    
    def move_down(self, draw=None):
        return self._confirm_changes(self.blocks, self.blocks, (0, 1), draw)

    def move_left(self, draw=None):
        return self._confirm_changes(self.blocks, self.blocks, (-1, 0), draw)
    
    def move_right(self, draw=None):
        return self._confirm_changes(self.blocks, self.blocks, (1, 0), draw)

    @classmethod
    def _get_block_changes(cls, start, end, offset=(0, 0)):
        start = set(zip(*np.where(start)[::-1]))
        end = zip(*np.where(end)[::-1])
        end = {(pos[0] + offset[0], pos[1] + offset[1]) for pos in end}
        #print(end, {pos for pos in end if pos not in start})
        return {pos for pos in end if pos not in start}

    def _confirm_changes(self, start, end, offset, draw=None):
        draw = draw or self.drawn
        additions = self._get_block_changes(start, end, offset)
        if all(self._rel_pixel_isempty(addition) for addition in additions):
            if draw:
                removals = self._get_block_changes(end, start, [-dim for dim in offset])
                for addition in additions: self._set_rel_pixel(self, addition)
                self.position = [pos + off for pos, off in zip(self.position, offset)]
                for removal in removals: self._set_rel_pixel(None, removal)
            self.blocks = end
            return True
        return False

    def _rel_pixel_isempty(self, offset, offset_y=None):
        x, y = self._get_abs_pixel(offset, offset_y)
        if (side_walls and 
                not (0 <= x < pf.shape[0]
                and  0 <= y < pf.shape[1])):
            return False
        return (not pf[x, y]) or pf[x, y] is self.ghost or pf[x, y] is self

    def _set_rel_pixel(self, type_, offset, offset_y=None):
        x, y = self._get_abs_pixel(offset, offset_y)
        pf[x, y] = type_

    def _get_abs_pixel(self, offset, offset_y=None, pos=None):
        if offset_y is None: offset_x, offset_y = offset
        else: offset_x, offset_y = offset, offset_y
        pos = self.position if pos is None else pos
        return pos[0]+offset_x, pos[1]+offset_y

    def draw(self, erase=False):
        blocks = tuple(zip(*np.where(self.blocks)[::-1]))
        self.drawn = not bool(erase)
        if erase: self.clear_ghost()
        if all(self._rel_pixel_isempty(pos) for pos in blocks):
            #print(blocks)
            for pos in blocks:
                if not erase: self._set_rel_pixel(self, pos)
                else: self._set_rel_pixel(None, pos)
            return True
        return False

    def clear_ghost(self):
        pf[pf == self.ghost] = None

    def set_ghost(self):
        self.clear_ghost()
        pos_list = tuple(self._get_abs_pixel(pos) for pos in zip(*np.where(self.blocks)[::-1]))
        drop_list = [0] * len(pos_list)
        for i, pos in enumerate(pos_list):
            for block in pf[pos[0], pos[1] + 1:]:
                if not block or block == self: drop_list[i] += 1
                else: break
        min_drop = min(drop_list)
        for pos in ((y, x + min_drop) for y, x in pos_list):
            pf[pos] = self.ghost if pf[pos] != self else self

    def lock(self):
        return sum(remove_line(row)
                   for row in sorted(list(set(
                       self.position[1] + row
                       for row in set(np.where(self.blocks)[0])
                   )))
                   if all(pf[:, row]))
    
    def is_all_hidden(self):    
        return all(
            True in (pos[i] < pf.shape[i] - pf_shown[i] for i in (0, 1))
            for pos in (
                current_piece._get_abs_pixel(pos)
                for pos in zip(*np.where(current_piece.blocks)[::-1])
                )
            )

    def get_original(self):
        return deepcopy(self.original)



if __name__ == '__main__':

    init_window(scr_pf)

    level = 0
    last_level_lines = 0
    lines_per_level_func = lambda level: 10 * level

    points = 0
    points_per_clear_func = lambda lines: 50 * factorial(lines) * (level + 1)
    zone_bonus_func = lambda lines: 100 * lines

    fall_rate_Gs = 1/60
    fall_rate_ms = (1000 / 60) / fall_rate_Gs
    fall_rate_Gs_func = lambda level: 1 / (30 - min(29, level))
    last_fall = pygame.time.get_ticks()
    
    
    cleared_lines = 0
    zone_time = 17500
    zone = 0
    zone_active = False
    this_zone_cleared = 0
    zone_clear_names = {
        1: 'Single',
        2: 'Double',
        3: 'Triple',
        4: 'Tetris',
        8: 'Octoris',
        10: 'Decatris',
        12: 'Dodecatris',
        16: 'Decahexatris',
        18: 'Perfectris',
        20: 'Ultimatris',
        21: 'Kirbtris',
    }

    hold = None
    used_hold = False

    game_over = False
    
    def end():
        global game_over
        game_over = True
 
    '''
    T = get_tetromino_instance_from_name('T')
    current_piece = T
    T.draw()
    T.move_down()
    T.move_down()
    T.move_down()
    T._set_rel_pixel(None, 0, 0)
    T._set_rel_pixel(None, 1, -1)
    T._set_rel_pixel(None, 2, 0)
    '''
    update_pf()
    
    def next_piece(bag):
        if len(bag) == 0:
            bag = Tetromino.get_randomized_bag()
        piece = bag.pop(0)
        if not piece.draw(): end()
        return piece, bag
    
    current_piece, bag = next_piece([])

    def lock_current_piece():
        global cleared_lines, used_hold, zone, this_zone_cleared, bag, points
        used_hold = False
        if current_piece.is_all_hidden(): end()
        this_cleared = current_piece.lock()
        if zone_active:
            this_zone_cleared += this_cleared
            return next_piece(bag)
        cleared_lines += this_cleared
        zone = min(zone + this_cleared, 16)
        if this_cleared >= min(zone_clear_names):
            print(zone_clear_names[max([k for k in zone_clear_names if k <= this_cleared])])
        if this_cleared > 0:
            print('Total lines cleared:', cleared_lines)
        check_increase_level()
        points += points_per_clear_func(this_cleared)
        return next_piece(bag)

    def check_increase_level():
        global level, last_level_lines
        if cleared_lines - last_level_lines >= lines_per_level_func(level):
            last_level_lines += lines_per_level_func(level)
            level += 1
            fall_rate_Gs = fall_rate_Gs_func(level)
            return True
        return False

    def swap_hold():
        global hold, used_hold, current_piece, bag
        if used_hold: print('Hold failed'); return False
        used_hold = True
        current_piece.draw(erase=True)
        if hold is None:
            hold = current_piece
            current_piece, bag = next_piece(bag)
            return True
        hold, current_piece = current_piece, hold.get_original()
        if not current_piece.draw(): end()
        return True

    def print_info():
        global bag
        print('\n\n\n')
        print('Lines cleared:', cleared_lines)
        print('Level:', level)
        print('Points:', points)
        print('Zone:', zone)
        if hold is not None:
            print('Hold:', hold.name)
        while len(bag) <= 3:
            bag += Tetromino.get_randomized_bag()
        print(f'Next 3 pieces: {bag[0].name}, {bag[1].name}, and {bag[2].name}')

    
    while not game_over:
        
        if zone_active:
            if pygame.time.get_ticks() - last_fall > zone_time:
                zone_active = False
                print(zone_clear_names[max([k for k in zone_clear_names if k <= this_zone_cleared])])
                print('Total lines cleared:', cleared_lines)
                points += zone_bonus_func(this_zone_cleared)
                this_zone_cleared = 0
                check_increase_level()
        elif pygame.time.get_ticks() - last_fall > fall_rate_ms:
            if not current_piece.move_down():
                current_piece, bag = lock_current_piece()
            last_fall = pygame.time.get_ticks()
            update_pf()
        
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_DOWN:
                    if not current_piece.move_down():
                        current_piece, bag = lock_current_piece()
                    pass
                elif event.key == pygame.K_LEFT:
                    current_piece.move_left()
                elif event.key == pygame.K_RIGHT:
                    current_piece.move_right()
                elif event.key == pygame.K_x:
                    current_piece.rotate_cw()
                elif event.key == pygame.K_z:
                    current_piece.rotate_ccw()
                elif event.key == pygame.K_SPACE:
                    while current_piece.move_down(): pass
                    current_piece, bag = lock_current_piece()
                elif event.key == pygame.K_c:
                    swap_hold()
                elif event.key == pygame.K_e:
                    print(current_piece.draw())
                elif event.key == pygame.K_v:
                    if zone < 16:
                        print('Current zone level insufficient (max=16):', zone)
                    else:
                        print('Zone activated')
                        zone_active = True
                        last_fall = pygame.time.get_ticks()
                        zone = 0
                elif event.key == pygame.K_a:
                    print_info()
                else:
                    break
            else:
                break
        else:
            current_piece.set_ghost()
            update_pf()

    print(cleared_lines, 'lines cleared')
