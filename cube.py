from itertools import combinations, product, permutations
import numpy as np
from typing import List, Union, Any

from PIL import Image, ImageDraw, ImageFont


from enum import Enum

class Cubie(Enum):
    CORNERS = 0
    EDGES = 1
    BOTH = 2


# TODO
# 1. extend get_rotated_cube_state() and check_pass() (naive version) to be extendable to any amount of indexes
# 2. write a function to clean commutator string (so that it can get passed to heuristic generator)
# 3. Incorporate 4 movers into edge commutators
class Cube:
    # Define a dictionary to keep track of canceling moves
    cancel_dict = {'U': "U'", "U'": 'U', 'U2': 'U2', 'R': "R'", "R'": 'R', 'R2': 'R2',
                   'F': "F'", "F'": 'F', 'F2': 'F2', 'L': "L'", "L'": 'L', 'L2': 'L2',
                   'B': "B'", "B'": 'B', 'B2': 'B2', 'D': "D'", "D'": 'D', 'D2': 'D2',
                   'M': "M'", "M'": 'M', 'M2': 'M2', 'E': "E'", "E'": 'E', 'E2': 'E2',
                   'S': "S'", "S'": 'S', 'S2': 'S2'}

    # Define a dictionary to keep track of reducing moves
    reduce_dict = {'U2': 'U', "U'": 'U2', 'R2': 'R', "R'": 'R2',
                   'F2': 'F', "F'": 'F2', 'L2': 'L', "L'": 'L2',
                   'B2': 'B', "B'": 'B2', 'D2': 'D', "D'": 'D2',
                   'M2': 'M', "M'": 'M2', 'E2': 'E', "E'": 'E2',
                   'S2': 'S', "S'": 'S2'}
    
    # Define a dictionary to keep track of which moves do not affect which
    opposite_dict = {
        'L': ['R', 'R2', "R'"],
        'R': ['L', 'L2', "L'"],
        'U': ['D', 'D2', "D'"],
        'D': ['U', 'U2', "U'"],
        'F': ['B', 'B2', "B'"],
        'B': ['F', 'F2', "F'"],

        'L\'': ['R', 'R2', "R'"],
        'R\'': ['L', 'L2', "L'"],
        'U\'': ['D', 'D2', "D'"],
        'D\'': ['U', 'U2', "U'"],
        'F\'': ['B', 'B2', "B'"],
        'B\'': ['F', 'F2', "F'"],

        'L2': ['R', 'R2', "R'"],
        'R2': ['L', 'L2', "L'"],
        'U2': ['D', 'D2', "D'"],
        'D2': ['U', 'U2', "U'"],
        'F2': ['B', 'B2', "B'"],
        'B2': ['F', 'F2', "F'"]
        }

    heuristics = {
        "R": 0,
        "U": 0,
        "F": 1,
        "D": 0,
        "L": 2, 
        "B": 5
    }

    adjacencies = {
        0: [9, 38],
        1: [37],
        2: [36, 29],
        3: [10],
        4: [],
        5: [28],
        6: [18, 11],
        7: [19],
        8: [27, 20],
        9: [38, 0],
        10: [7],
        11: [6, 18],
        12: [41],
        13: [],
        14: [21],
        15: [51, 44],
        16: [48],
        17: [24, 45],
        18: [11, 6],
        19: [7],
        20: [8, 27],
        21: [14],
        22: [],
        23: [30],
        24: [45, 17],
        25: [46],
        26: [33, 47],
        27: [20, 8],
        28: [5],
        29: [8, 36],
        30: [23],
        31: [],
        32: [39],
        33: [47, 26],
        34: [50],
        35: [42, 53],
        36: [29, 2],
        37: [1],
        38: [0, 9],
        39: [32],
        40: [],
        41: [12],
        42: [53, 35],
        43: [52],
        44: [15, 51],
        45: [17, 24],
        46: [25],
        47: [26, 33],
        48: [16],
        49: [],
        50: [34],
        51: [44, 15],
        52: [43],
        53: [35, 42]
    }


    inverses = {
        "L": "L'",
        "L'": "L",
        "R": "R'",
        "R'": "R",
        "U": "U'",
        "U'": "U",
        "D": "D'",
        "D'": "D",
        "F": "F'",
        "F'": "F",
        "B": "B'",
        "B'": "B",
        "M": "M'",
        "M'": "M",
        "E": "E'",
        "E'": "E",
        "S": "S'",
        "S'": "S",
        "x": "x'",
        "x'": "x",
        "y": "y'",
        "y'": "y",
        "z": "z'",
        "z'": "z",
        "D2": "D2",
        "U2": "U2",
        "L2": "L2",
        "R2": "R2",
        "F2": "F2",
        "B2": "B2",
        "M2": "M2",
        "E2": "E2",
        "S2": "S2",
        "x2": "x2",
        "y2": "y2",
        "z2": "z2"
    }

    move_to_move_group = {
        "U": 0, "U'": 0, "U2": 0,
        "R": 1, "R'": 1, "R2": 1,
        "F": 2, "F'": 2, "F2": 2,
        "L": 3, "L'": 3, "L2": 3,
        "B": 4, "B'": 4, "B2": 4,
        "D": 5, "D'": 5, "D2": 5,
        "M": 6, "M'": 6, "M2": 6,
        "E": 7, "E'": 7, "E2": 7,
        "S": 8, "S'": 8, "S2": 8
    }

    move_groups = [
        ["U", "U2", "U'"],
        ["R", "R2", "R'"],
        ["F", "F2", "F'"],
        ["L", "L2", "L'"],
        ["B", "B2", "B'"],
        ["D", "D2", "D'"],  
        ["M", "M2", "M'"],
        ["E", "E2", "E'"],
        ["S", "S2", "S'"]
    ]

    move_dict = {
        "U": 0, "U'": 1, "U2": 2,
        "R": 3, "R'": 4, "R2": 5,
        "F": 6, "F'": 7, "F2": 8,
        "L": 9, "L'": 10, "L2": 11,
        "B": 12, "B'": 13, "B2": 14,
        "D": 15, "D'": 16, "D2": 17,
        "M": 18, "M'": 19, "M2": 20,
        "E": 21, "E'": 22, "E2": 23,
        "S": 24, "S'": 25, "S2": 26
    }

    idxs = np.array([
        [6,3,0,7,4,1,8,5,2,18,19,20,12,13,14,15,16,17,27,28,29,21,22,23,24,25,26,36,37,38,30,31,32,33,34,35,9,10,11,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53], # U  0
        [2,5,8,1,4,7,0,3,6,36,37,38,12,13,14,15,16,17,9,10,11,21,22,23,24,25,26,18,19,20,30,31,32,33,34,35,27,28,29,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53], # U' 1
        [8,7,6,5,4,3,2,1,0,27,28,29,12,13,14,15,16,17,36,37,38,21,22,23,24,25,26,9,10,11,30,31,32,33,34,35,18,19,20,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53], # U2 2

        [0,1,20,3,4,23,6,7,26,9,10,11,12,13,14,15,16,17,18,19,47,21,22,50,24,25,53,33,30,27,34,31,28,35,32,29,8,37,38,5,40,41,2,43,44,45,46,42,48,49,39,51,52,36], # R  3
        [0,1,42,3,4,39,6,7,36,9,10,11,12,13,14,15,16,17,18,19,2,21,22,5,24,25,8,29,32,35,28,31,34,27,30,33,53,37,38,50,40,41,47,43,44,45,46,20,48,49,23,51,52,26], # R' 4
        [0,1,47,3,4,50,6,7,53,9,10,11,12,13,14,15,16,17,18,19,42,21,22,39,24,25,36,35,34,33,32,31,30,29,28,27,26,37,38,23,20,41,42,43,44,45,46,2,48,49,5,51,52,8], # R2 5

        [0,1,2,3,4,5,17,14,11,9,10,45,12,13,46,15,16,47,24,21,18,25,22,19,26,23,20,6,28,29,7,31,32,8,34,35,36,37,38,39,40,41,42,43,44,33,30,27,48,49,50,51,52,53], # F  6
        [0,1,2,3,4,5,27,30,33,9,10,8,12,13,7,15,16,6,20,23,26,19,22,25,18,21,24,47,28,29,46,31,32,45,34,35,36,37,38,39,40,41,42,43,44,11,14,17,48,49,50,51,52,53], # F' 7
        [0,1,2,3,4,5,47,46,45,9,10,33,12,13,30,15,16,27,26,25,24,23,22,21,20,19,18,17,28,29,14,31,32,11,34,35,36,37,38,39,40,41,42,43,44,8,7,6,48,49,50,51,52,53], # F2 8

        [44,1,2,41,4,5,38,7,8,15,12,9,16,13,10,17,14,11,0,19,20,3,22,23,6,25,26,27,28,29,30,31,32,33,34,35,36,37,51,39,40,48,42,43,45,18,46,47,21,49,50,24,52,53], # L  9
        [18,1,2,21,4,5,24,7,8,11,14,17,10,13,16,9,12,15,45,19,20,48,22,23,51,25,26,27,28,29,30,31,32,33,34,35,36,37,6,39,40,3,42,43,0,44,46,47,41,49,50,38,52,53], # L' 10
        [45,1,2,48,4,5,51,7,8,17,16,15,14,13,12,11,10,9,44,19,20,41,22,23,38,25,26,27,28,29,30,31,32,33,34,35,36,37,24,39,40,21,42,43,18,0,46,47,3,49,50,6,52,53], # L2 11

        [29,32,35,3,4,5,6,7,8,2,10,11,1,13,14,0,16,17,18,19,20,21,22,23,24,25,26,27,28,53,30,31,52,33,34,51,42,39,36,43,40,37,44,41,38,45,46,47,48,49,50,9,12,15], # B  12
        [15,12,9,3,4,5,6,7,8,51,10,11,52,13,14,53,16,17,18,19,20,21,22,23,24,25,26,27,28,0,30,31,1,33,34,2,38,41,44,37,40,43,36,39,42,45,46,47,48,49,50,35,32,29], # B' 13
        [53,52,51,3,4,5,6,7,8,35,10,11,32,13,14,29,16,17,18,19,20,21,22,23,24,25,26,27,28,15,30,31,12,33,34,9,44,43,42,41,40,39,38,43,36,45,46,47,48,49,50,2,1,0], # B2 14

        [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,42,43,44,18,19,20,21,22,23,15,16,17,27,28,29,30,31,32,24,25,26,36,37,38,39,40,41,33,34,35,51,48,45,52,49,46,53,50,47], # D  15
        [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,24,25,26,18,19,20,21,22,23,33,34,35,27,28,29,30,31,32,42,43,44,36,37,38,39,40,41,15,16,17,47,50,53,46,49,52,45,48,51], # D' 16
        [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,33,34,35,18,19,20,21,22,23,42,43,44,27,28,29,30,31,32,15,16,17,36,37,38,39,40,41,24,25,26,53,52,51,50,49,48,47,46,45], # D2 17

        [0,43,2,3,40,5,6,37,8,9,10,11,12,13,14,15,16,17,18,1,20,21,4,23,24,7,26,27,28,29,30,31,32,33,34,35,36,52,38,39,49,41,42,46,44,45,19,47,48,22,50,51,25,53], # M  18
        [0,19,2,3,22,5,6,25,8,9,10,11,12,13,14,15,16,17,18,46,20,21,49,23,24,52,26,27,28,29,30,31,32,33,34,35,36,7,38,39,4,41,42,1,44,45,43,47,48,40,50,51,37,53], # M' 19
        [0,46,2,3,49,5,6,52,8,9,10,11,12,13,14,15,16,17,18,43,20,21,40,23,24,37,26,27,28,29,30,31,32,33,34,35,36,25,38,39,22,41,42,19,44,45,1,47,48,4,50,51,7,53], # M2 20

        [0,1,2,3,4,5,6,7,8,9,10,11,39,40,41,15,16,17,18,19,20,12,13,14,24,25,26,27,28,29,21,22,23,33,34,35,36,37,38,30,31,32,42,43,44,45,46,47,48,49,50,51,52,53], # E  21  
        [0,1,2,3,4,5,6,7,8,9,10,11,21,22,23,15,16,17,18,19,20,30,31,32,24,25,26,27,28,29,39,40,41,33,34,35,36,37,38,12,13,14,42,43,44,45,46,47,48,49,50,51,52,53], # E' 22
        [0,1,2,3,4,5,6,7,8,9,10,11,30,31,32,15,16,17,18,19,20,39,40,41,24,25,26,27,28,29,12,13,14,33,34,35,36,37,38,21,22,23,42,43,44,45,46,47,48,49,50,51,52,53], # E2 23

        [0,1,2,16,13,10,6,7,8,9,48,11,12,49,14,15,50,17,18,19,20,21,22,23,24,25,26,27,3,29,30,4,32,33,5,35,36,37,38,39,40,41,42,43,44,45,46,47,34,31,28,51,52,53], # S  24
        [0,1,2,28,31,34,6,7,8,9,5,11,12,4,14,15,3,17,18,19,20,21,22,23,24,25,26,27,50,29,30,49,32,33,48,35,36,37,38,39,40,41,42,43,44,45,46,47,10,13,16,51,52,53], # S' 25
        [0,1,2,50,49,48,6,7,8,9,34,11,12,31,14,15,28,17,18,19,20,21,22,23,24,25,26,27,16,29,30,13,32,33,10,35,36,37,38,39,40,41,42,43,44,45,46,47,5,4,3,51,52,53], # S2 26
    ])

    solved = (np.arange(0, 6, 1, dtype=np.ushort).reshape((6, 1)) +
              np.zeros((6, 9), dtype=np.ushort)).flatten()

    def __init__(self):
        self.state = Cube.solved.copy()

    def move(self, moves):
        try:
            moves = [Cube.move_dict[move] for move in moves]
        except KeyError as e:
            print(f"Invalid move: {e}")
        for move in moves:
            self.state = self.state[Cube.idxs[move]] 

    @staticmethod
    def move_from_solved(moves):
        try:
            moves = [Cube.move_dict[move] for move in moves]
        except KeyError as e:
            print(f"Invalid move: {e}")
        state = Cube.solved
        for move in moves:
            state = state[Cube.idxs[move]] 
        return state

    def __str__(self):
        color_map = {0: 'G', 1: 'Y', 2: 'R', 3: 'W', 4: 'O', 5: 'B'}
        faces = [''.join([color_map[color] for color in self.state[i:i + 9]]) for i in range(0, 54, 9)]
        
        result = []
        result.append(" "*3 + faces[0][:3] + " "*3 + "\n")
        result.append(" "*3 + faces[0][3:6] + " "*3 + "\n")
        result.append(" "*3 + faces[0][6:] + " "*3 + "\n")
        result.append(faces[1][:3] + faces[2][:3] + faces[3][:3] + faces[4][:3] + "\n")
        result.append(faces[1][3:6] + faces[2][3:6] + faces[3][3:6] + faces[4][3:6] + "\n")
        result.append(faces[1][6:] + faces[2][6:] + faces[3][6:] + faces[4][6:] + "\n")
        result.append(" "*3 + faces[5][:3] + " "*3 + "\n")
        result.append(" "*3 + faces[5][3:6] + " "*3 + "\n")
        result.append(" "*3 + faces[5][6:] + " "*3 + "\n")
        
        return ''.join(result)
    
    def display(self):
        self.display_from_state(self.state)

    @staticmethod
    def display_from_state(state):
        # Define the size of each square and the overall image
        square_size = 50
        image_size = (square_size * 12, square_size * 9)

        # Create a new image with a white background
        image = Image.new('RGB', image_size, (255, 255, 255))
        draw = ImageDraw.Draw(image)

        # Define a font (you may need to download or specify the path to a .ttf or .otf font file on your system)
        font = ImageFont.truetype("arial.ttf", 15)

        # Define the color map as RGB values
        color_map = {0: (0, 255, 0),    # Green
                    1: (255, 255, 0),  # Yellow
                    2: (255, 0, 0),    # Red
                    3: (255, 255, 255),# White
                    4: (255, 165, 0),  # Orange
                    5: (0, 0, 255)}    # Blue

        # Calculate the positions of the squares
        positions = [
            [(3, 0), (4, 0), (5, 0), (3, 1), (4, 1), (5, 1), (3, 2), (4, 2), (5, 2)],  # top face
            [(0, 3), (1, 3), (2, 3), (0, 4), (1, 4), (2, 4), (0, 5), (1, 5), (2, 5)],  # left face
            [(3, 3), (4, 3), (5, 3), (3, 4), (4, 4), (5, 4), (3, 5), (4, 5), (5, 5)],  # front face
            [(6, 3), (7, 3), (8, 3), (6, 4), (7, 4), (8, 4), (6, 5), (7, 5), (8, 5)],  # right face
            [(9, 3), (10, 3), (11, 3), (9, 4), (10, 4), (11, 4), (9, 5), (10, 5), (11, 5)],  # back face
            [(3, 6), (4, 6), (5, 6), (3, 7), (4, 7), (5, 7), (3, 8), (4, 8), (5, 8)]   # bottom face
        ]

        # Draw each square in the image
        for face_index in range(6):
            for square_index in range(9):
                index = face_index*9 + square_index
                x, y = positions[face_index][square_index]
                color = color_map[state[index]]
                draw.rectangle([x*square_size, y*square_size, (x+1)*square_size, (y+1)*square_size], fill=color, outline=(0, 0, 0))
                # Draw the index number in the center of the square
                text = str(index)
                text_width, text_height = draw.textsize(text, font=font)
                text_x = x * square_size + (square_size - text_width) / 2
                text_y = y * square_size + (square_size - text_height) / 2
                draw.text((text_x, text_y), text, fill=(0, 0, 0), font=font)

        # Save the image
        image.show()

    # TODO
    # also check so that opposite arent being put together
    @staticmethod
    def get_commutator_triplets(cubie: Cubie):
        move_groups = Cube.move_groups
        if cubie == Cubie.CORNERS:
            move_groups = move_groups[:-3]

        triplets = []
        for group_triplet in combinations(move_groups, 3):
            for triplet_combo in product(*group_triplet):
                for triplet in permutations(triplet_combo):
                    if cubie == Cubie.CORNERS or cubie == Cubie.BOTH:
                        triplets.append(list(triplet))
                    
                    # edge commutators require a middle slice move by nature
                    elif cubie == Cubie.EDGES and any(move in ["M", "M'", "M2", "E", "E'", "E2", "S", "S'", "S2"] for move in triplet): 
                        triplets.append(list(triplet))

        return triplets
    
    
    @staticmethod
    def get_commutators(cubie: Cubie, degree=0):
        triplets = Cube.get_commutator_triplets(cubie)
        setups = Cube.generate_setups(degree, cubie)
        commutators = []
        for triplet in triplets:
            for setup in setups:
                commutator = Cube.make_commutator(triplet, setup)
                commutators.append(commutator)
        return commutators

    @staticmethod
    def group_of_move(move):
        """Return the group that contains the move"""
        for group in Cube.move_groups:
            if move in group:
                return group
        return None

    @staticmethod
    def is_valid_setup(setup):
        """Checks if any two consecutive moves come from the same group"""
        for i in range(len(setup) - 1):
            if Cube.group_of_move(setup[i]) == Cube.group_of_move(setup[i+1]):
                return False
        return True

    @staticmethod
    def generate_setups(n, cubie: Cubie):
        all_moves = list(Cube.move_dict.keys())

        if cubie == Cubie.CORNERS:
            all_moves = all_moves[:-9]

        if n > len(all_moves):
            raise ValueError("n cannot be greater than number of moves")

        move_setups = list(product(all_moves, repeat=n))
        valid_setups = [setup for setup in move_setups if Cube.is_valid_setup(setup)]

        return valid_setups
    
    @staticmethod
    def make_commutator(triplet, setup=None):
        a,b,c = triplet
        commutator = [a, b, Cube.inverses[a], c, a, Cube.inverses[b], Cube.inverses[a], Cube.inverses[c]] # X Y X' Y'
        if setup:
            for move in setup[::-1]:
                commutator.insert(0, move)
                commutator.append(Cube.inverses[move])

        return commutator
    

    @staticmethod
    def get_rotated_cube_array(indexes: List[List[int]]) -> np.ndarray:
        """
        This function takes a list of indexes, and returns a new array where the values at these 
        indexes and their adjacent indexes (according to Cube.adjacencies) are circularly rotated. 
        It returns an error string if the adjacency lists for the provided indexes are not all the same length.
        
        Parameters:
        indexes (List[int]): List of indexes to be rotated.

        Returns:
        new_array (np.ndarray): The array after rotation.
        or 
        str: Error message if adjacency lists do not have the same length.
        """
        new_array = Cube.solved
        for subindexes in indexes:
            if len(subindexes) > 1: # only if 2 or more pieces are iven to be rotated
                # Perform the rotation for the given indexes
                new_array[subindexes] = np.roll(new_array[subindexes], 1)
                # Retrieve adjacency tuples
                adjacencies = [Cube.adjacencies[i] for i in subindexes]
                # Check if all adjacency tuples have the same length
                if len(set(map(len, adjacencies))) > 1:
                    raise ValueError("Adjacency lists do not have the same length. Please make sure the indexes are all of the same cubie type.")

                # Perform the rotation for the adjacencies only if
                for i in range(len(adjacencies[0])):
                    adj_indexes = [adj[i] for adj in adjacencies]
                    new_array[adj_indexes] = np.roll(new_array[adj_indexes], 1)

        return new_array

    # TODO
    # change to incorporate multiple lists of int, not just one, within indexes
    @staticmethod
    def check_pass(indexes: List[List[int]], commutator, naive=True):
        moved_state = Cube.move_from_solved(commutator)
        state_to_match = Cube.get_rotated_cube_array(indexes)

        if not naive:  # check that ONLY the specified cubies have been moved
            return np.array_equal(moved_state, state_to_match)
        else:  # allow other cubies to be moved as well
            for subindexes in indexes:
                if not all(moved_state[i] == state_to_match[i] for i in subindexes):
                    return False

                # get all the adjacent subindexes
                adjacencies = [item for sub in subindexes for item in Cube.adjacencies[sub]]  
                grouped_adjacencies = zip(*[iter(adjacencies)] * len(subindexes))

                for adj_group in grouped_adjacencies:
                    if not all(moved_state[idx] == state_to_match[idx] for idx in adj_group):
                        return False
            else:
                return True

        
    # @staticmethod
    # def check_pass(pos1, pos2, pos3, commutator, is_corners):
    #     moved_state = Cube.move_from_solved(commutator)

    #     passed = moved_state[pos1] == Cube.solved[pos3] and moved_state[pos2] == Cube.solved[pos1] and moved_state[pos3] == Cube.solved[pos2]
    #     if passed:
    #         passed = moved_state[Cube.adjacencies[pos1][0]] == Cube.solved[Cube.adjacencies[pos3][0]] and moved_state[Cube.adjacencies[pos2][0]] == Cube.solved[Cube.adjacencies[pos1][0]] and moved_state[Cube.adjacencies[pos3][0]] == Cube.solved[Cube.adjacencies[pos2][0]]
    #         if is_corners and passed:
    #             passed = moved_state[Cube.adjacencies[pos1][1]] == Cube.solved[Cube.adjacencies[pos3][1]] and moved_state[Cube.adjacencies[pos2][1]] == Cube.solved[Cube.adjacencies[pos1][1]] and moved_state[Cube.adjacencies[pos3][1]] == Cube.solved[Cube.adjacencies[pos2][1]]
    #     return passed
        

    @staticmethod
    def get_heuristic(commutator: List[str]) -> int:
        return sum([Cube.heuristics[move[0]] for move in Cube.reduce_commutator(commutator)])

    @staticmethod
    def clean_commutator(commutator: List[str]) -> str:
        return ' '.join([move.ljust(2) for move in commutator])
    
    # TODO
    # make this neater!
    @staticmethod
    def reduce_commutator(commutator: List[str]) -> List[str]:
        reduction_dict = {
            " ": {"'": None, "2": "'", " ": "2"},
            "'": {" ": None, "2": "", "'": "2"},
            "2": {" ": "'", "'": "", "2": None}
        }

        if len(commutator) == 8:
            return commutator
        
        not_reduced = True
        new_commutator = commutator.copy()
        while not_reduced:
            current_round = new_commutator
            for i in range(len(new_commutator)-1):
                # pad with empty space
                move_a = new_commutator[i] + " "
                move_b = new_commutator[i+1] + " "
                if move_a[0] == move_b[0]: # same move type, e.g. U' and U
                    if reduction_dict[move_a[1]][move_b[1]] is not None:
                        new_commutator[i] = move_a[0] + reduction_dict[move_a[1]][move_b[1]]
                        new_commutator[i+1] = '#'
                    else: # completely cancel out both moves
                        new_commutator[i] = '#'
                        new_commutator[i+1] = '#'
            new_commutator = [move for move in new_commutator if '#' not in move]


            if new_commutator != current_round:
                new_commutator == current_round
            else:
                not_reduced = False

        return new_commutator
    
if __name__ == "__main__":
    ...