import math

board = [" "] * 9
HUMAN, AI = "X", "O"

def print_board():
    for i in range(0, 9, 3):
        print(board[i], "|", board[i+1], "|", board[i+2])
    print()

def winner(p):
    wins = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8)]
    return any(board[a]==board[b]==board[c]==p for a,b,c in wins)

def draw():
    return " " not in board

def minimax(is_ai):
    if winner(AI): return 1
    if winner(HUMAN): return -1
    if draw(): return 0

    scores = []
    for i in range(9):
        if board[i] == " ":
            board[i] = AI if is_ai else HUMAN
            scores.append(minimax(not is_ai))
            board[i] = " "
    return max(scores) if is_ai else min(scores)

def ai_move():
    best, move = -math.inf, 0
    for i in range(9):
        if board[i] == " ":
            board[i] = AI
            score = minimax(False)
            board[i] = " "
            if score > best:
                best, move = score, i
    board[move] = AI

# Game loop
while True:
    print_board()
    m = int(input("Your move (0-8): "))
    board[m] = HUMAN

    if winner(HUMAN):
        print_board(); print("You win!"); break
    if draw():
        print_board(); print("Draw!"); break

    ai_move()

    if winner(AI):
        print_board(); print("AI wins!"); break