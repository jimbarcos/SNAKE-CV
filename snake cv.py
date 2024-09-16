import cv2
import mediapipe as mp
import pygame
import numpy as np
import random

# Initialize Pygame
pygame.init()

# Set up the game window
width, height = 800, 600
window = pygame.display.set_mode((width, height))
pygame.display.set_caption("Hand-Controlled Snake Game")

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

# Snake properties
snake_block = 20
snake_speed = 15

# Initialize snake
snake_list = []
snake_length = 1
snake_head = [width // 2, height // 2]

# Initialize direction
direction = 'RIGHT'

# Food
food_pos = [random.randrange(1, (width // snake_block)) * snake_block,
            random.randrange(1, (height // snake_block)) * snake_block]

# Score
score = 0
font = pygame.font.SysFont(None, 50)

# Initialize OpenCV video capture
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

def our_snake(snake_block, snake_list):
    for x in snake_list:
        pygame.draw.rect(window, GREEN, [x[0], x[1], snake_block, snake_block])

def message(msg, color):
    mesg = font.render(msg, True, color)
    window.blit(mesg, [width / 6, height / 3])

def get_hand_direction(hand_landmarks):
    # Get index finger tip and base positions
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_base = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]

    # Calculate direction vector
    dx = index_tip.x - index_base.x
    dy = index_tip.y - index_base.y

    # Determine direction based on the larger component
    if abs(dx) > abs(dy):
        return 'RIGHT' if dx > 0 else 'LEFT'
    else:
        return 'UP' if dy < 0 else 'DOWN'

def game_loop():
    global direction, snake_head, snake_list, snake_length, food_pos, score

    game_over = False
    game_close = False

    clock = pygame.time.Clock()

    while not game_over:
        while game_close:
            window.fill(BLACK)
            message("You Lost! Press C-Play Again or Q-Quit", RED)
            pygame.display.update()

            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        game_over = True
                        game_close = False
                    if event.key == pygame.K_c:
                        return game_loop()

        # Capture frame from webcam
        ret, frame = cap.read()
        if not ret:
            continue

        # Flip the frame horizontally for a later selfie-view display
        frame = cv2.flip(frame, 1)

        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame and detect hands
        results = hands.process(rgb_frame)

        # If hands are detected, update snake direction
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Get hand direction
                new_direction = get_hand_direction(hand_landmarks)
                
                # Update direction if it's not opposite to current direction
                if (new_direction == 'LEFT' and direction != 'RIGHT') or \
                   (new_direction == 'RIGHT' and direction != 'LEFT') or \
                   (new_direction == 'UP' and direction != 'DOWN') or \
                   (new_direction == 'DOWN' and direction != 'UP'):
                    direction = new_direction

                # Draw hand landmarks on the frame
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Move snake
        if direction == 'LEFT':
            snake_head[0] -= snake_block
        elif direction == 'RIGHT':
            snake_head[0] += snake_block
        elif direction == 'UP':
            snake_head[1] -= snake_block
        elif direction == 'DOWN':
            snake_head[1] += snake_block

        # Wrap-around logic
        if snake_head[0] >= width:
            snake_head[0] = 0
        elif snake_head[0] < 0:
            snake_head[0] = width - snake_block
        if snake_head[1] >= height:
            snake_head[1] = 0
        elif snake_head[1] < 0:
            snake_head[1] = height - snake_block

        # Update snake
        snake_list.append(list(snake_head))
        if len(snake_list) > snake_length:
            del snake_list[0]

        # Check self-collision
        for x in snake_list[:-1]:
            if x == snake_head:
                game_close = True

        # Check food collision
        if snake_head[0] == food_pos[0] and snake_head[1] == food_pos[1]:
            food_pos = [random.randrange(1, (width // snake_block)) * snake_block,
                        random.randrange(1, (height // snake_block)) * snake_block]
            snake_length += 1
            score += 10

        # Draw game
        window.fill(BLACK)
        pygame.draw.rect(window, RED, [food_pos[0], food_pos[1], snake_block, snake_block])
        our_snake(snake_block, snake_list)
        
        # Display score
        score_text = font.render(f"Score: {score}", True, WHITE)
        window.blit(score_text, [0, 0])

        pygame.display.update()

        # Display the resulting frame
        cv2.imshow('Hand Tracking', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            game_over = True

        clock.tick(snake_speed)

    return game_over

def main():
    game_over = False
    while not game_over:
        game_over = game_loop()

    cap.release()
    pygame.quit()

if __name__ == "__main__":
    main()
