import numpy as np
import random
import matris
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.matris = matris.Matris()

        self.state_size = state_size
        self.action_size = action_size

        self.memory = deque(maxlen=2000)

        self.gamma = 0.95  # discount rate
        self.epsilon = 1.  # exploration rate

        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.Q_max_model = self.Q_build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(48, input_dim=self.state_size, activation='relu'))
        model.add(Dense(48, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def Q_build_model(self):
        model = Sequential()
        model.add(Dense(48, input_dim=self.state_size, activation='relu'))
        model.add(Dense(48, activation='relu'))
        model.add(Dense(40, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def replay(self, batch_size, model=None):    # train
        
        if model is None:
            minibatch = random.sample(self.memory, batch_size)
            for state, tetromino, action, reward, new_state, done in minibatch:
                target = reward

                target_f = self.model.predict(state)
                res = self.model.predict(new_state)
                if not done:
                    target += self.gamma * np.amax(res)
                target_f[0][action] = target

            

                # 根据公式更新Q表
                # Q(s,a) = R(s,a)+λmax{Q(s`,a`)}
                # target_f:当前状态的Q值 target:下一状态的Q值

                self.model.fit(state, target_f, epochs=1, verbose=0)
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
        else:
            minibatch = random.sample(self.memory, batch_size)
            for state, tetromino, action, reward, new_state, done in minibatch:
                target = reward

                target_f = self.Q_max_model.predict(state)
                res = self.Q_max_model.predict(new_state)
                if not done:
                    target += self.gamma * np.amax(res)
                target_f[0][action] = target

                self.Q_max_model.fit(state, target_f, epochs=1, verbose=0)
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay



    def Q(self, state, piece):
        """
        Finds the 
        :param state: A dict that represents the tetris board (with shadow
               piece)
               
        :return: An array representing the "scores" of each move 
                 (Will be none if another move is not possible)
        """
        
        try:  
            game_matrix = self.matris.matrix
            game_matrix = self.matris.blend(shape=piece.shape,matrix=game_matrix)
            
        #self.encoder_net.forward(self.encoded_state(game_matrix))
        #encoded_state = self.encoder_net.outputs_cache[1]
        #return self.net.forward(encoded_state)
            return self.model.predict(np.reshape(self.matris.dict_to_matrix(game_matrix), [1, state_size]))
        except:
            return None

    def Q_max(self, state, piece):
        """
        State - a matris board dict
        Piece - a tetromino object (not the shape)
        
        Given a game state, this function
        will compute (from the neural network)
        and return the best moves to make (argmax).
        
        Also returns the score for the best move (amax)
        """
        next_states = self.matris.generate_next_states(state, piece)

        state = np.reshape(self.matris.dict_to_matrix(state), [1, state_size])

        scores = np.zeros(0)
        for next_state in next_states:
            next_state, moves  = next_state
            
            score = self.Q(next_state, piece) 
            scores = np.append(scores, score)
            
        best_state = np.argmax(np.array(scores))
        
        """
        execute the actions to reach
        the best state
        """
        
        moves = next_states[best_state][1]
        
        return (moves, scores[best_state])

    def Q_max_moves(self, state, piece):
        """
        Same as Q_max but deals with raw actions rather
        than final piece locations
        
        Returns the index of the highest value action
        
        This will return None if no moves can be made.
        """
        scores = self.Q(state, piece)
        try:
            max_index = np.argmax(np.array(scores))
            return max_index, scores[0][max_index]
        except:
            """
            gameover state
            """
            return None, None
        

    def execute_moves(self, moves):
        """
        Moves is a tuple of (deltaX_movement, rotation_number)
        After the moves are executed the block is automatically
        hard dropped.
        """
        self.matris.execute_moves(moves)
    

    def remember(self, state, tetromino, action, reward, next_state, done):
        self.memory.append((state, tetromino, action, reward, next_state, done))
        
    def act(self):
        """
        Play one iteration
        """
        state = self.matris.matrix
        #state = self.matris.dict_to_matrix(np.reshape(state, [1, state_size]))
        tetromino = self.matris.current_tetromino
        k = np.random.rand()
        if k <= self.epsilon / 10:
            for action in range(7):
                if self.matris.holes(self.matris.get_harddrop_state(4, [0,x])) is not None:
                    #True
                    continue

        elif k <= self.epsilon:
            action = np.random.randint(0, 7)
        else:
            action, score = self.Q_max_moves(state, tetromino)
        if action == None:
            print("WARNING: Deep Q could not find a move (probably because the tetris game board is invalid.")
            return True
        
        original_score = self.matris.lines
        self.matris.execute_move_index(action)
        reward = 0.
        reward = (self.matris.lines - original_score) ** 2
        new_state = np.reshape(self.matris.dict_to_matrix(self.matris.matrix), [1, state_size])

        if not self.matris.blend():
            done = True
            return True
        else:
            done = False
        if reward > 0.0:
            reward -= self.matris.emptiness() + self.matris.holes(self.matris.get_harddrop_state()) * 0.2
        else:
            reward = -self.matris.emptiness(self.matris.get_harddrop_state()) - self.matris.holes(self.matris.get_harddrop_state()) * 0.2

        state = np.reshape(self.matris.dict_to_matrix(state), [1, state_size])
        self.remember(state, tetromino, action, reward, new_state, done)


    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

if __name__ == "__main__":
    #env = gym.make('tictac4-v0')
    state_size = 220
    action_size = 7
    agent = DQNAgent(state_size, action_size)
    game = matris.Game()
    try:
        agent.load("./tetris.h5")
    except:
        True
    done = False
    batch_size = 32
    EPISODES = 5000

    def callback(matris):    
        agent.matris = matris
        done = agent.act()

    for e in range(EPISODES):
        done = game.main(matris.screen, callback) 
        if done[0]:
            print("episode: {}/{}, score: {}, e: {:.2}"
                    .format(e, EPISODES, done[1], agent.epsilon))
            #temp[e % 50] = time
            
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
            if e % 10 == 0:
                print("Saving:")
                agent.save("./tetris.h5")

