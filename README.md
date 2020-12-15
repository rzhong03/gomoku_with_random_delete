Gomoku with random delete pieces
=====
# How to play
## 1.Start
```python
python game.py <size> <win_number>
```
*Explanation: \
size: Board size\
win_number: The player who firstly form <win_number> consecutive pieces wins.*


exp: 
```python
python game.py 8 5
```
**Note**: It is recommended to use 5 for the *win_number* parameter when using AI to play because the evaluation function has not been designed for other sizes.

## 2. Choose game model
```python
Please choose game mode: 5
```

## 3. Choose starting player
```python
Please choose starting player: 1
```

## 4.Move
Input the coordinate in the terminal, like:
```python
Move: 2 3
```
### 4.1 Human vs. AI
If you choose these two kinds of modes, you just need to input your move cause the AI could automatically choose the point using different AI algorithms.
### 4.2 AI (Tree-based) vs. AI (Random)
If you choose this kind of model, you do not need to input any move cause the AI could battle with each other automatically and print the winner into the terminal.
### 4.3 AI (Tree-based) vs. AI (CNN)
If you choose this kind of model, you do not need to input any move cause the AI could battle with each other automatically and print the winner into the terminal.

# How to train
## Run trainer
```python
python trainer.py
```
**Note**: You need install Pytorch and your computer need to have Cuda GPU.
## Choose the traning size
```python
Please input the training size: 6
```
The result will save to this folder automatically.
